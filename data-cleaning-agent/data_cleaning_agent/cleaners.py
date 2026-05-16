"""Reusable pandas cleaning helpers for LLM-generated code.

Unless a function says otherwise, ``df`` is not mutated: helpers return a new
frame or series. Duplicate column labels on ``df`` are not supported; behavior
is undefined.
"""

from __future__ import annotations

import re
from typing import Hashable, Iterable, cast

import numpy as np
import pandas as pd

# Non-empty placeholder tokens for missing detection (compared after strip).
# Literal ``""`` is omitted here; blanks are handled via ``treat_blank_as_missing``
# in :func:`missing_share`. Tuple form so ``DataFrame.isin`` type checkers accept it.
_PLACEHOLDER_TOKENS: tuple[str, ...] = (
    "N/A",
    "n/a",
    "NA",
    "null",
    "NULL",
    "None",
    "?",
    "missing",
    "-",
    "unknown",
)

# Prepends ``""`` to :data:`_PLACEHOLDER_TOKENS` so stripped-empty cells count as
# missing in :func:`missing_share` when ``treat_blank_as_missing`` is true.
_DEFAULT_PLACEHOLDER_REPLACE_TOKENS: tuple[str, ...] = ("",) + _PLACEHOLDER_TOKENS

# Strip ``$``, comma, and ``%`` from string columns before ``pd.to_numeric``.
_NUMERIC_CURRENCY_STRIP_PATTERN = r"[$,%]"

# Casefolded string tokens mapped to True / False for boolean-like coercion.
_TRUE_BOOL_TOKENS: frozenset[str] = frozenset({"yes", "true", "t", "y", "1"})
_FALSE_BOOL_TOKENS: frozenset[str] = frozenset({"no", "false", "f", "n", "0"})


def _target_columns(
    df: pd.DataFrame, cols: Iterable[Hashable] | None
) -> list[Hashable]:
    """Return column labels to use; when ``cols`` is None, use ``df.columns`` order."""
    return list(df.columns if cols is None else cols)


def _string_like_subframe(
    df: pd.DataFrame, cols: Iterable[Hashable] | None
) -> pd.DataFrame:
    """Object / string columns from the target slice."""
    work = df.reindex(columns=_target_columns(df, cols))
    return work.select_dtypes(include=["object", "string"])


def strip_strings(
    df: pd.DataFrame,
    cols: Iterable[Hashable] | None = None,
    *,
    exclude: Iterable[Hashable] = (),
) -> pd.DataFrame:
    """Strip leading/trailing whitespace on object and pandas string columns.

    Only string-like dtypes get ``.str.strip()``; casing unchanged. ``cols``
    limits the scan (``None`` = all columns). Names in ``exclude`` are skipped;
    unknown ``exclude`` labels are ignored.

    Parameters
    ----------
    df
        Frame to copy and update.
    cols
        Columns to scan, or ``None`` for all.
    exclude
        Column names never stripped.

    Returns
    -------
    pandas.DataFrame
        Copy with stripped cells where applicable.
    """
    out = df.copy()
    str_work = _string_like_subframe(df, cols)
    if str_work.shape[1] == 0:
        return out
    stripped = cast(
        pd.DataFrame,
        str_work.astype("string").apply(
            lambda s: s.str.strip(),
            axis=0,
        ),
    )
    eligible = stripped.columns[~stripped.columns.isin(set(exclude))]
    if eligible.shape[0] > 0:
        out[eligible] = stripped[eligible]
    return out


def _normalize_single_column_name(name: Hashable) -> str:
    """Lowercase, strip, map runs of non-``\\w`` characters to a single underscore."""
    s = str(name).strip().lower()
    merged = re.sub(r"[^\w]+", "_", s, flags=re.UNICODE)
    merged = re.sub(r"_+", "_", merged)
    return merged.strip("_")


def normalize_column_label(name: Hashable) -> str:
    """Normalize a single column label (same rules as ``normalize_column_names``)."""
    return _normalize_single_column_name(name)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip and lowercase column names; map non-word runs to a single underscore (Unicode ``\\w``)."""
    return df.copy().rename(columns=_normalize_single_column_name, copy=False)


def missing_share(
    df: pd.DataFrame,
    cols: Iterable[Hashable] | None = None,
    *,
    treat_blank_as_missing: bool = True,
) -> pd.Series:
    """Per-column missing fraction in ``[0, 1]``.

    Uses ``pd.isna`` on all dtypes; on string-like columns also counts stripped
    :data:`_PLACEHOLDER_TOKENS` and, if ``treat_blank_as_missing``, stripped empty
    strings (via :func:`strip_strings` on the string slice). **Zero-row** frames
    yield all-NaN shares.

    Parameters
    ----------
    df
        Frame to score.
    cols
        Columns to score, or ``None`` for all in order.
    treat_blank_as_missing
        Treat stripped-empty strings as missing on string-like columns.

    Returns
    -------
    pandas.Series
        Float shares indexed like the scored columns.
    """
    target_cols = _target_columns(df, cols)
    work = df.reindex(columns=target_cols)
    if len(work) == 0:
        return pd.Series({c: float("nan") for c in target_cols}, dtype="float64")

    mask = work.isna()
    str_only_raw = _string_like_subframe(df, cols)
    str_only = strip_strings(str_only_raw)

    if str_only.shape[1] > 0:
        str_bad = str_only.isin(_PLACEHOLDER_TOKENS) | (
            treat_blank_as_missing & str_only.eq("")
        )
        mask = mask | str_bad.reindex(columns=work.columns, fill_value=False)

    return cast(pd.Series, mask.mean(axis=0))


def drop_columns_by_missing(
    df: pd.DataFrame,
    threshold: float,
    *,
    exclude: Iterable[Hashable] = (),
) -> pd.DataFrame:
    """Drop columns whose :func:`missing_share` is at least ``threshold`` (0–1 inclusive).

    Names in ``exclude`` are kept; unknown ``exclude`` labels are ignored.

    Parameters
    ----------
    df
        Frame to drop columns from.
    threshold
        Inclusive minimum missing share for a column to be dropped.
    exclude
        Column names never dropped.

    Raises
    ------
    ValueError
        If ``threshold`` is not in ``[0.0, 1.0]``.

    Returns
    -------
    pandas.DataFrame
        New frame without dropped columns.
    """
    if not 0.0 <= threshold <= 1.0:
        msg = "threshold must be between 0.0 and 1.0 inclusive"
        raise ValueError(msg)

    shares = missing_share(df)
    exclude_set = set(exclude)
    drop_mask = (shares >= threshold) & ~shares.index.isin(exclude_set)
    to_drop = shares.loc[drop_mask].index.tolist()
    return df.drop(columns=to_drop)


def replace_placeholders_with_na(
    df: pd.DataFrame,
    placeholders: Iterable[str] | None = None,
    cols: Iterable[Hashable] | None = None,
) -> pd.DataFrame:
    """Replace placeholder tokens with NaN on object/string columns.

    String columns are first stripped then cells whose stripped value is
    in ``placeholders`` become ``numpy.nan`` (``None`` →
    :data:`_DEFAULT_PLACEHOLDER_REPLACE_TOKENS`). Empty ``placeholders`` is a no-op.

    Parameters
    ----------
    df
        Frame to copy and update.
    placeholders
        Tokens after strip, or ``None`` for the default set.
    cols
        Columns to scan, or ``None`` for all.

    Returns
    -------
    pandas.DataFrame
        Copy with matching cells set to NaN.
    """
    out = df.copy()
    str_work = _string_like_subframe(df, cols)
    if str_work.shape[1] == 0:
        return out

    stripped = strip_strings(str_work)
    if placeholders is None:
        tokens = _DEFAULT_PLACEHOLDER_REPLACE_TOKENS
    else:
        tokens = tuple(p.strip() for p in placeholders)
    if not tokens:
        return out

    for col in stripped.columns:
        matches = stripped[col].isin(tokens)
        if bool(matches.any()):
            out.loc[matches, col] = np.nan
    return out


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return cast(pd.Series, pd.to_numeric(s, errors="coerce"))
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        cleaned = cast(
            pd.Series,
            s.astype("string").str.replace(
                _NUMERIC_CURRENCY_STRIP_PATTERN,
                "",
                regex=True,
            ),
        )
        return cast(pd.Series, pd.to_numeric(cleaned, errors="coerce"))
    return s


def _series_to_nullable_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return cast(pd.Series, s.astype("boolean"))
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return s
    strv = cast(
        pd.Series,
        s.astype("string").str.strip().str.casefold(),
    )
    true_m = strv.isin(_TRUE_BOOL_TOKENS)
    false_m = strv.isin(_FALSE_BOOL_TOKENS)
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    return out.mask(true_m, True).mask(false_m, False)


def _to_datetime_mixed(s: pd.Series) -> pd.Series:
    return cast(
        pd.Series,
        pd.to_datetime(s, errors="coerce", format="mixed"),
    )


def coerce_datetime_columns(
    df: pd.DataFrame,
    columns: Iterable[Hashable],
) -> pd.DataFrame:
    """Parse named columns with ``pd.to_datetime(..., errors='coerce', format='mixed')``.

    Unparseable values become ``NaT``. Unknown ``columns`` names are ignored.

    Parameters
    ----------
    df
        Frame to copy and update.
    columns
        Names to coerce if present on ``df``.

    Returns
    -------
    pandas.DataFrame
        Copy with coerced datetime columns where applicable.
    """
    out = df.copy()
    present = out.columns.intersection(columns)
    if present.empty:
        return out
    out[present] = out[present].apply(_to_datetime_mixed, axis=0)
    return out


def coerce_numeric_columns(
    df: pd.DataFrame,
    columns: Iterable[Hashable],
) -> pd.DataFrame:
    """Coerce named columns to numeric.

    Object/string columns: strip :data:`_NUMERIC_CURRENCY_STRIP_PATTERN` then
    ``pd.to_numeric(..., errors='coerce')``. Already-numeric columns get
    ``to_numeric`` only. Other dtypes unchanged. Unknown ``columns`` names ignored.

    Parameters
    ----------
    df
        Frame to copy and update.
    columns
        Names to coerce if present on ``df``.

    Returns
    -------
    pandas.DataFrame
        Copy with coerced numeric columns where applicable.
    """
    out = df.copy()
    present = out.columns.intersection(columns)
    if present.empty:
        return out
    out[present] = out[present].apply(_coerce_numeric_series, axis=0)
    return out


def coerce_bool_columns(
    df: pd.DataFrame,
    columns: Iterable[Hashable],
) -> pd.DataFrame:
    """Map stripped, casefolded tokens to nullable ``boolean``.

    :data:`_TRUE_BOOL_TOKENS` / :data:`_FALSE_BOOL_TOKENS` → True/False; other
    non-null strings → ``pd.NA``. Existing bool columns become nullable boolean;
    non-string, non-bool dtypes unchanged. Unknown ``columns`` names ignored.

    Parameters
    ----------
    df
        Frame to copy and update.
    columns
        Names to coerce if present on ``df``.

    Returns
    -------
    pandas.DataFrame
        Copy with coerced boolean columns where applicable.
    """
    out = df.copy()
    present = out.columns.intersection(columns)
    if present.empty:
        return out
    out[present] = out[present].apply(_series_to_nullable_bool, axis=0)
    return out


def drop_constant_columns(
    df: pd.DataFrame,
    *,
    exclude: Iterable[Hashable] = (),
) -> pd.DataFrame:
    """Drop columns with exactly one distinct non-null value (``nunique(dropna=True) == 1``).

    All-NA columns (``nunique == 0``) are left unchanged; use :func:`drop_all_null_columns`.
    Names in ``exclude`` are kept; unknown ``exclude`` labels ignored.

    Parameters
    ----------
    df
        Frame to drop columns from.
    exclude
        Column names never dropped.

    Returns
    -------
    pandas.DataFrame
        New frame without dropped columns.
    """
    nu = df.nunique(dropna=True)
    is_constant = nu.eq(1)
    drop_mask = is_constant & ~is_constant.index.isin(exclude)
    to_drop = drop_mask[drop_mask].index.tolist()
    return df.drop(columns=to_drop)


def drop_all_null_columns(
    df: pd.DataFrame,
    *,
    exclude: Iterable[Hashable] = (),
) -> pd.DataFrame:
    """Drop columns that are NA in every row.

    **Zero rows:** returns a copy with no columns dropped (avoids vacuous ``all``).
    Names in ``exclude`` are kept; unknown ``exclude`` labels ignored.

    Parameters
    ----------
    df
        Frame to drop columns from.
    exclude
        Column names never dropped.

    Returns
    -------
    pandas.DataFrame
        New frame without dropped columns.
    """
    if len(df) == 0:
        return df.copy()
    all_na = cast(pd.Series, df.isna().all(axis=0))
    drop_mask = all_na & ~all_na.index.isin(exclude)
    to_drop = drop_mask[drop_mask].index.tolist()
    return df.drop(columns=to_drop)


def impute_numeric_median_or_mean(
    s: pd.Series,
    *,
    skew_threshold: float = 1.0,
) -> pd.Series:
    """Fill numeric NA using median if |skew| > threshold, else mean."""
    if not pd.api.types.is_numeric_dtype(s):
        return s.copy()
    out = s.copy()
    if len(out) == 0 or not out.notna().any():
        return out
    skew = float(out.skew(skipna=True, numeric_only=True))

    use_median = np.isfinite(skew) and abs(skew) > skew_threshold
    stat = (
        out.median(skipna=True, numeric_only=True)
        if use_median
        else out.mean(skipna=True, numeric_only=True)
    )
    stat_f = float(stat)

    if not np.isfinite(stat_f):
        return out
    return out.fillna(stat)


def impute_categorical_mode(s: pd.Series) -> pd.Series:
    """Fill NA in string/categorical-like series using the mode when sensible."""
    if not (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    ):
        return s.copy()
    out = s.copy()
    modes = out.mode(dropna=True)
    if modes.empty:
        return out
    fill_value = modes.iloc[0]
    return out.fillna(fill_value)


def drop_all_null_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are entirely NA across columns.

    Returns
    -------
    pandas.DataFrame
        New frame without all-null rows.
    """
    return df.dropna(how="all")


def drop_duplicate_rows(
    df: pd.DataFrame,
    subset: Iterable[Hashable] | None = None,
) -> pd.DataFrame:
    """Drop exact duplicate rows, keeping the first occurrence.

    Parameters
    ----------
    df
        Frame to deduplicate.
    subset
        Columns to compare, or ``None`` for all columns. Unknown labels are
        ignored; if none match, returns a copy unchanged.

    Returns
    -------
    pandas.DataFrame
        New frame with duplicate rows removed.
    """
    if subset is None:
        return df.drop_duplicates()
    present = df.columns.intersection(subset)
    if present.empty:
        return df.copy()
    return df.drop_duplicates(subset=list(present))
