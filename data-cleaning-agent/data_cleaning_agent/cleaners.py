"""Reusable pandas cleaning helpers for LLM-generated code."""

from __future__ import annotations

import re
from typing import Hashable, Iterable, cast

import numpy as np
import pandas as pd

# Non-empty tokens aligned with ``data_cleaning.md`` step 5; empty-after-strip uses
# ``treat_blank_as_missing`` instead of a literal ``""`` token here. Tuple form so
# ``DataFrame.isin`` type checkers accept the collection.
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

# Step 5 in ``data_cleaning.md`` includes empty string; :data:`_PLACEHOLDER_TOKENS` omits it
# for :func:`missing_share` blank handling via ``treat_blank_as_missing``.
_DEFAULT_PLACEHOLDER_REPLACE_TOKENS: tuple[str, ...] = ("",) + _PLACEHOLDER_TOKENS


def _target_columns(
    df: pd.DataFrame, cols: Iterable[Hashable] | None
) -> list[Hashable]:
    """Return column labels to use; when ``cols`` is None, use ``df.columns`` order."""
    return list(df.columns if cols is None else cols)


def _string_like_subframe(
    df: pd.DataFrame, cols: Iterable[Hashable] | None
) -> pd.DataFrame:
    """Target-column slice of ``df`` restricted to object / pandas string columns.

    Applies :func:`_target_columns` then ``DataFrame.select_dtypes``. Duplicate
    column labels on ``df`` are not supported; behavior is undefined. Labels not
    present in ``df`` follow ``reindex`` semantics for the intermediate slice.
    """
    work = df.reindex(columns=_target_columns(df, cols))
    return work.select_dtypes(include=["object", "string"])


def strip_strings(
    df: pd.DataFrame,
    cols: Iterable[Hashable] | None = None,
    *,
    exclude: Iterable[Hashable] = (),
) -> pd.DataFrame:
    """Strip leading/trailing whitespace on object/string columns only.

    Matches cleaning prompt step 4: only ``object`` / pandas string dtypes are
    updated with ``.str.strip()``; casing is unchanged. Columns in ``exclude``
    are not modified. ``exclude`` entries that are not column labels are
    ignored.

    Duplicate column labels on ``df`` are not supported; behavior is undefined.

    Parameters
    ----------
    df
        Input frame (not mutated).
    cols
        Columns to consider for stripping; default is all columns. Columns
        outside this list are unchanged. Non-string-like dtypes in the slice
        are skipped.
    exclude
        Column names to leave unchanged (even if string-like and in ``cols``).

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with stripped values where applicable.
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


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip, and replace runs of non-word characters with a single underscore.

    Word characters are those matched by ``\\w`` in Unicode mode (:mod:`re`);
    adjacent underscores are collapsed.
    """
    return df.copy().rename(columns=_normalize_single_column_name, copy=False)


def missing_share(
    df: pd.DataFrame,
    cols: Iterable[Hashable] | None = None,
    *,
    treat_blank_as_missing: bool = True,
) -> pd.Series:
    """Per-column missing fraction in ``[0, 1]``.

    A value counts as missing when it is null (``pd.isna``). For object / pandas
    string columns, values equal to stripped placeholders from the cleaning
    prompt (for example ``N/A``, ``null``, or ``-``) also count. When
    ``treat_blank_as_missing`` is true, a stripped empty string (including
    whitespace-only cells) counts as missing. Stripped values for those checks
    use the same step-4 semantics as :func:`strip_strings` (via
    ``strip_strings`` on the string-like subframe from :func:`_string_like_subframe`).

    Duplicate column labels on ``df`` are not supported; behavior is undefined.

    Parameters
    ----------
    df
        Input frame (read-only; not mutated).
    cols
        Columns to score; default is all columns in order.
    treat_blank_as_missing
        If true, treat stripped-empty string cells as missing on string-like
        columns.

    Returns
    -------
    pandas.Series
        Float shares indexed like ``cols`` (or ``df.columns`` when ``cols`` is
        None). If ``df`` has zero rows, every value is NaN.
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
    """Drop columns whose missing share is at least ``threshold`` (0–1 inclusive).

    Shares come from :func:`missing_share` (default placeholder and blank
    semantics). Columns listed in ``exclude`` are never removed. ``exclude``
    entries that are not column labels on ``df`` are ignored.

    Duplicate column labels on ``df`` are not supported; behavior is undefined.

    Parameters
    ----------
    df
        Input frame (not mutated).
    threshold
        Minimum missing share (inclusive) for a column to be dropped.
    exclude
        Column names to keep regardless of missing share.

    Raises
    ------
    ValueError
        If ``threshold`` is not in ``[0.0, 1.0]``.
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
    """Replace placeholder strings with NaN in object/string columns.

    Matches cleaning prompt step 5: only ``object`` / pandas string dtypes in the
    column slice are considered. Each cell is compared to the placeholder set
    **after** applying :func:`strip_strings` to the string-like subframe (same
    step-4 behavior as elsewhere in this module). Cells whose stripped value is in
    the set become ``numpy.nan``; other cells are left unchanged (including
    original whitespace on non-matching cells). When ``placeholders`` is ``None``,
    uses the default list from the prompt (including empty string after strip).

    Duplicate column labels on ``df`` are not supported; behavior is undefined.
    Column labels in ``cols`` that are missing from ``df`` follow ``reindex``
    semantics for the string-like slice.

    Parameters
    ----------
    df
        Input frame (not mutated).
    placeholders
        Tokens to treat as missing (compared after ``.strip()`` on each token).
        ``None`` uses :data:`_DEFAULT_PLACEHOLDER_REPLACE_TOKENS`. An empty
        iterable performs no replacements.
    cols
        Columns to scan; default is all columns in ``df`` order. Only
        string-like columns in this slice are updated.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with placeholder cells set to NaN where applicable.
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


def coerce_datetime_columns(
    df: pd.DataFrame,
    columns: Iterable[Hashable],
) -> pd.DataFrame:
    """Apply ``pd.to_datetime(..., errors='coerce')`` only to the named columns."""
    ...


def coerce_numeric_columns(
    df: pd.DataFrame,
    columns: Iterable[Hashable],
) -> pd.DataFrame:
    """Coerce named columns to numeric after optional currency/percent stripping."""
    ...


def coerce_bool_columns(
    df: pd.DataFrame,
    columns: Iterable[Hashable],
) -> pd.DataFrame:
    """Map boolean-like tokens in named columns to bool dtype."""
    ...


def drop_constant_columns(
    df: pd.DataFrame,
    *,
    exclude: Iterable[Hashable] = (),
) -> pd.DataFrame:
    """Drop columns with a single unique non-null value."""
    ...


def drop_all_null_columns(
    df: pd.DataFrame,
    *,
    exclude: Iterable[Hashable] = (),
) -> pd.DataFrame:
    """Drop columns that are entirely NA."""
    ...


def impute_numeric_median_or_mean(
    s: pd.Series,
    *,
    skew_threshold: float = 1.0,
) -> pd.Series:
    """Fill numeric NA using median if |skew| > threshold, else mean."""
    ...


def impute_categorical_mode(
    s: pd.Series,
    *,
    dropna_before_mode: bool = True,
) -> pd.Series:
    """Fill NA in string/categorical-like series using the mode when sensible."""
    ...


def safe_assign_series(
    df: pd.DataFrame,
    col: Hashable,
    values: pd.Series,
) -> pd.DataFrame:
    """Assign a like-indexed series to ``df[col]`` without mutating the caller."""
    ...


def reset_index_drop(df: pd.DataFrame) -> pd.DataFrame:
    """``reset_index(drop=True)`` wrapper."""
    ...


def drop_all_null_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are entirely NA across columns."""
    ...


def drop_duplicate_rows(
    df: pd.DataFrame,
    subset: Iterable[Hashable] | None = None,
) -> pd.DataFrame:
    """Drop exact duplicate rows (optionally on a subset of columns)."""
    ...
