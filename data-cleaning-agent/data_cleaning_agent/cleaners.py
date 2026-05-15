"""Reusable pandas cleaning helpers for LLM-generated code."""

from __future__ import annotations

import re
from typing import Hashable, Iterable, cast

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


def _target_columns(
    df: pd.DataFrame, cols: Iterable[Hashable] | None
) -> list[Hashable]:
    """Return column labels to use; when ``cols`` is None, use ``df.columns`` order."""
    return list(df.columns if cols is None else cols)


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
    work = df.reindex(columns=_target_columns(df, cols))
    str_work = work.select_dtypes(include=["object", "string"])
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
    come from :func:`strip_strings` with ``exclude=()`` (same step-4 strip as the
    public helper; builds a full copy of ``df``).

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
    str_only_raw = work.reindex(columns=work.columns).select_dtypes(
        include=["object", "string"],
    )
    str_only = strip_strings(str_only_raw, cols=cols, exclude=())

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
    """Replace common placeholder strings with NaN in object/string columns."""
    ...


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
