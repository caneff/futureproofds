# Utility functions for lightweight data cleaning agent

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Synthetic stable row id injected by the Streamlit app (see ``preview_helpers.AGENT_ROW_ID``).
APP_SYNTHETIC_ALIGN_ROW_ID_COLUMN = "__agent_row_id__"


# int()/tolist() when building row-id sets — use a bound tuple so formatters cannot
# rewrite ``except (TypeError, ValueError)`` into the comma form (wrong semantics).
_EXC_ROW_ID_SET_COERCION = (TypeError, ValueError)

# Casefolded tokens that boolean-like detection accepts as members of a binary
# categorical (e.g. "Yes"/"No", "true"/"false", "T"/"F", "0"/"1").
_BOOL_LIKE_TOKENS = frozenset({
    "yes",
    "no",
    "true",
    "false",
    "t",
    "f",
    "y",
    "n",
    "0",
    "1",
})


def first_column_as_series(df: pd.DataFrame, name: str) -> pd.Series:
    """Return ``df[name]`` as a single Series even when column labels are duplicated.

    Duplicate labels make ``df[name]`` a DataFrame; boolean ops on that object
    then propagate ambiguous truth-value errors.
    """
    sel = df[name]
    if isinstance(sel, pd.DataFrame):
        if sel.shape[1] == 0:
            return pd.Series(dtype=object)
        return sel.iloc[:, 0]
    return sel


def summarize_cleaning_row_effects(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    *,
    row_id_col: str = APP_SYNTHETIC_ALIGN_ROW_ID_COLUMN,
) -> dict[str, Any]:
    """
    Summarize row removals between two frames (e.g. for outcome summaries).

    Uses ``row_id_col`` when present in both frames to count removed row ids and
    how many removed rows were all-null on non-id columns in ``df_before``.

    Parameters
    ----------
    df_before
        Frame passed into the cleaner (includes synthetic row id when used).
    df_after
        Returned cleaned frame.
    row_id_col
        Stable row identifier column for alignment.

    Returns
    -------
    dict
        Keys: ``n_in``, ``n_out``, ``removed_total``, ``rows_removed_by_id``,
        ``rows_added_by_id`` (id counts are ``None`` when ``row_id_col`` is absent
        in either frame), and optionally ``removed_all_null_input_user_cols``.
    """
    n_in = len(df_before)
    n_out = len(df_after)
    result: dict[str, Any] = {
        "n_in": n_in,
        "n_out": n_out,
        "removed_total": n_in - n_out,
        "rows_removed_by_id": None,
        "rows_added_by_id": None,
        "removed_all_null_input_user_cols": None,
    }
    if row_id_col not in df_before.columns or row_id_col not in df_after.columns:
        return result
    try:
        in_ids = set(first_column_as_series(df_before, row_id_col).tolist())
        out_ids = set(first_column_as_series(df_after, row_id_col).tolist())
    except _EXC_ROW_ID_SET_COERCION:
        return result
    result["rows_removed_by_id"] = len(in_ids - out_ids)
    result["rows_added_by_id"] = len(out_ids - in_ids)
    user_cols = [c for c in df_before.columns if c != row_id_col]
    if not user_cols:
        result["removed_all_null_input_user_cols"] = 0
        return result
    dropped_ids = list(in_ids - out_ids)
    dropped_mask = first_column_as_series(df_before, row_id_col).isin(dropped_ids)
    if int(dropped_mask.sum()) == 0:
        result["removed_all_null_input_user_cols"] = 0
        return result
    all_null_input = df_before.loc[dropped_mask, user_cols].isna().all(axis=1)
    result["removed_all_null_input_user_cols"] = int(all_null_input.sum())
    return result


@dataclass(frozen=True)
class NumericStats:
    """Summary statistics for a numeric column."""

    min: float
    max: float
    mean: float
    median: float
    std: float
    # pandas Series.skew() returns NaN for fewer than 3 non-null values; we
    # coerce to 0.0 so this field is always a finite float for downstream use.
    skew: float


@dataclass(frozen=True)
class ColumnSummary:
    """Per-column summary used by the cleaning agent prompt.

    Attributes
    ----------
    missing_pct
        Percent of rows with ``pandas`` missing values only (``isna()`` on the
        series as loaded).
    """

    name: str
    dtype: str
    missing_pct: float
    cardinality: int
    sample_values: list[Any]
    numeric_stats: NumericStats | None
    # List of {"value": str, "pct": float} entries, top 3 by frequency.
    top_categories: list[dict] | None
    looks_date_like: bool
    looks_numeric_string_like: bool
    looks_boolean_like: bool


@dataclass(frozen=True)
class DataFrameSummary:
    """Structured summary of a DataFrame, one ColumnSummary per column."""

    n_rows: int
    n_cols: int
    columns: dict[str, ColumnSummary]


def _sample_values(series: pd.Series, n: int = 3) -> list[Any]:
    """Return the first ``n`` unique non-null values, coerced to JSON-friendly types."""
    unique = series.dropna().unique()[:n]
    out: list[Any] = []
    for v in unique:
        if isinstance(v, np.generic):
            out.append(v.item())
        elif isinstance(v, pd.Timestamp):
            out.append(v.isoformat())
        else:
            out.append(v)
    return out


def _numeric_stats(series: pd.Series) -> NumericStats:
    """Compute summary stats for a numeric column. Skew is coerced to 0.0 when undefined."""
    skew = float(series.skew())
    if pd.isna(skew):
        skew = 0.0
    std = float(series.std())
    if pd.isna(std):
        std = 0.0
    return NumericStats(
        min=round(float(series.min()), 4),
        max=round(float(series.max()), 4),
        mean=round(float(series.mean()), 4),
        median=round(float(series.median()), 4),
        std=round(std, 4),
        skew=round(skew, 4),
    )


def _top_categories(series: pd.Series, n: int = 3) -> list[dict]:
    """Return top-N values with frequency percentages (0-100)."""
    non_null = series.dropna()
    if non_null.empty:
        return []
    counts = non_null.value_counts(normalize=True).head(n)
    return [
        {"value": str(v), "pct": round(float(p) * 100, 2)} for v, p in counts.items()
    ]


def _detect_date_like(series: pd.Series) -> bool:
    """At least 90% of non-null values parse as datetime."""
    non_null = series.dropna()
    if non_null.empty:
        return False
    # format="mixed" lets pandas accept heterogeneous date strings without the
    # noisy "could not infer format" warning emitted in pandas 2.x+.
    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= 0.9)


def _detect_numeric_string_like(series: pd.Series) -> bool:
    """At least 90% of non-null values parse as numeric after stripping currency/percent/whitespace."""
    non_null = series.dropna()
    if non_null.empty:
        return False
    cleaned = non_null.astype(str).str.replace(r"[$,%\s]", "", regex=True)
    parsed = pd.Series(pd.to_numeric(cleaned, errors="coerce"))
    return bool(parsed.notna().mean() >= 0.9)


def _detect_boolean_like(series: pd.Series, cardinality: int) -> bool:
    """Cardinality <= 2 AND all casefolded non-null values are in the known boolean tokens."""
    if cardinality == 0 or cardinality > 2:
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    values = {str(v).strip().casefold() for v in non_null.unique()}
    return values.issubset(_BOOL_LIKE_TOKENS)


def _summarize_column(name: str, series: pd.Series, n_rows: int) -> ColumnSummary:
    """Build a ColumnSummary for a single DataFrame column."""
    cardinality = int(series.nunique(dropna=True))
    is_numeric = pd.api.types.is_numeric_dtype(series)
    # Include object columns so ISO date strings etc. still get string-like
    # heuristics when pandas does not use StringDtype (common for read_csv).
    is_string = not is_numeric and (
        pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)
    )

    return ColumnSummary(
        name=name,
        dtype=str(series.dtype),
        missing_pct=round(float(series.isna().mean()) * 100, 2) if n_rows else 0.0,
        cardinality=cardinality,
        sample_values=_sample_values(series),
        numeric_stats=_numeric_stats(series) if is_numeric else None,
        top_categories=(
            _top_categories(series)
            if not is_numeric and 0 < cardinality <= 20
            else None
        ),
        looks_date_like=_detect_date_like(series) if is_string else False,
        looks_numeric_string_like=_detect_numeric_string_like(series)
        if is_string
        else False,
        looks_boolean_like=_detect_boolean_like(series, cardinality)
        if is_string
        else False,
    )


def get_dataframe_summary(df: pd.DataFrame) -> DataFrameSummary:
    """
    Build a structured summary of a DataFrame for the cleaning-agent prompt.

    Captures per-column dtype, missingness, cardinality, sample values, numeric
    stats (when numeric), top categories (when low-cardinality non-numeric), and
    detection flags for date-like strings, numeric-string-like values, and
    boolean-like values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to summarize.

    Returns
    -------
    DataFrameSummary
        Structured summary with ``n_rows``, ``n_cols``, and a ``columns`` dict
        mapping column name to ``ColumnSummary`` (preserving input column order).
    """
    n_rows = len(df)
    columns: dict[str, ColumnSummary] = {}
    # Duplicate column labels make ``df[name]`` a DataFrame, not a Series; type
    # checkers and ``_summarize_column`` expect a single Series, so we walk
    # columns in order and take the first duplicate when needed.
    for name in df.columns:
        col = df[name]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        columns[name] = _summarize_column(name, col, n_rows)
    return DataFrameSummary(n_rows=n_rows, n_cols=len(df.columns), columns=columns)


def format_dataframe_summary(summary: DataFrameSummary) -> str:
    """
    Render a ``DataFrameSummary`` into a deterministic, LLM-friendly text block.

    Produces the string interpolated into the ``{all_datasets_summary}`` slot of
    the cleaning prompt. Lines are emitted only when relevant (numeric stats
    only for numeric columns, top categories only when populated, detection only
    when at least one flag is True). Imputation choices are defined in the plan
    prompt and applied by the hybrid pipeline, not in this summary.

    Parameters
    ----------
    summary : DataFrameSummary
        Output of :func:`get_dataframe_summary`.

    Returns
    -------
    str
        Multi-line summary suitable for direct prompt interpolation.
    """
    lines: list[str] = [
        "Dataset Summary",
        "===============",
        f"Rows: {summary.n_rows}",
        f"Columns: {summary.n_cols}",
        "",
        "Per-column details:",
        "",
    ]
    for col in summary.columns.values():
        lines.append(f"- {col.name} ({col.dtype})")
        lines.append(f"  missing: {col.missing_pct:.2f}%")
        lines.append(f"  cardinality: {col.cardinality}")
        lines.append(f"  sample values: {col.sample_values}")
        if col.numeric_stats is not None:
            s = col.numeric_stats
            lines.append(
                f"  numeric stats: min={s.min}, max={s.max}, mean={s.mean:.2f}, "
                f"median={s.median:.2f}, std={s.std:.2f}, skew={s.skew:.2f}"
            )
        if col.top_categories:
            cats = ", ".join(
                f"{c['value']} ({c['pct']:.1f}%)" for c in col.top_categories
            )
            lines.append(f"  top categories: {cats}")
        if (
            col.looks_date_like
            or col.looks_numeric_string_like
            or col.looks_boolean_like
        ):
            lines.append(
                f"  detection: date_like={col.looks_date_like}, "
                f"numeric_string_like={col.looks_numeric_string_like}, "
                f"boolean_like={col.looks_boolean_like}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()
