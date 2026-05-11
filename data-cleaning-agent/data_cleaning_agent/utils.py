# Utility functions for lightweight data cleaning agent

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.output_parsers import BaseOutputParser

logger = logging.getLogger(__name__)

# Casefolded tokens that boolean-like detection accepts as members of a binary
# categorical (e.g. "Yes"/"No", "true"/"false", "T"/"F", "0"/"1").
_BOOL_LIKE_TOKENS = frozenset({
    "yes", "no", "true", "false", "t", "f", "y", "n", "0", "1",
})


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
    """Per-column summary used by the cleaning agent prompt."""

    name: str
    dtype: str
    missing_pct: float
    cardinality: int
    sample_values: list[Any]
    numeric_stats: NumericStats | None
    # List of {"value": str, "pct": float} entries, top 3 by frequency.
    top_categories: list[dict] | None
    id_like: bool
    looks_date_like: bool
    looks_numeric_string_like: bool
    looks_boolean_like: bool


@dataclass(frozen=True)
class DataFrameSummary:
    """Structured summary of a DataFrame, one ColumnSummary per column."""

    n_rows: int
    n_cols: int
    columns: dict[str, ColumnSummary]


class PythonOutputParser(BaseOutputParser):
    """Extract Python code from LLM responses."""

    def parse(self, text: str):
        """Extract code from ```python``` blocks or return text as-is."""
        python_code_match = re.search(r'```python(.*?)```', text, re.DOTALL)
        if python_code_match:
            return python_code_match.group(1).strip()
        return text


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
    skew = series.skew()
    if pd.isna(skew):
        skew = 0.0
    std = series.std()
    if pd.isna(std):
        std = 0.0
    return NumericStats(
        min=round(float(series.min()), 4),
        max=round(float(series.max()), 4),
        mean=round(float(series.mean()), 4),
        median=round(float(series.median()), 4),
        std=round(float(std), 4),
        skew=round(float(skew), 4),
    )


def _top_categories(series: pd.Series, n: int = 3) -> list[dict]:
    """Return top-N values with frequency percentages (0-100)."""
    non_null = series.dropna()
    if non_null.empty:
        return []
    counts = non_null.value_counts(normalize=True).head(n)
    return [{"value": str(v), "pct": round(float(p) * 100, 2)} for v, p in counts.items()]


def _detect_id_like(series: pd.Series, n_rows: int, cardinality: int) -> bool:
    """Cardinality must equal n_rows AND (name ends with id/uuid OR strictly increasing int)."""
    if cardinality != n_rows or n_rows == 0:
        return False
    if str(series.name).lower().endswith(("id", "uuid")):
        return True
    if pd.api.types.is_integer_dtype(series):
        return bool(series.is_monotonic_increasing and series.is_unique)
    return False


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
    parsed = pd.to_numeric(cleaned, errors="coerce")
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
    # is_string_dtype covers both legacy object-with-strings and pandas 3.x StringDtype.
    is_string = pd.api.types.is_string_dtype(series) and not is_numeric

    return ColumnSummary(
        name=name,
        dtype=str(series.dtype),
        missing_pct=round(float(series.isna().mean()) * 100, 2) if n_rows else 0.0,
        cardinality=cardinality,
        sample_values=_sample_values(series),
        numeric_stats=_numeric_stats(series) if is_numeric else None,
        top_categories=(
            _top_categories(series) if not is_numeric and 0 < cardinality <= 20 else None
        ),
        id_like=_detect_id_like(series, n_rows, cardinality),
        looks_date_like=_detect_date_like(series) if is_string else False,
        looks_numeric_string_like=_detect_numeric_string_like(series) if is_string else False,
        looks_boolean_like=_detect_boolean_like(series, cardinality) if is_string else False,
    )


def get_dataframe_summary(df: pd.DataFrame) -> DataFrameSummary:
    """
    Build a structured summary of a DataFrame for the cleaning-agent prompt.

    Captures per-column dtype, missingness, cardinality, sample values, numeric
    stats (when numeric), top categories (when low-cardinality non-numeric), and
    detection flags for ID-likeness, date-like strings, numeric-string-like
    values, and boolean-like values.

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
    columns = {name: _summarize_column(name, df[name], n_rows) for name in df.columns}
    return DataFrameSummary(n_rows=n_rows, n_cols=len(df.columns), columns=columns)


def format_dataframe_summary(summary: DataFrameSummary) -> str:
    """
    Render a ``DataFrameSummary`` into a deterministic, LLM-friendly text block.

    Produces the string interpolated into the ``{all_datasets_summary}`` slot of
    the cleaning prompt. Lines are emitted only when relevant (numeric stats
    only for numeric columns, top categories only when populated, detection only
    when at least one flag is True).

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
            cats = ", ".join(f"{c['value']} ({c['pct']:.1f}%)" for c in col.top_categories)
            lines.append(f"  top categories: {cats}")
        if col.id_like:
            lines.append("  id_like: True")
        if col.looks_date_like or col.looks_numeric_string_like or col.looks_boolean_like:
            lines.append(
                f"  detection: date_like={col.looks_date_like}, "
                f"numeric_string_like={col.looks_numeric_string_like}, "
                f"boolean_like={col.looks_boolean_like}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def execute_agent_code(state, data_key, code_snippet_key, result_key, error_key, agent_function_name):
    """
    Execute the generated agent code on the data.
    
    Parameters
    ----------
    state : dict
        The current state containing data and code.
    data_key : str
        Key in state where the input data is stored.
    code_snippet_key : str
        Key in state where the generated code is stored.
    result_key : str
        Key to store the result in.
    error_key : str
        Key to store any error message in.
    agent_function_name : str
        Name of the function to execute from the generated code.
    
    Returns
    -------
    dict
        Dictionary with result and error keys.
    """
    logger.info("Executing agent code")
    
    data = state.get(data_key)
    agent_code = state.get(code_snippet_key)
    df = pd.DataFrame.from_dict(data)
    
    # exec() runs LLM-generated code in an isolated namespace; only use with trusted models.
    local_vars = {}
    global_vars = {}
    exec(agent_code, global_vars, local_vars)
    
    agent_function = local_vars.get(agent_function_name)
    if not agent_function or not callable(agent_function):
        raise ValueError(f"Function '{agent_function_name}' not found in generated code.")
    
    agent_error = None
    result = None
    try:
        result = agent_function(df)
        if isinstance(result, pd.DataFrame):
            result = result.to_dict()
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        agent_error = f"An error occurred during data cleaning: {str(e)}"
    
    return {result_key: result, error_key: agent_error}


def fix_agent_code(state, code_snippet_key, error_key, llm, prompt_template, function_name, retry_count_key="retry_count"):
    """
    Fix errors in the generated agent code using the LLM.
    
    Parameters
    ----------
    state : dict
        The current state containing code and error information.
    code_snippet_key : str
        Key in state where the broken code is stored.
    error_key : str
        Key in state where the error message is stored.
    llm : LLM
        The language model to use for fixing the code.
    prompt_template : str
        Template for the fix prompt (should have {code_snippet}, {error}, {function_name} placeholders).
    function_name : str
        Name of the function being fixed.
    retry_count_key : str, optional
        Key in state for tracking retry count. Defaults to "retry_count".
    
    Returns
    -------
    dict
        Dictionary with updated code, cleared error, and incremented retry count.
    """
    logger.info("Fixing agent code")
    logger.debug(f"Retry count: {state.get(retry_count_key)}")
    
    code_snippet = state.get(code_snippet_key)
    error_message = state.get(error_key)
    
    prompt = prompt_template.format(
        code_snippet=code_snippet,
        error=error_message,
        function_name=function_name,
    )
    
    response = (llm | PythonOutputParser()).invoke(prompt)
    
    return {
        code_snippet_key: response,
        error_key: None,
        retry_count_key: state.get(retry_count_key) + 1
    }
