# Utility functions for lightweight data cleaning agent

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.output_parsers import BaseOutputParser

logger = logging.getLogger(__name__)


# Synthetic stable row id injected by the Streamlit app (see ``preview_helpers.AGENT_ROW_ID``).
# Keep identical; omit from user-facing cleaning plans.
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


def normalize_cleaning_column_name(name: str) -> str:
    """
    Match pipeline step 2 in ``prompts/data_cleaning_code_only.md``: lowercase, strip,
    replace non-alphanumeric runs with a single underscore.
    """
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _coerce_plan_columns_to_rows(columns: Any) -> list[dict[str, Any]]:
    """
    Normalize ``plan['columns']`` to a list of dict rows for sanitization.

    The LLM sometimes emits a JSON object mapping column name -> actions instead
    of an array of ``{ "name", "actions" }`` objects; that path must still be filtered.
    """
    if isinstance(columns, list):
        return [r for r in columns if isinstance(r, dict)]
    if isinstance(columns, dict):
        rows: list[dict[str, Any]] = []
        for key, val in columns.items():
            if key is None:
                continue
            k = str(key).strip()
            if not k:
                continue
            if isinstance(val, list):
                actions = [str(x) for x in val]
            elif val is None:
                actions = []
            else:
                actions = [str(val)]
            rows.append({"name": k, "actions": actions})
        return rows
    return []


def coerce_cleaning_plan_columns(columns: Any) -> list[dict[str, Any]]:
    """
    Normalize ``plan['columns']`` to a list of ``dict`` rows for diffing or editing.

    Accepts the same shapes as :func:`sanitize_cleaning_plan` (list of dicts or
    name-to-actions mapping dict).
    """
    return _coerce_plan_columns_to_rows(columns)


def merged_plan_actions_by_column(columns: Any) -> dict[str, list[str]]:
    """
    Concatenate ``actions`` for all plan rows that share the same stripped ``name``.

    The model often emits several ``{"name": "<col>", "actions": [...]}`` rows for
    one column (e.g. early transforms, then a separate imputation or drop row).
    Merging preserves step order and keeps the UI diff/snapshot logic consistent
    with multiset helpers that already aggregate by column name.

    Rows whose ``name`` is empty after stripping are skipped.

    Parameters
    ----------
    columns
        ``plan['columns']`` in any shape accepted by :func:`coerce_cleaning_plan_columns`.

    Returns
    -------
    dict[str, list[str]]
        Map stripped column name -> ordered action strings. Key order is first
        appearance of each name in the coerced row list (Python 3.7+ insertion order).
    """
    merged: dict[str, list[str]] = {}
    for row in coerce_cleaning_plan_columns(columns):
        nm = str(row.get("name", "")).strip()
        if not nm:
            continue
        raw = row.get("actions")
        if isinstance(raw, list):
            acts = [str(x) for x in raw]
        elif raw is None:
            acts = []
        else:
            acts = [str(raw)]
        merged.setdefault(nm, []).extend(acts)
    return merged


def sanitize_cleaning_plan(
    plan: dict[str, Any] | None, df: pd.DataFrame
) -> dict[str, Any] | None:
    """
    Drop ``columns`` entries whose ``name`` is not an input column label or
    its step-2 normalized form.

    Also drops the app-injected synthetic row-id column from the plan (it is
    present in ``df`` for alignment but should not appear in the user-facing
    summary).

    Appends a short note when rows are removed so the UI can explain pruning.

    Parameters
    ----------
    plan : dict or None
        Parsed cleaning plan (``columns``, ``row_ops``, ``notes``).
    df : pd.DataFrame
        The ``source_df`` frame the plan was generated for (same columns as summary).

    Returns
    -------
    dict or None
        A shallow-copied plan with filtered ``columns``, or None if ``plan`` is None.
    """
    if plan is None or not isinstance(plan, dict):
        return plan
    raw_cols = list(df.columns)
    norms = {normalize_cleaning_column_name(c) for c in raw_cols}

    synth_norm = normalize_cleaning_column_name(APP_SYNTHETIC_ALIGN_ROW_ID_COLUMN)

    cols_in = _coerce_plan_columns_to_rows(plan.get("columns"))

    def _is_synthetic_row_id(name: str) -> bool:
        n = str(name).strip()
        if n == APP_SYNTHETIC_ALIGN_ROW_ID_COLUMN:
            return True
        return normalize_cleaning_column_name(n) == synth_norm

    def _row_allowed(name: str) -> bool:
        """Allow only names that match an input column (after step-2 normalization)."""
        norm_label = normalize_cleaning_column_name(name)
        if not norm_label:
            return False
        return norm_label in norms

    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    dropped_synth: list[str] = []
    for row in cols_in:
        if not isinstance(row, dict):
            continue
        nm = row.get("name")
        if nm is None:
            continue
        label = str(nm).strip()
        if _is_synthetic_row_id(label):
            dropped_synth.append(label)
            continue
        if _row_allowed(label):
            kept.append(row)
        else:
            dropped.append(label)

    if dropped:
        logger.info(
            "Removed cleaning plan column rows not in dataset: %s",
            sorted(set(dropped)),
        )
    if dropped_synth:
        logger.debug(
            "Removed synthetic row-id column from cleaning plan: %s",
            sorted(set(dropped_synth)),
        )

    out = dict(plan)
    out["columns"] = kept
    note_parts: list[str] = []
    if dropped:
        note_parts.append(
            "Plan rows removed (not in input columns): "
            f"{', '.join(sorted(set(dropped)))}."
        )
    if note_parts:
        prev = str(out.get("notes") or "").strip()
        tag = " ".join(note_parts)
        out["notes"] = f"{tag} {prev}".strip() if prev else tag

    return out


def run_cleaner_code_on_dataframe(
    code: str,
    df: pd.DataFrame,
    *,
    function_name: str = "data_cleaner",
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Execute generated cleaner code once on ``df``.

    Parameters
    ----------
    code
        Full Python source defining ``function_name``.
    df
        Input frame (same contract as ``execute_stored_cleaning``).
    function_name
        Name of the callable to invoke from ``code``.

    Returns
    -------
    tuple
        ``(cleaned_df, None)`` on success, or ``(None, error_message)`` on failure.
        Never raises: ``exec``/syntax errors and missing cleaner functions are
        returned as ``(None, message)``.
    """
    state = {
        "source_df": df.to_dict(),
        "data_cleaner_function": code,
        "data_cleaner_function_name": function_name,
    }
    try:
        out = execute_agent_code(
            state=state,
            data_key="source_df",
            result_key="data_cleaned",
            error_key="data_cleaner_error",
            code_snippet_key="data_cleaner_function",
            agent_function_name=function_name,
        )
    except Exception as exc:
        return None, str(exc)
    err = out.get("data_cleaner_error")
    if err:
        return None, str(err)
    raw = out.get("data_cleaned")
    if raw is None:
        return None, "cleaner returned no result"
    return pd.DataFrame.from_dict(raw), None


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
    Summarize row removals between two frames for plan UI labels.

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
        Keys: ``n_in``, ``n_out``, ``removed_total``, and optionally
        ``removed_all_null_input_user_cols`` (int or None).
    """
    n_in = len(df_before)
    n_out = len(df_after)
    removed_total = n_in - n_out
    result: dict[str, Any] = {
        "n_in": n_in,
        "n_out": n_out,
        "removed_total": removed_total,
        "removed_all_null_input_user_cols": None,
    }
    if row_id_col not in df_before.columns or row_id_col not in df_after.columns:
        return result
    user_cols = [c for c in df_before.columns if c != row_id_col]
    if not user_cols:
        result["removed_all_null_input_user_cols"] = 0
        return result
    try:
        in_ids = set(first_column_as_series(df_before, row_id_col).tolist())
        out_ids = set(first_column_as_series(df_after, row_id_col).tolist())
    except _EXC_ROW_ID_SET_COERCION:
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


def _extract_python_fenced_block(text: str) -> str | None:
    """Return the body of the first `` ```python `` … `` ``` `` region, or ``None``.

    The closing fence must be on its own line (optional surrounding whitespace only).
    This avoids truncating on `` ``` `` that appear inside a line—e.g.
    ``example = "```"``—which breaks naive non-greedy ``(.*?)``` `` matching and
    yields invalid Python (often ``SyntaxError: unmatched ')'`` at a later line).
    """
    start_m = re.search(r"```\s*python\s*", text, flags=re.IGNORECASE)
    if not start_m:
        return None
    pos = start_m.end()
    while pos < len(text) and text[pos] in "\r\n":
        pos += 1
    rest = text[pos:]
    lines = rest.splitlines(keepends=True)
    body_parts: list[str] = []
    for line in lines:
        if re.fullmatch(r"\s*```\s*", line.rstrip("\r\n")):
            return "".join(body_parts).rstrip("\r\n")
        body_parts.append(line)
    # No line-final closing fence (legacy / malformed): fall back to first ``` anywhere.
    legacy = re.search(r"(.*?)```", rest, flags=re.DOTALL)
    if legacy:
        return legacy.group(1).strip()
    return rest.rstrip("\r\n")


class PythonOutputParser(BaseOutputParser):
    """Extract Python code from LLM responses."""

    def parse(self, text: str):
        """Extract code from ```python``` blocks or return text as-is."""
        extracted = _extract_python_fenced_block(text)
        if extracted is not None:
            return extracted.strip()
        return text


def parse_json_plan_block(text: str) -> dict[str, Any] | None:
    """
    Extract and parse the first ```json``` object from ``text``.

    Returns
    -------
    dict or None
        Parsed JSON object, or ``None`` if missing or invalid.
    """
    json_match = re.search(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not json_match:
        return None
    raw_json = json_match.group(1).strip()
    try:
        loaded = json.loads(raw_json)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON in cleaning plan block; treating plan as None.")
        return None
    if not isinstance(loaded, dict):
        logger.warning("Cleaning plan JSON was not an object; treating plan as None.")
        return None
    return loaded


class DataCleaningOutputParser(BaseOutputParser):
    """Extract Python code and optional JSON cleaning plan from LLM responses."""

    @property
    def _type(self) -> str:
        return "data_cleaning_dual_output"

    def parse(self, text: str) -> dict[str, Any]:
        """
        Parse ``text`` into code and an optional structured plan.

        Returns
        -------
        dict
            Keys: ``code`` (str), ``cleaning_plan`` (dict or None if missing/invalid).
        """
        extracted = _extract_python_fenced_block(text)
        if extracted is not None:
            code = extracted.strip()
        else:
            code = text.strip()

        plan: dict[str, Any] | None = parse_json_plan_block(text)

        return {"code": code, "cleaning_plan": plan}


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
    when at least one flag is True). Imputation choices are defined only in the
    pipeline prompt and implemented in generated code, not in this summary.

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


def execute_agent_code(
    state, data_key, code_snippet_key, result_key, error_key, agent_function_name
):
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
        raise ValueError(
            f"Function '{agent_function_name}' not found in generated code."
        )

    agent_error = None
    result = None
    try:
        result = agent_function(df)
        if isinstance(result, pd.DataFrame):
            result = result.to_dict()
    except KeyError as e:
        logger.error("Execution failed: %s", e)
        agent_error = (
            "An error occurred during data cleaning: missing column or label "
            f"{str(e)!r}. If this name was removed in pipeline steps 3 or 7 (high "
            "missingness, constant, or all-null), the cleaner must not reference "
            "it in later steps—use only columns still on df after those drops."
        )
    except Exception as e:
        logger.error("Execution failed: %s", e)
        msg = str(e)
        hint = ""
        if "Length of values" in msg and "length of index" in msg.lower():
            hint = (
                " Hint: a column assignment used a short list/array/Series (often "
                "`.unique()`, `.cat.categories`, or `pd.Series(list)` without "
                "`index=df.index`). Assign a scalar, `Series(..., index=df.index)`, "
                "or `df[other_col].copy()` so the RHS has one value per row."
            )
        agent_error = f"An error occurred during data cleaning: {msg}{hint}"

    return {result_key: result, error_key: agent_error}


def fix_agent_code(
    state,
    code_snippet_key,
    error_key,
    llm,
    prompt_template,
    function_name,
    retry_count_key="retry_count",
    output_parser: BaseOutputParser | None = None,
):
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
    output_parser : BaseOutputParser, optional
        Parser for the LLM response. Defaults to :class:`PythonOutputParser`.
        If the parser returns a dict with keys ``code`` and ``cleaning_plan``,
        those are merged into the returned state update.

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

    parser = output_parser or PythonOutputParser()
    response = (llm | parser).invoke(prompt)

    out: dict[str, Any] = {
        error_key: None,
        retry_count_key: state.get(retry_count_key) + 1,
    }
    if isinstance(response, dict) and "code" in response:
        out[code_snippet_key] = response["code"]
        if "cleaning_plan" in response:
            out["cleaning_plan"] = response["cleaning_plan"]
    else:
        out[code_snippet_key] = response

    return out
