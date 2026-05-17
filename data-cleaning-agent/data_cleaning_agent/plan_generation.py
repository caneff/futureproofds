"""Generate and parse JSON cleaning plans from an LLM."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from data_cleaning_agent.cleaning_plan import (
    DEFAULT_ROW_ID_COL,
    CleaningPlan,
    default_plan_from_summary,
)
from data_cleaning_agent.pipeline_steps import PIPELINE_STEP_ORDER
from data_cleaning_agent.utils import (
    DataFrameSummary,
    format_dataframe_summary,
    get_dataframe_summary,
)

logger = logging.getLogger(__name__)

_PLAN_PROMPT_PATH = Path(__file__).parent / "prompts" / "data_cleaning_plan.md"
PLAN_PROMPT_TEMPLATE = _PLAN_PROMPT_PATH.read_text(encoding="utf-8")

_FIX_PLAN_PROMPT_PATH = Path(__file__).parent / "prompts" / "data_cleaning_plan_fix.md"
FIX_PLAN_PROMPT_TEMPLATE = _FIX_PLAN_PROMPT_PATH.read_text(encoding="utf-8")

_PIPELINE_STEP_IDS_TEXT = ", ".join(step.value for step in PIPELINE_STEP_ORDER)
_STR_OUTPUT_PARSER = StrOutputParser()
_JSON_OUTPUT_PARSER = JsonOutputParser()


def _coerce_column_names(plan: CleaningPlan, field: str) -> set[str]:
    value = getattr(plan, field)
    return set(value) if value else set()


def validate_cleaning_plan(
    plan: CleaningPlan,
    summary: DataFrameSummary,
    *,
    row_id_col: str = DEFAULT_ROW_ID_COL,
) -> None:
    """Check a parsed plan; raise on hard failures, log warnings for likely gaps."""
    if row_id_col not in plan.protected_columns:
        msg = f"protected_columns must include row id column {row_id_col!r}"
        raise ValueError(msg)

    date_like = {name for name, col in summary.columns.items() if col.looks_date_like}
    missing_dt = sorted(
        date_like - _coerce_column_names(plan, "coerce_datetime_columns")
    )
    if missing_dt:
        logger.warning(
            "plan omits date_like columns from coerce_datetime_columns: %s",
            missing_dt,
        )

    numeric_like = {
        name for name, col in summary.columns.items() if col.looks_numeric_string_like
    }
    missing_num = sorted(
        numeric_like - _coerce_column_names(plan, "coerce_numeric_columns")
    )
    if missing_num:
        logger.warning(
            "plan omits numeric_string_like columns from coerce_numeric_columns: %s",
            missing_num,
        )

    bool_like = {
        name for name, col in summary.columns.items() if col.looks_boolean_like
    }
    missing_bool = sorted(bool_like - _coerce_column_names(plan, "coerce_bool_columns"))
    if missing_bool:
        logger.warning(
            "plan omits boolean_like columns from coerce_bool_columns: %s",
            missing_bool,
        )


def parse_cleaning_plan_json(raw: str) -> CleaningPlan:
    """Parse LLM output (optionally fenced in ```json) into a :class:`CleaningPlan`."""
    data = _JSON_OUTPUT_PARSER.parse(raw.strip())
    if not isinstance(data, dict):
        msg = "cleaning plan JSON must be an object"
        raise ValueError(msg)
    return CleaningPlan(**data)


def render_plan_prompt(
    *,
    user_instructions: str,
    dataset_summary: str,
    example_plan: CleaningPlan,
    row_id_col: str = DEFAULT_ROW_ID_COL,
) -> str:
    """Render the plan-generation prompt with runtime values."""
    example_plan_json = json.dumps(asdict(example_plan), indent=2)
    return f"""{
        PLAN_PROMPT_TEMPLATE.format(
            user_instructions=user_instructions,
            all_datasets_summary=dataset_summary,
            pipeline_step_ids=_PIPELINE_STEP_IDS_TEXT,
            example_plan_json=example_plan_json,
            row_id_col=row_id_col,
        )
    }"""


def generate_cleaning_plan(
    model: Any,
    source_df: pd.DataFrame,
    user_instructions: str | None = None,
    *,
    row_id_col: str = DEFAULT_ROW_ID_COL,
) -> CleaningPlan:
    """Call the LLM to produce a validated :class:`CleaningPlan` from JSON output."""
    summary = get_dataframe_summary(source_df)
    example = default_plan_from_summary(summary, row_id_col=row_id_col)
    ui = user_instructions or "Follow the basic cleaning steps."
    dataset_summary = format_dataframe_summary(summary)
    prompt = render_plan_prompt(
        user_instructions=ui,
        dataset_summary=dataset_summary,
        example_plan=example,
        row_id_col=row_id_col,
    )
    raw: str = (model | _STR_OUTPUT_PARSER).invoke(prompt)
    plan = parse_cleaning_plan_json(raw)
    validate_cleaning_plan(plan, summary, row_id_col=row_id_col)
    return plan


def render_fix_plan_prompt(
    *,
    user_instructions: str,
    dataset_summary: str,
    plan_snippet: str,
    error: str,
    row_id_col: str = DEFAULT_ROW_ID_COL,
) -> str:
    """Render the plan-fix prompt with runtime values."""
    return FIX_PLAN_PROMPT_TEMPLATE.format(
        user_instructions=user_instructions,
        all_datasets_summary=dataset_summary,
        pipeline_step_ids=_PIPELINE_STEP_IDS_TEXT,
        plan_snippet=plan_snippet,
        error=error,
        row_id_col=row_id_col,
    )


def fix_cleaning_plan(
    model: Any,
    source_df: pd.DataFrame,
    *,
    broken_plan: dict | None,
    error: str,
    user_instructions: str | None = None,
    row_id_col: str = DEFAULT_ROW_ID_COL,
) -> CleaningPlan:
    """Ask the LLM to correct a plan that failed validation or pipeline execution."""
    summary = get_dataframe_summary(source_df)
    ui = user_instructions or "Follow the basic cleaning steps."
    dataset_summary = format_dataframe_summary(summary)
    plan_snippet = json.dumps(broken_plan or {}, indent=2)
    prompt = render_fix_plan_prompt(
        user_instructions=ui,
        dataset_summary=dataset_summary,
        plan_snippet=plan_snippet,
        error=error,
        row_id_col=row_id_col,
    )
    raw: str = (model | _STR_OUTPUT_PARSER).invoke(prompt)
    plan = parse_cleaning_plan_json(raw)
    validate_cleaning_plan(plan, summary, row_id_col=row_id_col)
    return plan
