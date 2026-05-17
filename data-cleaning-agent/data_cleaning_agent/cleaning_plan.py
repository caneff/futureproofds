"""Cleaning plan types and summary-derived example plans for LLM prompts."""

from __future__ import annotations

import dataclasses

import data_cleaning_agent.pipeline_steps as pipeline_steps
import data_cleaning_agent.utils as utils

DEFAULT_ROW_ID_COL = "__agent_row_id__"
DEFAULT_DROP_HIGH_MISSING_THRESHOLD = 0.4


@dataclasses.dataclass
class CleaningPlan:
    """Validated cleaning plan for the hybrid pipeline.

    ``protected_columns`` is the single keep-list for destructive steps (drops,
    strip, impute skips). Step-specific fields only cover parameters that vary
    per run (threshold, dtype coercion targets, impute column lists).
    """

    skip_steps: list[str] = dataclasses.field(default_factory=list)
    protected_columns: list[str] = dataclasses.field(default_factory=list)
    drop_high_missing_threshold: float = DEFAULT_DROP_HIGH_MISSING_THRESHOLD
    coerce_datetime_columns: tuple[str, ...] = ()
    coerce_numeric_columns: tuple[str, ...] = ()
    coerce_bool_columns: tuple[str, ...] = ()
    impute_numeric_columns: tuple[str, ...] = ()
    impute_categorical_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = set(self.skip_steps) - pipeline_steps.ALL_STEP_IDS
        if unknown:
            msg = f"unknown skip_steps: {sorted(unknown)}"
            raise ValueError(msg)


def default_plan_from_summary(
    summary: utils.DataFrameSummary,
    *,
    row_id_col: str = DEFAULT_ROW_ID_COL,
) -> CleaningPlan:
    """Build an example plan from summary flags for the plan-generation prompt.

    Not applied at pipeline runtime; the LLM must return a complete plan.
    """
    dt_cols = [n for n, c in summary.columns.items() if c.looks_date_like]
    num_cols = [n for n, c in summary.columns.items() if c.looks_numeric_string_like]
    bool_cols = [n for n, c in summary.columns.items() if c.looks_boolean_like]

    return CleaningPlan(
        protected_columns=[row_id_col],
        coerce_datetime_columns=tuple(dt_cols),
        coerce_numeric_columns=tuple(num_cols),
        coerce_bool_columns=tuple(bool_cols),
    )
