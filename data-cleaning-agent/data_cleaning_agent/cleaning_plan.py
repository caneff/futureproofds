"""Cleaning plan types and summary-based defaults for the hybrid pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from data_cleaning_agent.pipeline_steps import ALL_STEP_IDS
from data_cleaning_agent.utils import DataFrameSummary

DEFAULT_ROW_ID_COL = "__agent_row_id__"
DEFAULT_DROP_HIGH_MISSING_THRESHOLD = 0.4


@dataclass
class CleaningPlan:
    """Validated cleaning plan for the hybrid pipeline.

    ``protected_columns`` is the single keep-list for destructive steps (drops,
    strip, impute skips). Step-specific fields only cover parameters that vary
    per run (threshold, dtype coercion targets, impute column lists).
    """

    skip_steps: list[str] = field(default_factory=list)
    protected_columns: list[str] = field(default_factory=list)
    drop_high_missing_threshold: float = DEFAULT_DROP_HIGH_MISSING_THRESHOLD
    coerce_datetime_columns: tuple[str, ...] = ()
    coerce_numeric_columns: tuple[str, ...] = ()
    coerce_bool_columns: tuple[str, ...] = ()
    impute_numeric_columns: tuple[str, ...] = ()
    impute_categorical_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = set(self.skip_steps) - ALL_STEP_IDS
        if unknown:
            msg = f"unknown skip_steps: {sorted(unknown)}"
            raise ValueError(msg)


def default_plan_from_summary(
    summary: DataFrameSummary,
    *,
    row_id_col: str = DEFAULT_ROW_ID_COL,
) -> CleaningPlan:
    """Build a summary-driven baseline plan (coerce targets + row id protection).

    User instructions are not interpreted here; they are passed to the LLM when
    generating or revising a plan.
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
