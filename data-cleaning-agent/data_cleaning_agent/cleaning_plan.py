"""Pydantic schema for the hybrid cleaning pipeline plan and summary-based defaults."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from data_cleaning_agent.pipeline_steps import ALL_STEP_IDS
from data_cleaning_agent.utils import DataFrameSummary

DEFAULT_ROW_ID_COL = "__agent_row_id__"
DEFAULT_DROP_HIGH_MISSING_THRESHOLD = 0.4


class DropHighMissingParams(BaseModel):
    """Parameters for dropping columns above a missingness threshold."""

    threshold: float = DEFAULT_DROP_HIGH_MISSING_THRESHOLD
    exclude: tuple[str, ...] = ()


class ImputeParams(BaseModel):
    """Column lists for numeric and categorical imputation."""

    numeric_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    skip_columns: tuple[str, ...] = ()


def _frozenset_from_iterable(value: object) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, frozenset):
        return value
    if isinstance(value, (set, list, tuple)):
        return frozenset(str(item) for item in value)
    return frozenset([str(value)])


class CleaningPlan(BaseModel):
    """Validated cleaning plan: step skips, protected columns, and per-step parameters."""

    skip_steps: frozenset[str] = frozenset()
    protected_columns: frozenset[str] = frozenset()
    drop_high_missing: DropHighMissingParams = Field(
        default_factory=DropHighMissingParams
    )
    strip_exclude: tuple[str, ...] = ()
    coerce_datetime_columns: tuple[str, ...] = ()
    coerce_numeric_columns: tuple[str, ...] = ()
    coerce_bool_columns: tuple[str, ...] = ()
    drop_constant_exclude: tuple[str, ...] = ()
    drop_all_null_exclude: tuple[str, ...] = ()
    impute: ImputeParams = Field(default_factory=ImputeParams)

    @field_validator("skip_steps", "protected_columns", mode="before")
    @classmethod
    def _coerce_frozenset_fields(cls, value: object) -> frozenset[str]:
        return _frozenset_from_iterable(value)

    @field_validator("skip_steps")
    @classmethod
    def _validate_skip_steps(cls, value: frozenset[str]) -> frozenset[str]:
        unknown = value - ALL_STEP_IDS
        if unknown:
            msg = f"unknown skip_steps: {sorted(unknown)}"
            raise ValueError(msg)
        return value


def _columns_mentioned_in_instructions(
    summary: DataFrameSummary,
    user_instructions: str,
) -> set[str]:
    text = user_instructions.lower()
    return {name for name in summary.columns if name.lower() in text}


def default_plan_from_summary(
    summary: DataFrameSummary,
    *,
    user_instructions: str | None,
    row_id_col: str = DEFAULT_ROW_ID_COL,
) -> CleaningPlan:
    """Build a baseline plan from column detection flags and optional user text."""
    dt_cols: list[str] = []
    num_cols: list[str] = []
    bool_cols: list[str] = []
    for name, col in summary.columns.items():
        if col.looks_date_like:
            dt_cols.append(name)
        if col.looks_numeric_string_like:
            num_cols.append(name)
        if col.looks_boolean_like:
            bool_cols.append(name)

    protected: set[str] = {row_id_col}
    if user_instructions:
        protected |= _columns_mentioned_in_instructions(summary, user_instructions)

    return CleaningPlan(
        protected_columns=frozenset(protected),
        coerce_datetime_columns=tuple(dt_cols),
        coerce_numeric_columns=tuple(num_cols),
        coerce_bool_columns=tuple(bool_cols),
    )
