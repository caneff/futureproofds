"""Cleaning plan types and summary-derived example plans for LLM prompts."""

from __future__ import annotations

import dataclasses
import json

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


def _user_protected_columns(
    plan: CleaningPlan,
    *,
    row_id_col: str = DEFAULT_ROW_ID_COL,
) -> list[str]:
    return [col for col in plan.protected_columns if col != row_id_col]


def _format_column_list(columns: tuple[str, ...]) -> str:
    if not columns:
        return "—"
    return ", ".join(f"`{name}`" for name in columns)


def format_plan_summary_markdown(
    plan: CleaningPlan,
    *,
    row_id_col: str = DEFAULT_ROW_ID_COL,
) -> str:
    """Human-readable plan summary for Streamlit (omits row id from protected list)."""
    skips = set(plan.skip_steps)
    step_lines: list[str] = []
    for step in pipeline_steps.PIPELINE_STEP_ORDER:
        step_id = step.value
        if step_id in skips:
            step_lines.append(f"- ~~{step_id}~~ — skipped")
        else:
            step_lines.append(f"- {step_id}")

    protected = _user_protected_columns(plan, row_id_col=row_id_col)

    lines: list[str] = [
        "#### Pipeline steps",
        *step_lines,
    ]
    if protected:
        lines.extend([
            "",
            "#### Protected columns",
            ", ".join(f"`{name}`" for name in protected),
            "",
            "_The app-injected synthetic row id is always protected when you apply._",
        ])
    lines.extend([
        "",
        "#### High-missing drop threshold",
        f"Drop columns with **≥ {plan.drop_high_missing_threshold:.0%}** missing "
        "(except protected columns).",
        "",
        "#### Coerce dtypes",
        f"- **Dates:** {_format_column_list(plan.coerce_datetime_columns)}",
        f"- **Numeric strings:** {_format_column_list(plan.coerce_numeric_columns)}",
        f"- **Booleans:** {_format_column_list(plan.coerce_bool_columns)}",
        "",
        "#### Impute",
        f"- **Numeric:** {_format_column_list(plan.impute_numeric_columns)}",
        f"- **Categorical:** {_format_column_list(plan.impute_categorical_columns)}",
    ])
    return "\n".join(lines)


def plan_display_json(
    plan: CleaningPlan,
    *,
    row_id_col: str = DEFAULT_ROW_ID_COL,
) -> str:
    """JSON snapshot for tests or debugging; omits row id from ``protected_columns``."""
    payload = dataclasses.asdict(plan)
    payload["protected_columns"] = _user_protected_columns(plan, row_id_col=row_id_col)
    return json.dumps(payload, indent=2)


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
