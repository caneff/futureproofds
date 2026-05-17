from pathlib import Path

import pytest

from data_cleaning_agent.plan_generation import (
    FIX_PLAN_PROMPT_TEMPLATE,
    PLAN_PROMPT_TEMPLATE,
)

_PLAN_MD = (
    Path(__file__).resolve().parent.parent
    / "data_cleaning_agent"
    / "prompts"
    / "data_cleaning_plan.md"
)
_FIX_PLAN_MD = (
    Path(__file__).resolve().parent.parent
    / "data_cleaning_agent"
    / "prompts"
    / "data_cleaning_plan_fix.md"
)


@pytest.mark.unit
def test_plan_prompt_md_is_langchain_template_with_expected_variables() -> None:
    """Plan prompt file must interpolate runtime variables."""
    text = _PLAN_MD.read_text(encoding="utf-8")
    assert "{user_instructions}" in text
    assert "{all_datasets_summary}" in text
    assert "{pipeline_step_ids}" in text
    assert "{example_plan_json}" in text
    assert "{row_id_col}" in text
    assert "Do **not** write Python code" in text


@pytest.mark.unit
def test_plan_prompt_renders_with_only_expected_placeholders() -> None:
    """Catch unescaped {braces} in the plan prompt without an LLM round-trip."""
    rendered = PLAN_PROMPT_TEMPLATE.format(
        user_instructions="<u>",
        all_datasets_summary="<s>",
        pipeline_step_ids="copy, normalize_names",
        example_plan_json='{"skip_steps": []}',
        row_id_col="__agent_row_id__",
    )
    assert "CleaningPlan" in rendered
    assert "```json" in rendered
    assert "<u>" in rendered
    assert "<s>" in rendered
    assert "__agent_row_id__" in rendered
    assert "copy, normalize_names" in rendered


@pytest.mark.unit
def test_fix_plan_prompt_formats_with_expected_placeholders() -> None:
    """Fix prompt is loaded from markdown; uses ``str.format``."""
    rendered = FIX_PLAN_PROMPT_TEMPLATE.format(
        user_instructions="protect country",
        all_datasets_summary="Rows: 5",
        pipeline_step_ids="normalize_names, impute",
        plan_snippet='{"skip_steps": []}',
        error="ValueError: unknown skip_steps",
        row_id_col="__agent_row_id__",
    )
    assert "CleaningPlan" in rendered
    assert "```json" in rendered
    assert "unknown skip_steps" in rendered
    assert "__agent_row_id__" in rendered
