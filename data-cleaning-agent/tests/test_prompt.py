import pytest
from pathlib import Path

from data_cleaning_agent.data_cleaning_agent import (
    _CODE_ONLY_PROMPT_TEMPLATE,
    _FIX_DATA_CLEANER_PROMPT_TEMPLATE,
    _PLAN_FROM_CODE_PROMPT_TEMPLATE,
)
from langchain_core.prompts import PromptTemplate

_DATA_CLEANING_INDEX = (
    Path(__file__).resolve().parent.parent
    / "data_cleaning_agent"
    / "prompts"
    / "data_cleaning.md"
)


@pytest.mark.unit
def test_data_cleaning_md_is_index_linking_runtime_prompts():
    """data_cleaning.md must stay a non-template index (single source of truth story)."""
    text = _DATA_CLEANING_INDEX.read_text(encoding="utf-8")
    assert "./data_cleaning_code_only.md" in text
    assert "./data_cleaning_plan_from_code.md" in text
    assert "./data_cleaning_fix.md" in text
    assert "{user_instructions}" not in text


@pytest.mark.unit
def test_code_only_prompt_renders_with_only_expected_variables():
    """Catch unescaped {braces} in the code-only prompt without an LLM round-trip."""
    prompt = PromptTemplate(
        template=_CODE_ONLY_PROMPT_TEMPLATE,
        input_variables=[
            "user_instructions",
            "all_datasets_summary",
            "function_name",
        ],
    )
    rendered = prompt.format(
        user_instructions="<u>",
        all_datasets_summary="<s>",
        function_name="data_cleaner",
    )
    assert "data_cleaner(source_df)" in rendered
    assert "Pipeline (in order)" in rendered
    assert "missing share **> 0.4**" in rendered
    assert "Forbidden: do not loop over all object" in rendered
    assert "employee_id" in rendered
    assert "<u>" in rendered
    assert "__agent_row_id__" in rendered
    assert "<s>" in rendered
    assert "Step 9 cheat sheet" in rendered
    assert "A structured cleaning-plan JSON is produced" in rendered


@pytest.mark.unit
def test_plan_from_code_prompt_renders_with_only_expected_variables():
    """Plan-from-code template must format with the same three variables as code-only."""
    prompt = PromptTemplate(
        template=_PLAN_FROM_CODE_PROMPT_TEMPLATE,
        input_variables=[
            "user_instructions",
            "all_datasets_summary",
            "function_name",
        ],
    )
    rendered = prompt.format(
        user_instructions="<u>",
        all_datasets_summary="<s>",
        function_name="data_cleaner",
    )
    assert "```json" in rendered
    assert "Plan JSON rules" in rendered
    assert "<u>" in rendered
    assert "<s>" in rendered
    assert "data_cleaner" in rendered


@pytest.mark.unit
def test_render_check_rejects_unescaped_braces():
    """Sanity-check the guardrail: an unescaped {col} placeholder must raise."""
    bad_template = _CODE_ONLY_PROMPT_TEMPLATE + "\nExample: df.fillna({col: value})"
    prompt = PromptTemplate(
        template=bad_template,
        input_variables=[
            "user_instructions",
            "all_datasets_summary",
            "function_name",
        ],
    )
    with pytest.raises(KeyError, match="col"):
        prompt.format(
            user_instructions="<u>",
            all_datasets_summary="<s>",
            function_name="data_cleaner",
        )


@pytest.mark.unit
def test_fix_prompt_formats_with_expected_placeholders():
    """Fix prompt is loaded from markdown; ``fix_agent_code`` uses ``str.format``."""
    rendered = _FIX_DATA_CLEANER_PROMPT_TEMPLATE.format(
        function_name="data_cleaner",
        code_snippet="def data_cleaner(df):\n    return df",
        error="TypeError: ...",
    )
    assert "data_cleaner" in rendered
    assert "```python" in rendered
    assert "TypeError" in rendered
    assert "def data_cleaner(df):" in rendered
    assert "make the code honor retain" in rendered
