import pytest
from data_cleaning_agent.data_cleaning_agent import (
    _DATA_CLEANING_PROMPT_TEMPLATE,
    _FIX_DATA_CLEANER_PROMPT_TEMPLATE,
)
from langchain_core.prompts import PromptTemplate


@pytest.mark.unit
def test_prompt_renders_with_only_expected_variables():
    """Catch unescaped {braces} in the prompt without an LLM round-trip."""
    prompt = PromptTemplate(
        template=_DATA_CLEANING_PROMPT_TEMPLATE,
        input_variables=[
            "user_instructions",
            "supplemental_instructions",
            "all_datasets_summary",
            "function_name",
        ],
    )
    rendered = prompt.format(
        user_instructions="<u>",
        supplemental_instructions="<sup>",
        all_datasets_summary="<s>",
        function_name="data_cleaner",
    )
    assert "data_cleaner(source_df)" in rendered
    assert "```json" in rendered
    assert "row_ops" in rendered
    assert "drop column (>40% missing)" in rendered
    assert "rows removed" in rendered
    assert "impute missing values (mode)" in rendered
    assert "only numeric" in rendered
    assert "employee_id" in rendered
    assert "<u>" in rendered
    assert "<sup>" in rendered
    assert "<s>" in rendered


@pytest.mark.unit
def test_render_check_rejects_unescaped_braces():
    """Sanity-check the guardrail: an unescaped {col} placeholder must raise."""
    bad_template = _DATA_CLEANING_PROMPT_TEMPLATE + "\nExample: df.fillna({col: value})"
    prompt = PromptTemplate(
        template=bad_template,
        input_variables=[
            "user_instructions",
            "supplemental_instructions",
            "all_datasets_summary",
            "function_name",
        ],
    )
    with pytest.raises(KeyError, match="col"):
        prompt.format(
            user_instructions="<u>",
            supplemental_instructions="<sup>",
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
