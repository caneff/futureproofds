from pathlib import Path

import pytest
from data_cleaning_agent.data_cleaning_agent import (
    _FIX_DATA_CLEANER_PROMPT_TEMPLATE,
    _PIPELINE_PROMPT_TEMPLATE,
)
from langchain_core.prompts import PromptTemplate

_DATA_CLEANING_MD = (
    Path(__file__).resolve().parent.parent
    / "data_cleaning_agent"
    / "prompts"
    / "data_cleaning.md"
)


@pytest.mark.unit
def test_data_cleaning_md_is_langchain_template_with_expected_variables():
    """Main pipeline prompt file must interpolate the three runtime variables."""
    text = _DATA_CLEANING_MD.read_text(encoding="utf-8")
    assert "{user_instructions}" in text
    assert "{all_datasets_summary}" in text
    assert "{function_name}" in text
    assert "./data_cleaning_code_only.md" not in text
    assert "./data_cleaning_plan_from_code.md" not in text


@pytest.mark.unit
def test_pipeline_prompt_renders_with_only_expected_variables():
    """Catch unescaped {braces} in the pipeline prompt without an LLM round-trip."""
    prompt = PromptTemplate(
        template=_PIPELINE_PROMPT_TEMPLATE,
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
    assert "Never treat Dataset Summary as User Instructions" in rendered
    assert "if ... not in [...]" in rendered
    assert "Step-3 exemptions are **only**" in rendered
    assert "**never** high-missing" in rendered
    assert "<u>" in rendered
    assert "__agent_row_id__" in rendered
    assert "<s>" in rendered
    assert "**no** built-in list of" in rendered
    assert "This prompt emits **Python only**" in rendered
    assert "is_object_dtype" in rendered


@pytest.mark.unit
def test_render_check_rejects_unescaped_braces():
    """Sanity-check the guardrail: an unescaped {col} placeholder must raise."""
    bad_template = _PIPELINE_PROMPT_TEMPLATE + "\nExample: df.fillna({col: value})"
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
    assert "no json" in rendered.lower()
    assert "Can only use .str accessor" in rendered
