import pytest
from data_cleaning_agent.data_cleaning_agent import _DATA_CLEANING_PROMPT_TEMPLATE
from langchain_core.prompts import PromptTemplate


@pytest.mark.unit
def test_prompt_renders_with_only_expected_variables():
    """Catch unescaped {braces} in the prompt without an LLM round-trip."""
    prompt = PromptTemplate(
        template=_DATA_CLEANING_PROMPT_TEMPLATE,
        input_variables=["user_instructions", "all_datasets_summary", "function_name"],
    )
    rendered = prompt.format(
        user_instructions="<u>",
        all_datasets_summary="<s>",
        function_name="data_cleaner",
    )
    assert "data_cleaner(data_raw)" in rendered
    assert "<u>" in rendered
    assert "<s>" in rendered


@pytest.mark.unit
def test_render_check_rejects_unescaped_braces():
    """Sanity-check the guardrail: an unescaped {col} placeholder must raise."""
    bad_template = _DATA_CLEANING_PROMPT_TEMPLATE + "\nExample: df.fillna({col: value})"
    prompt = PromptTemplate(
        template=bad_template,
        input_variables=["user_instructions", "all_datasets_summary", "function_name"],
    )
    with pytest.raises(KeyError, match="col"):
        prompt.format(
            user_instructions="<u>",
            all_datasets_summary="<s>",
            function_name="data_cleaner",
        )
