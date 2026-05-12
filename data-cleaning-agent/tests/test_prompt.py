import pytest
from data_cleaning_agent.data_cleaning_agent import (
    _DATA_CLEANING_PROMPT_TEMPLATE,
    _PROMPT_PATH,
)


@pytest.mark.unit
class TestCleaningPromptGuardrails:
    """Regression locks for cleaning-prompt guidance that prevents known LLM failure modes."""

    def test_prompt_file_exists_and_is_non_empty(self):
        assert _PROMPT_PATH.is_file()
        assert _DATA_CLEANING_PROMPT_TEMPLATE.strip(), "prompt file loaded empty"

    def test_currency_strip_uses_raw_string_regex(self):
        assert 'r"[$,%]"' in _DATA_CLEANING_PROMPT_TEMPLATE
        assert "regex=True" in _DATA_CLEANING_PROMPT_TEMPLATE

    def test_currency_strip_forbids_bad_escape(self):
        assert (
            'Do NOT use plain-string escapes like "\\$"'
            in _DATA_CLEANING_PROMPT_TEMPLATE
        )

    def test_skew_uses_builtin_abs_on_scalar(self):
        assert "abs(skew_val)" in _DATA_CLEANING_PROMPT_TEMPLATE
        assert "SCALAR" in _DATA_CLEANING_PROMPT_TEMPLATE
        assert "skew().abs()" not in _DATA_CLEANING_PROMPT_TEMPLATE

    def test_forbids_chained_inplace_assignment(self):
        assert "Never use chained-assignment with inplace=True" in _DATA_CLEANING_PROMPT_TEMPLATE
        assert "df[col] = df[col].fillna(value)" in _DATA_CLEANING_PROMPT_TEMPLATE
        assert "ChainedAssignmentError" in _DATA_CLEANING_PROMPT_TEMPLATE
