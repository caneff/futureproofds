import math
from unittest.mock import MagicMock

import pandas as pd
import pytest
from data_cleaning_agent.utils import (
    APP_SYNTHETIC_ALIGN_ROW_ID_COLUMN,
    PythonOutputParser,
    execute_agent_code,
    fix_agent_code,
    format_dataframe_summary,
    get_dataframe_summary,
    plan_step9_policy_host_supplement,
)


@pytest.mark.unit
class TestGetDataFrameSummary:
    """Unit tests for ``get_dataframe_summary`` and its detection helpers."""

    def test_reports_correct_row_and_column_counts(self, summary):
        assert (summary.n_rows, summary.n_cols) == (5, 6)

    def test_numeric_stats_skew_is_finite_for_age(self, summary):
        stats = summary.columns["age"].numeric_stats
        assert stats is not None
        assert math.isfinite(stats.skew)

    def test_top_categories_present_for_low_cardinality_column(self, summary):
        cats = summary.columns["country"].top_categories
        assert cats is not None
        assert {c["value"] for c in cats} == {"US", "UK", "FR"}

    def test_missing_pct_for_age_is_twenty_percent(self, summary):
        assert summary.columns["age"].missing_pct == 20.0

    def test_top_categories_percentages_sum_to_one_hundred(self, summary):
        cats = summary.columns["country"].top_categories
        assert cats is not None
        total_pct = sum(c["pct"] for c in cats)
        assert total_pct == pytest.approx(100.0, abs=0.01)

    @pytest.mark.parametrize(
        "col,flag,expected",
        [
            pytest.param("user_id", "id_like", True, id="id_like-true-user_id"),
            pytest.param("country", "id_like", False, id="id_like-false-country"),
            pytest.param("signup_date", "looks_date_like", True, id="date_like-true"),
            pytest.param("country", "looks_date_like", False, id="date_like-false"),
            pytest.param(
                "income_str", "looks_numeric_string_like", True, id="numstr-true"
            ),
            pytest.param(
                "country", "looks_numeric_string_like", False, id="numstr-false"
            ),
            pytest.param("is_active", "looks_boolean_like", True, id="bool-true"),
            pytest.param(
                "country", "looks_boolean_like", False, id="bool-false-3values"
            ),
        ],
    )
    def test_detection_flag(self, summary, col, flag, expected):
        assert getattr(summary.columns[col], flag) is expected

    def test_skew_coerced_to_zero_for_single_value(self, small_numeric_df):
        stats = get_dataframe_summary(small_numeric_df).columns["x"].numeric_stats
        assert stats is not None
        assert stats.skew == 0.0

    def test_std_coerced_to_zero_for_single_value(self, small_numeric_df):
        stats = get_dataframe_summary(small_numeric_df).columns["x"].numeric_stats
        assert stats is not None
        assert stats.std == 0.0

    def test_detects_id_like_via_monotonic_int(self, monotonic_int_df):
        summary = get_dataframe_summary(monotonic_int_df)
        assert summary.columns["counter"].id_like is True

    def test_handles_empty_dataframe(self, empty_df):
        summary = get_dataframe_summary(empty_df)
        assert summary.n_rows == 0
        assert summary.columns["a"].missing_pct == 0.0
        assert summary.columns["a"].id_like is False

    def test_boolean_like_false_when_cardinality_above_two(self, summary):
        assert summary.columns["country"].looks_boolean_like is False


@pytest.mark.unit
class TestFormatDataFrameSummary:
    """Unit tests for the deterministic text rendering of ``DataFrameSummary``."""

    def test_renders_rows_and_columns_header(self, summary):
        text = format_dataframe_summary(summary)
        assert "Rows: 5" in text
        assert "Columns: 6" in text

    def test_renders_id_like_flag_for_user_id(self, summary):
        text = format_dataframe_summary(summary)
        assert "id_like: True" in text

    def test_renders_numeric_stats_for_age(self, summary):
        text = format_dataframe_summary(summary)
        assert "numeric stats: min=" in text
        assert "skew=" in text

    def test_renders_top_categories_line_for_country(self, summary):
        text = format_dataframe_summary(summary)
        assert "top categories:" in text
        assert "US (" in text

    def test_renders_detection_line_for_signup_date(self, summary):
        text = format_dataframe_summary(summary)
        assert "detection: date_like=True" in text

    def test_omits_id_like_line_for_non_id_column(self, summary):
        text = format_dataframe_summary(summary)
        country_block = text.split("- country")[1].split("- signup_date")[0]
        assert "id_like" not in country_block


@pytest.mark.unit
class TestPythonOutputParser:
    """Unit tests for the regex-based Python-code extraction parser."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            pytest.param("```python\nprint('x')\n```", "print('x')", id="fenced"),
            pytest.param(
                "Sure, here:\n```python\nx = 1\n```\nDone.",
                "x = 1",
                id="fenced-with-prose",
            ),
            pytest.param("no fences here", "no fences here", id="passthrough"),
        ],
    )
    def test_extracts_or_passes_through(self, text, expected):
        assert PythonOutputParser().parse(text) == expected


@pytest.mark.unit
class TestExecuteAgentCode:
    """Unit tests for ``execute_agent_code``: success, runtime failure, missing fn."""

    def test_success_adds_column_and_returns_no_error(self):
        state = {
            "data": {"a": [1, 2, 3]},
            "code": "def clean(df):\n    df['b'] = df['a'] * 2\n    return df\n",
        }

        out = execute_agent_code(state, "data", "code", "result", "error", "clean")

        assert out["error"] is None
        assert out["result"] == {"a": {0: 1, 1: 2, 2: 3}, "b": {0: 2, 1: 4, 2: 6}}

    def test_passes_through_columns_when_cleaner_returns_copy(self):
        state = {
            "data": {
                "employee_id": ["", "", "", "", "", "", "", "", "", "E1"],
                "x": list(range(10)),
            },
            "code": "def clean(df):\n    return df.copy()\n",
        }

        out = execute_agent_code(state, "data", "code", "result", "error", "clean")

        assert out["error"] is None
        res = out["result"]
        assert res is not None
        assert "employee_id" in res
        assert "x" in res

    def test_runtime_failure_captures_error_and_returns_none_result(self):
        state = {
            "data": {"a": [1]},
            "code": "def clean(df):\n    raise ValueError('boom')\n",
        }

        out = execute_agent_code(state, "data", "code", "result", "error", "clean")

        assert out["result"] is None
        assert out["error"].startswith("An error occurred during data cleaning:")
        assert "boom" in out["error"]

    def test_keyerror_message_hints_dropped_columns(self):
        state = {
            "data": {"a": [1]},
            "code": "def clean(df):\n    return df['missing_col']\n",
        }

        out = execute_agent_code(state, "data", "code", "result", "error", "clean")

        assert out["result"] is None
        assert "missing_col" in (out["error"] or "")
        assert "steps 3 or 7" in (out["error"] or "")

    def test_length_mismatch_error_includes_alignment_hint(self):
        state = {
            "data": {"a": list(range(96))},
            "code": ("def clean(df):\n    df['b'] = [1, 2, 3, 4]\n    return df\n"),
        }

        out = execute_agent_code(state, "data", "code", "result", "error", "clean")

        assert out["result"] is None
        err = out["error"] or ""
        assert "Length of values" in err
        assert "Hint:" in err
        assert "index=df.index" in err

    def test_raises_when_named_function_not_in_generated_code(self):
        state = {
            "data": {"a": [1]},
            "code": "def other(df):\n    return df\n",
        }

        with pytest.raises(
            ValueError, match="Function 'clean' not found in generated code."
        ):
            execute_agent_code(state, "data", "code", "result", "error", "clean")


@pytest.mark.unit
class TestFixAgentCode:
    """Unit tests for ``fix_agent_code``: prompt formatting, retry increment, LLM mock."""

    def test_increments_retry_and_formats_prompt(self):
        fake_chain_output = "def clean(df):\n    return df"
        llm = MagicMock()
        llm.__or__.return_value.invoke.return_value = fake_chain_output

        state = {"code": "broken", "error": "boom", "retry_count": 0}

        out = fix_agent_code(
            state,
            "code",
            "error",
            llm,
            "fix: {code_snippet} err: {error} fn: {function_name}",
            "clean",
        )

        assert out["code"] == fake_chain_output
        assert out["error"] is None
        assert out["retry_count"] == 1
        llm.__or__.return_value.invoke.assert_called_once_with(
            "fix: broken err: boom fn: clean"
        )

    def test_uses_custom_retry_count_key(self):
        llm = MagicMock()
        llm.__or__.return_value.invoke.return_value = "fixed code"
        state = {"code": "broken", "error": "boom", "attempts": 2}

        out = fix_agent_code(
            state,
            "code",
            "error",
            llm,
            "{code_snippet}|{error}|{function_name}",
            "clean",
            retry_count_key="attempts",
        )

        assert out["attempts"] == 3


def test_plan_step9_policy_host_supplement_lists_columns_with_missingness() -> None:
    rid = APP_SYNTHETIC_ALIGN_ROW_ID_COLUMN
    df = pd.DataFrame({rid: [0, 1], "city": ["London", None], "age": [1.0, 2.0]})
    out = plan_step9_policy_host_supplement(df, row_id_col=rid)
    assert "`city`" in out
    assert "age" not in out
    assert "impute missing values" in out or "retain missing values" in out


def test_plan_step9_policy_host_supplement_empty_when_no_missing() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert plan_step9_policy_host_supplement(df, row_id_col="__unused__") == ""
