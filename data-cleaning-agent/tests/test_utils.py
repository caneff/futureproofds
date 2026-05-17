import math

import pytest

import data_cleaning_agent.utils as utils


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
        stats = utils.get_dataframe_summary(small_numeric_df).columns["x"].numeric_stats
        assert stats is not None
        assert stats.skew == 0.0

    def test_std_coerced_to_zero_for_single_value(self, small_numeric_df):
        stats = utils.get_dataframe_summary(small_numeric_df).columns["x"].numeric_stats
        assert stats is not None
        assert stats.std == 0.0

    def test_monotonic_int_column_summarizes(self, monotonic_int_df):
        summary = utils.get_dataframe_summary(monotonic_int_df)
        assert summary.columns["counter"].name == "counter"
        assert summary.n_rows == len(monotonic_int_df)

    def test_handles_empty_dataframe(self, empty_df):
        summary = utils.get_dataframe_summary(empty_df)
        assert summary.n_rows == 0
        assert summary.columns["a"].missing_pct == 0.0

    def test_boolean_like_false_when_cardinality_above_two(self, summary):
        assert summary.columns["country"].looks_boolean_like is False


@pytest.mark.unit
class TestFormatDataFrameSummary:
    """Unit tests for the deterministic text rendering of ``DataFrameSummary``."""

    def test_renders_rows_and_columns_header(self, summary):
        text = utils.format_dataframe_summary(summary)
        assert "Rows: 5" in text
        assert "Columns: 6" in text

    def test_formatted_summary_has_no_id_like_token(self, summary):
        """Regression: summary text must not revive removed id_like metadata."""
        text = utils.format_dataframe_summary(summary)
        assert "id_like" not in text

    def test_renders_numeric_stats_for_age(self, summary):
        text = utils.format_dataframe_summary(summary)
        assert "numeric stats: min=" in text
        assert "skew=" in text

    def test_renders_top_categories_line_for_country(self, summary):
        text = utils.format_dataframe_summary(summary)
        assert "top categories:" in text
        assert "US (" in text

    def test_renders_detection_line_for_signup_date(self, summary):
        text = utils.format_dataframe_summary(summary)
        assert "detection: date_like=True" in text

    def test_omits_id_like_line_for_non_id_column(self, summary):
        text = utils.format_dataframe_summary(summary)
        country_block = text.split("- country")[1].split("- signup_date")[0]
        assert "id_like" not in country_block
