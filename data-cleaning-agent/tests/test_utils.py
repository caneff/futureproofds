import math

from data_cleaning_agent.utils import (
    format_dataframe_summary,
    get_dataframe_summary,
)


def test_summary_reports_correct_row_and_column_counts(mixed_df):
    summary = get_dataframe_summary(mixed_df)
    assert summary.n_rows == 5
    assert summary.n_cols == 6


def test_summary_flags_user_id_as_id_like(mixed_df):
    summary = get_dataframe_summary(mixed_df)
    assert summary.columns["user_id"].id_like is True


def test_summary_does_not_flag_country_as_id_like(mixed_df):
    summary = get_dataframe_summary(mixed_df)
    assert summary.columns["country"].id_like is False


def test_summary_numeric_stats_skew_is_finite_for_age(mixed_df):
    stats = get_dataframe_summary(mixed_df).columns["age"].numeric_stats
    assert stats is not None
    assert math.isfinite(stats.skew)


def test_summary_detects_date_like_strings(mixed_df):
    summary = get_dataframe_summary(mixed_df)
    assert summary.columns["signup_date"].looks_date_like is True


def test_summary_detects_currency_strings_as_numeric(mixed_df):
    summary = get_dataframe_summary(mixed_df)
    assert summary.columns["income_str"].looks_numeric_string_like is True


def test_summary_detects_yes_no_as_boolean_like(mixed_df):
    summary = get_dataframe_summary(mixed_df)
    assert summary.columns["is_active"].looks_boolean_like is True


def test_summary_top_categories_present_for_low_cardinality_column(mixed_df):
    cats = get_dataframe_summary(mixed_df).columns["country"].top_categories
    assert cats is not None
    assert {c["value"] for c in cats} == {"US", "UK", "FR"}


def test_format_renders_rows_and_columns_header(mixed_df):
    text = format_dataframe_summary(get_dataframe_summary(mixed_df))
    assert "Rows: 5" in text
    assert "Columns: 6" in text


def test_format_renders_id_like_flag_for_user_id(mixed_df):
    text = format_dataframe_summary(get_dataframe_summary(mixed_df))
    assert "id_like: True" in text
