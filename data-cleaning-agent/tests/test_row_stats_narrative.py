"""Unit tests for plan row-stats UI helpers (no Streamlit)."""

import pytest
from row_stats_narrative import (
    glossary_bullets,
    verified_row_stats_strip_items,
)


@pytest.mark.unit
def test_strip_returns_none_when_stats_not_dict():
    assert verified_row_stats_strip_items(None) is None
    assert verified_row_stats_strip_items("bad") is None


@pytest.mark.unit
def test_strip_returns_none_on_error_or_missing_keys():
    assert verified_row_stats_strip_items({"error": "boom"}) is None
    assert verified_row_stats_strip_items({"n_in": 1}) is None
    assert verified_row_stats_strip_items({"n_out": 1}) is None


@pytest.mark.unit
def test_strip_three_metrics_when_all_null_stat_unavailable():
    stats = {
        "n_in": 100,
        "n_out": 95,
        "removed_total": 5,
        "removed_all_null_input_user_cols": None,
    }
    items = verified_row_stats_strip_items(stats)
    assert items == [
        ("Rows in", "100"),
        ("Rows out", "95"),
        ("Removed", "5"),
    ]


@pytest.mark.unit
def test_strip_four_metrics_when_all_null_present():
    stats = {
        "n_in": 4,
        "n_out": 2,
        "removed_total": 2,
        "removed_all_null_input_user_cols": 2,
    }
    items = verified_row_stats_strip_items(stats)
    assert len(items) == 4
    assert items[3] == ("All-null (removed)", "2")


@pytest.mark.unit
def test_strip_shows_zeros_when_no_rows_removed():
    """Spec: show strip with zeros for consistency when measurement succeeded."""
    stats = {
        "n_in": 10,
        "n_out": 10,
        "removed_total": 0,
        "removed_all_null_input_user_cols": 0,
    }
    items = verified_row_stats_strip_items(stats)
    assert items[2] == ("Removed", "0")


@pytest.mark.unit
def test_glossary_returns_non_empty_list():
    bullets = glossary_bullets()
    assert isinstance(bullets, list)
    assert len(bullets) >= 3
    assert all(isinstance(b, str) and b.strip() for b in bullets)
