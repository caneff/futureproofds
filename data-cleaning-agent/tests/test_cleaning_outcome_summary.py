"""Unit tests for verified cleaning outcome facts (no Streamlit)."""

import numpy as np
import pandas as pd
import pytest

from data_cleaning_agent.cleaning_outcome_summary import (
    build_cleaning_outcome_facts,
    format_outcome_summary_markdown,
    outcome_facts_show_any_change,
)
from data_cleaning_agent.utils import APP_SYNTHETIC_ALIGN_ROW_ID_COLUMN


@pytest.mark.unit
def test_dropped_column_listed_in_facts():
    df_before = pd.DataFrame({"a": range(10), "drop_me": ["", ""] * 5})
    df_after = df_before.drop(columns=["drop_me"])
    facts = build_cleaning_outcome_facts(df_before, df_after, row_id_col="__missing__")
    assert "drop_me" in facts["columns"]["dropped"]


@pytest.mark.unit
def test_dtype_changed_on_shared_column():
    df_before = pd.DataFrame({"x": ["1", "2", "3"]}, dtype=object)
    df_after = pd.DataFrame({"x": [1, 2, 3]})
    facts = build_cleaning_outcome_facts(df_before, df_after, row_id_col="__missing__")
    assert len(facts["columns"]["dtype_changed"]) >= 1
    assert facts["columns"]["dtype_changed"][0]["name"] == "x"


@pytest.mark.unit
def test_dtype_drift_suppressed_when_values_match():
    """int64 vs int32 (or similar) alone must not count as a cleaning dtype change."""
    df_before = pd.DataFrame({"x": [1, 2, 3]}, dtype="int64")
    df_after = pd.DataFrame({"x": [1, 2, 3]}, dtype="int32")
    facts = build_cleaning_outcome_facts(df_before, df_after, row_id_col="__missing__")
    assert facts["columns"]["dtype_changed"] == []


@pytest.mark.unit
def test_format_omits_empty_dtype_and_null_sections():
    facts = build_cleaning_outcome_facts(
        pd.DataFrame({"x": [1, 2]}),
        pd.DataFrame({"x": [1, 2]}),
        row_id_col="__missing__",
    )
    text = format_outcome_summary_markdown(facts)
    assert "**Dtype Changes**" not in text
    assert "**Missing Value Count Changes" not in text


@pytest.mark.unit
def test_build_facts_survives_duplicate_column_label_on_shared_name():
    df_before = pd.DataFrame(np.ones((2, 2)))
    df_before.columns = ["a", "a"]
    df_after = df_before.copy()
    facts = build_cleaning_outcome_facts(df_before, df_after, row_id_col="__missing__")
    assert facts["rows"]["n_before"] == 2


@pytest.mark.unit
def test_format_redacts_internal_row_id_column_name_in_lists():
    rid = APP_SYNTHETIC_ALIGN_ROW_ID_COLUMN
    facts = {
        "rows": {"n_before": 2, "n_after": 2},
        "columns": {"dropped": [rid, "x"], "added": [], "dtype_changed": []},
        "null_deltas": [
            {"column": rid, "missing_before": 0, "missing_after": 1, "delta": 1}
        ],
    }
    text = format_outcome_summary_markdown(facts)
    assert rid not in text
    assert "synthetic alignment column (app-injected)" in text


@pytest.mark.unit
def test_rows_section_includes_id_counts_when_row_id_present():
    rid = APP_SYNTHETIC_ALIGN_ROW_ID_COLUMN
    df_before = pd.DataFrame({rid: [0, 1, 2], "v": [1, 2, 3]})
    df_after = pd.DataFrame({rid: [0, 1], "v": [1, 2]})
    facts = build_cleaning_outcome_facts(df_before, df_after, row_id_col=rid)
    assert facts["rows"]["rows_removed_by_id"] == 1
    assert facts["rows"]["rows_added_by_id"] == 0
    text = format_outcome_summary_markdown(facts)
    assert "Row ids removed" in text
    assert "Row ids added" in text


@pytest.mark.unit
def test_outcome_facts_show_any_change_true_when_id_churn_same_row_count():
    rid = APP_SYNTHETIC_ALIGN_ROW_ID_COLUMN
    df_before = pd.DataFrame({rid: ["0", "1"], "v": [1, 2]})
    df_after = pd.DataFrame({rid: ["0", "2"], "v": [1, 3]})
    facts = build_cleaning_outcome_facts(df_before, df_after, row_id_col=rid)
    assert facts["rows"]["n_before"] == facts["rows"]["n_after"]
    assert facts["rows"]["rows_removed_by_id"] == 1
    assert facts["rows"]["rows_added_by_id"] == 1
    assert outcome_facts_show_any_change(facts) is True


@pytest.mark.unit
def test_format_includes_row_counts():
    facts = build_cleaning_outcome_facts(
        pd.DataFrame({"x": [1, 2]}),
        pd.DataFrame({"x": [1]}),
        row_id_col="__missing__",
    )
    text = format_outcome_summary_markdown(facts)
    assert "Row count:" in text
    assert "2" in text and "1" in text


@pytest.mark.unit
def test_outcome_facts_show_any_change_false_when_identical():
    df = pd.DataFrame({"x": [1, 2, 3]})
    facts = build_cleaning_outcome_facts(df, df.copy(), row_id_col="__missing__")
    assert outcome_facts_show_any_change(facts) is False


@pytest.mark.unit
def test_outcome_facts_show_any_change_true_on_row_count_delta():
    facts = build_cleaning_outcome_facts(
        pd.DataFrame({"x": [1, 2]}),
        pd.DataFrame({"x": [1]}),
        row_id_col="__missing__",
    )
    assert outcome_facts_show_any_change(facts) is True
