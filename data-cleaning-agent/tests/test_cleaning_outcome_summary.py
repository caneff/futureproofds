"""Unit tests for verified cleaning outcome facts (no Streamlit)."""

import numpy as np
import pandas as pd
import pytest
from cleaning_outcome_summary import (
    build_cleaning_outcome_facts,
    format_outcome_summary_markdown,
    outcome_facts_show_any_change,
)
from data_cleaning_agent.utils import APP_SYNTHETIC_ALIGN_ROW_ID_COLUMN


@pytest.mark.unit
def test_dropped_column_tagged_step3_when_high_missing_on_input():
    df_before = pd.DataFrame({"a": range(10), "drop_me": ["", ""] * 5})
    df_after = df_before.drop(columns=["drop_me"])
    facts = build_cleaning_outcome_facts(df_before, df_after, row_id_col="__missing__")
    assert "drop_me" in facts["columns"]["dropped"]
    tags = {t["column"]: t["tag"] for t in facts["drop_reasons"]}
    assert tags.get("drop_me") == "step_3_high_missing"


@pytest.mark.unit
def test_dtype_changed_on_shared_column():
    df_before = pd.DataFrame({"x": ["1", "2", "3"]}, dtype=object)
    df_after = pd.DataFrame({"x": [1, 2, 3]})
    facts = build_cleaning_outcome_facts(df_before, df_after, row_id_col="__missing__")
    assert len(facts["columns"]["dtype_changed"]) >= 1
    entry = facts["columns"]["dtype_changed"][0]
    assert entry["name"] == "x"


@pytest.mark.unit
def test_dtype_drift_suppressed_when_values_match():
    """int64 vs int32 (or similar) alone must not count as a cleaning dtype change."""
    df_before = pd.DataFrame({"x": [1, 2, 3]}, dtype="int64")
    df_after = pd.DataFrame({"x": [1, 2, 3]}, dtype="int32")
    facts = build_cleaning_outcome_facts(df_before, df_after, row_id_col="__missing__")
    assert facts["columns"]["dtype_changed"] == []


@pytest.mark.unit
def test_format_omits_empty_dtype_nulls_and_drop_reason_sections():
    facts = build_cleaning_outcome_facts(
        pd.DataFrame({"x": [1, 2]}),
        pd.DataFrame({"x": [1, 2]}),
        row_id_col="__missing__",
    )
    text = format_outcome_summary_markdown(facts)
    assert "**Dtype Changes**" not in text
    assert "**Missing Value Count Changes" not in text
    assert "**Dropped Columns (Tags)**" not in text


@pytest.mark.unit
def test_rows_section_uses_summarize_when_row_id_present():
    rid = "__agent_row_id__"
    df_before = pd.DataFrame({rid: [0, 1, 2], "v": [1, 2, 3]})
    df_after = pd.DataFrame({rid: [0, 1], "v": [1, 2]})
    facts = build_cleaning_outcome_facts(df_before, df_after, row_id_col=rid)
    assert facts["rows"]["aligned"] is True
    assert facts["rows"]["n_before"] == 3
    assert facts["rows"]["n_after"] == 2
    assert facts["rows"]["removed_total"] == 1


@pytest.mark.unit
def test_rows_aligned_false_when_row_id_missing_in_after():
    rid = "__agent_row_id__"
    df_before = pd.DataFrame({rid: [0, 1], "v": [1, 2]})
    df_after = pd.DataFrame({"v": [1, 2]})
    facts = build_cleaning_outcome_facts(df_before, df_after, row_id_col=rid)
    assert facts["rows"]["aligned"] is False


@pytest.mark.unit
def test_build_facts_survives_duplicate_column_label_on_shared_name():
    """Duplicate labels make df[name] a DataFrame; summary must not use ambiguous truth."""
    df_before = pd.DataFrame(np.ones((2, 2)))
    df_before.columns = ["a", "a"]
    df_after = df_before.copy()
    facts = build_cleaning_outcome_facts(df_before, df_after, row_id_col="__missing__")
    assert facts["rows"]["n_before"] == 2


@pytest.mark.unit
def test_format_redacts_internal_row_id_column_name_in_lists():
    rid = APP_SYNTHETIC_ALIGN_ROW_ID_COLUMN
    facts = {
        "rows": {
            "n_before": 2,
            "n_after": 2,
            "aligned": False,
            "removed_total": None,
            "added_rows_only_in_after": None,
        },
        "columns": {"dropped": [rid, "x"], "added": [], "dtype_changed": []},
        "null_deltas": [
            {"column": rid, "missing_before": 0, "missing_after": 1, "delta": 1}
        ],
        "drop_reasons": [{"column": rid, "tag": "dropped"}],
    }
    text = format_outcome_summary_markdown(facts)
    assert rid not in text
    assert "synthetic alignment column (app-injected)" in text


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
