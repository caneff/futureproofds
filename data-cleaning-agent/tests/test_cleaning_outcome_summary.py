"""Unit tests for verified cleaning outcome facts (no Streamlit)."""

import pandas as pd
import pytest
from cleaning_outcome_summary import (
    build_cleaning_outcome_facts,
    format_outcome_summary_markdown,
)


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
def test_format_includes_row_counts():
    facts = build_cleaning_outcome_facts(
        pd.DataFrame({"x": [1, 2]}),
        pd.DataFrame({"x": [1]}),
        row_id_col="__missing__",
    )
    text = format_outcome_summary_markdown(facts, row_id_label="row id")
    assert "Row count:" in text
    assert "2" in text and "1" in text
