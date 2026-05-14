"""Tests for plan column coercion and merged actions."""

from data_cleaning_agent.utils import (
    coerce_cleaning_plan_columns,
    merged_plan_actions_by_column,
)


def test_coerce_cleaning_plan_columns_dict_shape():
    before = {"col_a": ["p", "q"], "col_b": "single"}
    rows = coerce_cleaning_plan_columns(before)
    assert len(rows) == 2
    by_name = {r["name"]: r["actions"] for r in rows}
    assert by_name["col_a"] == ["p", "q"]
    assert by_name["col_b"] == ["single"]


def test_merged_plan_actions_duplicate_column_rows():
    """LLM often splits one logical column across multiple plan rows."""
    cols = [
        {"name": "city", "actions": ["strip whitespace"]},
        {"name": "city", "actions": ["impute missing values (mode)"]},
        {"name": "employee_id", "actions": ["note high missing rate"]},
        {"name": "employee_id", "actions": ["drop column (>40% missing)"]},
    ]
    m = merged_plan_actions_by_column(cols)
    assert list(m.keys()) == ["city", "employee_id"]
    assert m["city"] == ["strip whitespace", "impute missing values (mode)"]
    assert m["employee_id"] == [
        "note high missing rate",
        "drop column (>40% missing)",
    ]
