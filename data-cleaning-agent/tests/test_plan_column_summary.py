import pytest

from data_cleaning_agent.plan_column_summary import plan_columns_to_summary_rows


def test_none_plan_returns_empty():
    assert plan_columns_to_summary_rows(None) == []


def test_empty_columns_returns_empty():
    assert plan_columns_to_summary_rows({"columns": []}) == []


def test_merges_duplicate_column_rows():
    plan = {
        "columns": [
            {"name": "city", "actions": ["strip whitespace"]},
            {"name": "city", "actions": ["impute missing values (mode)"]},
        ]
    }
    rows = plan_columns_to_summary_rows(plan)
    assert rows == [
        {
            "column": "city",
            "actions": "strip whitespace; impute missing values (mode)",
        }
    ]


def test_dict_shaped_columns_coerced():
    plan = {"columns": {"a": ["x"], "b": "y"}}
    rows = plan_columns_to_summary_rows(plan)
    by = {r["column"]: r["actions"] for r in rows}
    assert by["a"] == "x"
    assert by["b"] == "y"
