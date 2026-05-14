"""Tests for plan column coercion and removed-action diffing."""

import pytest
from data_cleaning_agent.utils import (
    coerce_cleaning_plan_columns,
    merged_plan_actions_by_column,
    multiset_union_removed_plan_pairs,
    removed_plan_actions,
)


def test_multiset_union_removed_plan_pairs_sorts_and_counts():
    a = [("x", "1"), ("y", "2")]
    b = [("x", "1"), ("z", "3")]
    u = multiset_union_removed_plan_pairs(a, b)
    assert u == [("x", "1"), ("x", "1"), ("y", "2"), ("z", "3")]


def test_removed_plan_actions_basic():
    before = [
        {"name": "a", "actions": ["x", "y"]},
        {"name": "b", "actions": ["z"]},
    ]
    after = [
        {"name": "a", "actions": ["x"]},
        {"name": "b", "actions": ["z"]},
    ]
    assert removed_plan_actions(before, after) == [("a", "y")]


def test_removed_plan_actions_duplicate_multiset():
    before = [{"name": "a", "actions": ["x", "x", "y"]}]
    after = [{"name": "a", "actions": ["x", "y"]}]
    assert removed_plan_actions(before, after) == [("a", "x")]


def test_removed_plan_actions_drops_entire_column():
    before = [{"name": "a", "actions": ["x"]}, {"name": "b", "actions": ["z"]}]
    after = [{"name": "b", "actions": ["z"]}]
    assert removed_plan_actions(before, after) == [("a", "x")]


def test_removed_plan_actions_sorted_output():
    before = [
        {"name": "b", "actions": ["2"]},
        {"name": "a", "actions": ["1"]},
    ]
    after = [{"name": "a", "actions": []}, {"name": "b", "actions": []}]
    assert removed_plan_actions(before, after) == [("a", "1"), ("b", "2")]


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


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ([], [], []),
        ([{"name": "a", "actions": []}], [{"name": "a", "actions": []}], []),
    ],
)
def test_removed_plan_actions_empty(before, after, expected):
    assert removed_plan_actions(before, after) == expected
