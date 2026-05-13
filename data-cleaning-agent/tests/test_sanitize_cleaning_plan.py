"""Tests for :func:`data_cleaning_agent.utils.sanitize_cleaning_plan`."""

import pandas as pd
import pytest
from data_cleaning_agent.utils import (
    normalize_cleaning_column_name,
    sanitize_cleaning_plan,
)


@pytest.mark.unit
def test_normalize_matches_pipeline_step2():
    assert normalize_cleaning_column_name("Employee ID") == "employee_id"
    assert normalize_cleaning_column_name("  Foo-Bar  ") == "foo_bar"


@pytest.mark.unit
def test_sanitize_drops_hallucinated_column():
    df = pd.DataFrame({"city": ["a"], "sales": [1.0]})
    plan = {
        "columns": [
            {"name": "city", "actions": ["strip"]},
            {"name": "employee_id", "actions": ["impute"]},
        ],
        "row_ops": [],
        "notes": "",
    }
    out = sanitize_cleaning_plan(plan, df)
    assert len(out["columns"]) == 1
    assert out["columns"][0]["name"] == "city"
    assert "employee_id" in out["notes"]
    assert "Plan rows removed" in out["notes"]


@pytest.mark.unit
def test_sanitize_keeps_normalized_and_raw_suffix():
    df = pd.DataFrame({"City Name": [1, 2]})
    plan = {
        "columns": [
            {"name": "city_name", "actions": ["normalize"]},
            {"name": "city_name_raw", "actions": ["add _raw"]},
        ],
        "row_ops": [],
        "notes": "",
    }
    out = sanitize_cleaning_plan(plan, df)
    names = {c["name"] for c in out["columns"]}
    assert names == {"city_name", "city_name_raw"}


@pytest.mark.unit
def test_sanitize_none_returns_none():
    assert sanitize_cleaning_plan(None, pd.DataFrame()) is None


@pytest.mark.unit
def test_sanitize_drops_synthetic_row_id_even_when_in_dataframe():
    df = pd.DataFrame({"__agent_row_id__": [0, 1], "city": ["a", "b"]})
    plan = {
        "columns": [
            {"name": "__agent_row_id__", "actions": ["carry through"]},
            {"name": "city", "actions": ["strip"]},
        ],
        "row_ops": [],
        "notes": "",
    }
    out = sanitize_cleaning_plan(plan, df)
    names = [c["name"] for c in out["columns"]]
    assert "__agent_row_id__" not in names
    assert "agent_row_id" not in names
    assert "city" in names


@pytest.mark.unit
def test_sanitize_drops_normalized_alias_of_synthetic_row_id():
    df = pd.DataFrame({"__agent_row_id__": [0], "x": [1]})
    plan = {
        "columns": [
            {"name": "agent_row_id", "actions": ["normalize name"]},
            {"name": "x", "actions": ["strip"]},
        ],
        "row_ops": [],
        "notes": "",
    }
    out = sanitize_cleaning_plan(plan, df)
    names = {c["name"] for c in out["columns"]}
    assert "agent_row_id" not in names
    assert "x" in names


@pytest.mark.unit
def test_sanitize_dict_shaped_columns_is_filtered():
    """Object-shaped ``columns`` must not bypass filtering (regression)."""
    df = pd.DataFrame({"city": ["a"], "sales": [1.0]})
    plan = {
        "columns": {
            "city": ["strip"],
            "employee_id": ["impute"],
        },
        "row_ops": [],
        "notes": "",
    }
    out = sanitize_cleaning_plan(plan, df)
    assert len(out["columns"]) == 1
    assert out["columns"][0]["name"] == "city"
    assert "employee_id" in out["notes"]


@pytest.mark.unit
def test_sanitize_non_list_non_dict_columns_becomes_empty():
    df = pd.DataFrame({"x": [1]})
    plan = {"columns": "not a list", "row_ops": [], "notes": ""}
    out = sanitize_cleaning_plan(plan, df)
    assert out["columns"] == []
