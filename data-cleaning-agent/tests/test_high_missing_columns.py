"""Tests for step 3 (high missing share) and runtime enforcement."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from data_cleaning_agent.utils import (
    apply_high_missing_column_enforcement,
    find_retained_high_missing_columns,
    get_dataframe_summary,
    sparse_missing_share,
)


@pytest.mark.unit
def test_sparse_missing_share_counts_placeholders_and_blanks():
    s = pd.Series(["", "  ", "N/A", "x", None], dtype=object)
    assert sparse_missing_share(s) == pytest.approx(0.8)


@pytest.mark.unit
def test_sparse_missing_share_numeric_matches_na_fraction():
    s = pd.Series([1.0, np.nan, 3.0])
    assert sparse_missing_share(s) == pytest.approx(1.0 / 3.0)


@pytest.mark.unit
def test_find_retained_flags_high_missing_column_without_id_suffix():
    df_before = pd.DataFrame({"comment": [None] * 9 + ["ok"], "x": range(10)})
    df_after = df_before.copy()
    assert find_retained_high_missing_columns(df_before, df_after) == ["comment"]


@pytest.mark.unit
def test_find_retained_flags_high_missing_employee_id_when_column_kept():
    df_before = pd.DataFrame({"employee_id": [None] * 9 + ["E1"], "x": range(10)})
    df_after = df_before.copy()
    assert find_retained_high_missing_columns(df_before, df_after) == ["employee_id"]


@pytest.mark.unit
def test_find_retained_empty_when_high_missing_column_dropped():
    df_before = pd.DataFrame({"employee_id": [None] * 9 + ["E1"], "x": range(10)})
    df_after = df_before.drop(columns=["employee_id"])
    assert find_retained_high_missing_columns(df_before, df_after) == []


@pytest.mark.unit
def test_find_retained_respects_protected_columns():
    df_before = pd.DataFrame({"employee_id": [None] * 9 + ["E1"], "x": range(10)})
    df_after = df_before.copy()
    assert (
        find_retained_high_missing_columns(
            df_before,
            df_after,
            protected=frozenset({"employee_id"}),
        )
        == []
    )


@pytest.mark.unit
def test_sample_data_employee_id_exceeds_drop_threshold_in_summary():
    """Sample CSV: employee_id is high-missing under the same rules as step 3."""
    csv_path = Path(__file__).resolve().parents[1] / "data" / "sample_data.csv"
    df = pd.read_csv(csv_path)
    summary = get_dataframe_summary(df)
    emp = summary.columns["employee_id"]
    assert emp.missing_pct > 40.0
    assert emp.id_like is False


@pytest.mark.unit
def test_find_retained_catches_noop_cleaner_that_keeps_high_missing_column():
    df_in = pd.DataFrame({"employee_id": [None] * 8 + ["A", "B"], "k": range(10)})
    df_out = df_in.copy()
    assert find_retained_high_missing_columns(df_in, df_out, threshold=0.4) == [
        "employee_id"
    ]


@pytest.mark.unit
def test_enforcement_drops_retained_high_missing_employee_id():
    df_before = pd.DataFrame({"employee_id": [None] * 9 + ["E1"], "x": range(10)})
    df_after = df_before.copy()
    out = apply_high_missing_column_enforcement(df_before, df_after)
    assert "employee_id" not in out.columns
    assert list(out.columns) == ["x"]


@pytest.mark.unit
def test_enforcement_drops_retained_high_missing_non_id_column():
    df_before = pd.DataFrame({"notes": ["", ""] * 5, "k": range(10)})
    df_after = df_before.copy()
    out = apply_high_missing_column_enforcement(df_before, df_after)
    assert "notes" not in out.columns
    assert "k" in out.columns


@pytest.mark.unit
def test_enforcement_never_drops_named_row_id_column_even_if_high_missing():
    rid = "synth_row_id"
    df_before = pd.DataFrame({rid: [None] * 9 + [1], "x": range(10)})
    df_after = df_before.copy()
    out = apply_high_missing_column_enforcement(df_before, df_after, row_id_col=rid)
    assert rid in out.columns
    assert "x" in out.columns


@pytest.mark.unit
def test_enforcement_respects_protected_columns():
    df_before = pd.DataFrame({"employee_id": [None] * 9 + ["E1"], "x": range(10)})
    df_after = df_before.copy()
    out = apply_high_missing_column_enforcement(
        df_before,
        df_after,
        protected=frozenset({"employee_id"}),
    )
    assert "employee_id" in out.columns
