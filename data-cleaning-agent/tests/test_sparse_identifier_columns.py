"""Regression tests for sparse identifier columns (e.g. ``employee_id``) vs pipeline step 3."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from data_cleaning_agent.utils import (
    find_retained_sparse_identifier_columns,
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
def test_find_retained_flags_high_missing_employee_id_when_column_kept():
    df_before = pd.DataFrame({"employee_id": [None] * 9 + ["E1"], "x": range(10)})
    df_after = df_before.copy()
    assert find_retained_sparse_identifier_columns(df_before, df_after) == [
        "employee_id"
    ]


@pytest.mark.unit
def test_find_retained_empty_when_sparse_identifier_dropped():
    df_before = pd.DataFrame({"employee_id": [None] * 9 + ["E1"], "x": range(10)})
    df_after = df_before.drop(columns=["employee_id"])
    assert find_retained_sparse_identifier_columns(df_before, df_after) == []


@pytest.mark.unit
def test_find_retained_respects_protected_columns():
    df_before = pd.DataFrame({"employee_id": [None] * 9 + ["E1"], "x": range(10)})
    df_after = df_before.copy()
    assert (
        find_retained_sparse_identifier_columns(
            df_before,
            df_after,
            protected=frozenset({"employee_id"}),
        )
        == []
    )


@pytest.mark.unit
def test_sample_data_employee_id_exceeds_sparse_drop_threshold_in_summary():
    """Guards the prompt contract: step 3 should drop this column unless user exempts it."""
    csv_path = Path(__file__).resolve().parents[1] / "data" / "sample_data.csv"
    df = pd.read_csv(csv_path)
    summary = get_dataframe_summary(df)
    emp = summary.columns["employee_id"]
    assert emp.missing_pct > 40.0
    assert emp.id_like is False


@pytest.mark.unit
def test_find_retained_catches_noop_cleaner_that_keeps_sparse_employee_id():
    """A cleaner that copies the frame should be flagged when employee_id stays sparse."""

    def data_cleaner(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    df_in = pd.DataFrame({"employee_id": [None] * 8 + ["A", "B"], "k": range(10)})
    df_out = data_cleaner(df_in)
    assert find_retained_sparse_identifier_columns(df_in, df_out, threshold=0.4) == [
        "employee_id"
    ]
