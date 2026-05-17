"""Tests for row-effect helpers used with cleaning previews."""

import pandas as pd
import pytest

from data_cleaning_agent.utils import (
    APP_SYNTHETIC_ALIGN_ROW_ID_COLUMN,
    summarize_cleaning_row_effects,
)


@pytest.mark.unit
def test_summarize_row_effects_counts_removed_and_all_null_subset():
    rid = APP_SYNTHETIC_ALIGN_ROW_ID_COLUMN
    df_in = pd.DataFrame({
        rid: [0, 1, 2, 3],
        "a": [1.0, None, None, 2.0],
        "b": ["x", None, None, "y"],
    })
    df_out = pd.DataFrame({
        rid: [0, 3],
        "a": [1.0, 2.0],
        "b": ["x", "y"],
    })
    out = summarize_cleaning_row_effects(df_in, df_out, row_id_col=rid)
    assert out["n_in"] == 4
    assert out["n_out"] == 2
    assert out["removed_total"] == 2
    assert out["rows_removed_by_id"] == 2
    assert out["rows_added_by_id"] == 0
    assert out["removed_all_null_input_user_cols"] == 2


@pytest.mark.unit
def test_summarize_row_effects_without_row_id_column():
    df_in = pd.DataFrame({"x": [1, 2]})
    df_out = pd.DataFrame({"x": [1]})
    out = summarize_cleaning_row_effects(df_in, df_out, row_id_col="__missing__")
    assert out["removed_total"] == 1
    assert out["rows_removed_by_id"] is None
    assert out["rows_added_by_id"] is None
    assert out["removed_all_null_input_user_cols"] is None


@pytest.mark.unit
def test_summarize_row_effects_counts_added_ids():
    rid = APP_SYNTHETIC_ALIGN_ROW_ID_COLUMN
    df_in = pd.DataFrame({rid: ["0", "1"], "v": [1, 2]})
    df_out = pd.DataFrame({rid: ["0", "1", "2"], "v": [1, 2, 3]})
    out = summarize_cleaning_row_effects(df_in, df_out, row_id_col=rid)
    assert out["rows_removed_by_id"] == 0
    assert out["rows_added_by_id"] == 1
