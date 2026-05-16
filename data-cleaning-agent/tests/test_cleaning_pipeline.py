import numpy as np
import pandas as pd
import pytest

from data_cleaning_agent.cleaning_pipeline import run_cleaning_pipeline
from data_cleaning_agent.cleaning_plan import CleaningPlan
from data_cleaning_agent.pipeline_steps import PipelineStep

ROW_ID = "__agent_row_id__"


@pytest.mark.unit
def test_pipeline_drops_all_null_row() -> None:
    # Column missing rate must stay below default drop_high_missing threshold (0.4).
    df = pd.DataFrame({
        ROW_ID: ["0", "1", "2"],
        "a": [1.0, 2.0, np.nan],
        "b": [3.0, 4.0, np.nan],
    })
    plan = CleaningPlan()
    out, _trace = run_cleaning_pipeline(df, plan, row_id_col=ROW_ID)
    assert len(out) == 2
    assert out.iloc[0]["a"] == 1.0


@pytest.mark.unit
def test_pipeline_drops_duplicate_rows() -> None:
    df = pd.DataFrame({
        ROW_ID: ["0", "1", "2"],
        "x": [1, 1, 2],
        "y": ["a", "a", "b"],
    })
    plan = CleaningPlan()
    out, _trace = run_cleaning_pipeline(df, plan, row_id_col=ROW_ID)
    assert len(out) == 2
    assert set(out[ROW_ID]) == {"0", "2"}


@pytest.mark.unit
def test_pipeline_skip_impute_leaves_nan() -> None:
    df = pd.DataFrame({
        ROW_ID: ["0", "1", "2"],
        "score": [1.0, np.nan, 2.0],
        "tag": ["a", "b", "c"],
    })
    plan = CleaningPlan(
        skip_steps=[PipelineStep.IMPUTE.value],
        impute_numeric_columns=("score",),
    )
    out, trace = run_cleaning_pipeline(df, plan, row_id_col=ROW_ID)
    assert PipelineStep.IMPUTE.value not in trace.ran
    assert pd.isna(out.iloc[1]["score"])


@pytest.mark.unit
def test_pipeline_preserves_row_id_column() -> None:
    df = pd.DataFrame({
        ROW_ID: ["r1", "r2"],
        "value": [1.0, 2.0],
    })
    plan = CleaningPlan(protected_columns=[ROW_ID])
    out, _trace = run_cleaning_pipeline(df, plan, row_id_col=ROW_ID)
    assert ROW_ID in out.columns
    assert list(out[ROW_ID]) == ["r1", "r2"]


@pytest.mark.unit
def test_pipeline_impute_skips_column_dropped_by_missing_threshold() -> None:
    df = pd.DataFrame({
        ROW_ID: ["0", "1"],
        "keep": [1.0, 2.0],
        "mostly_missing": [np.nan, np.nan],
    })
    plan = CleaningPlan(
        drop_high_missing_threshold=0.5,
        impute_numeric_columns=("mostly_missing", "keep"),
        protected_columns=[ROW_ID, "keep"],
    )
    out, _trace = run_cleaning_pipeline(df, plan, row_id_col=ROW_ID)
    assert "mostly_missing" not in out.columns
    assert "keep" in out.columns
