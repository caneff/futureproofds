import pytest

from data_cleaning_agent.pipeline_steps import (
    PIPELINE_STEP_ORDER,
    PipelineStep,
)


def _runs_before(earlier: PipelineStep, later: PipelineStep) -> bool:
    return PIPELINE_STEP_ORDER.index(earlier) < PIPELINE_STEP_ORDER.index(later)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("earlier", "later"),
    [
        pytest.param(
            PipelineStep.COPY,
            PipelineStep.NORMALIZE_NAMES,
            id="copy_before_rename",
        ),
        pytest.param(
            PipelineStep.DROP_HIGH_MISSING,
            PipelineStep.STRIP_STRINGS,
            id="drop_high_missing_before_strip",
        ),
        pytest.param(
            PipelineStep.REPLACE_PLACEHOLDERS,
            PipelineStep.COERCE_DTYPES,
            id="placeholders_before_coerce",
        ),
        pytest.param(
            PipelineStep.IMPUTE,
            PipelineStep.DROP_ALL_NULL_ROWS,
            id="impute_before_drop_empty_rows",
        ),
        pytest.param(
            PipelineStep.DROP_DUPLICATE_ROWS,
            PipelineStep.RESET_INDEX,
            id="dedupe_before_reset_index",
        ),
    ],
)
def test_data_cleaning_prompt_step_precedence(
    earlier: PipelineStep,
    later: PipelineStep,
) -> None:
    assert _runs_before(earlier, later)


@pytest.mark.unit
@pytest.mark.parametrize("invalid_id", ["", "not_a_step", "COPY"])
def test_pipeline_step_rejects_unknown_step_id(invalid_id: str) -> None:
    with pytest.raises(ValueError):
        PipelineStep(invalid_id)
