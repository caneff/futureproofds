"""Stable step ids for the hybrid cleaning pipeline."""

from enum import StrEnum


class PipelineStep(StrEnum):
    """Pipeline steps; declare members in execution order (values = plan JSON / prompts)."""

    COPY = "copy"
    NORMALIZE_NAMES = "normalize_names"
    DROP_HIGH_MISSING = "drop_high_missing"
    STRIP_STRINGS = "strip_strings"
    REPLACE_PLACEHOLDERS = "replace_placeholders"
    COERCE_DTYPES = "coerce_dtypes"
    DROP_CONSTANT_COLUMNS = "drop_constant_columns"
    DROP_ALL_NULL_COLUMNS = "drop_all_null_columns"
    IMPUTE = "impute"
    DROP_ALL_NULL_ROWS = "drop_all_null_rows"
    DROP_DUPLICATE_ROWS = "drop_duplicate_rows"
    RESET_INDEX = "reset_index"


PIPELINE_STEP_ORDER: tuple[PipelineStep, ...] = tuple(PipelineStep)

ALL_STEP_IDS: frozenset[str] = frozenset(step.value for step in PipelineStep)
