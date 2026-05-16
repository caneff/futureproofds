"""Execute the hybrid cleaning pipeline from a validated :class:`CleaningPlan`."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from data_cleaning_agent import cleaners
from data_cleaning_agent.cleaners import normalize_column_label
from data_cleaning_agent.cleaning_plan import DEFAULT_ROW_ID_COL, CleaningPlan
from data_cleaning_agent.pipeline_steps import PipelineStep


@dataclass
class PipelineTrace:
    """Records which pipeline steps ran (step id strings)."""

    ran: list[str] = field(default_factory=list)


def _step_skipped(plan: CleaningPlan, step: PipelineStep) -> bool:
    return step.value in plan.skip_steps


def _columns_present(work: pd.DataFrame, columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(column for column in columns if column in work.columns)


def _drop_all_null_data_rows(
    work: pd.DataFrame,
    exclude: tuple[str, ...],
) -> pd.DataFrame:
    """Drop rows that are entirely NA on every non-excluded column."""
    exclude_set = set(exclude)
    check_cols = [col for col in work.columns if col not in exclude_set]
    if not check_cols:
        return work
    return work.dropna(subset=check_cols, how="all")


def _protect_exclude(
    plan: CleaningPlan,
    row_id_col: str,
    *,
    names_normalized: bool,
) -> tuple[str, ...]:
    labels: set[str] = set()
    for label in set(plan.protected_columns) | {row_id_col}:
        if label == row_id_col:
            labels.add(row_id_col)
        elif names_normalized:
            labels.add(normalize_column_label(label))
        else:
            labels.add(label)
    labels.discard(normalize_column_label(row_id_col))
    labels.add(row_id_col)
    return tuple(labels)


def _restore_row_id_column_name(
    work: pd.DataFrame,
    row_id_col: str,
) -> pd.DataFrame:
    """Keep the synthetic row id label unchanged after ``normalize_column_names``."""
    normalized = normalize_column_label(row_id_col)
    if normalized in work.columns and normalized != row_id_col:
        return work.rename(columns={normalized: row_id_col})
    return work


def run_cleaning_pipeline(
    df: pd.DataFrame,
    plan: CleaningPlan,
    *,
    row_id_col: str = DEFAULT_ROW_ID_COL,
) -> tuple[pd.DataFrame, PipelineTrace]:
    """Run fixed-order cleaning steps, honoring ``plan`` skips and parameters."""
    trace = PipelineTrace()
    work = df.copy()
    if not _step_skipped(plan, PipelineStep.COPY):
        trace.ran.append(PipelineStep.COPY.value)

    names_normalized = False
    if not _step_skipped(plan, PipelineStep.NORMALIZE_NAMES):
        work = cleaners.normalize_column_names(work)
        work = _restore_row_id_column_name(work, row_id_col)
        names_normalized = True
        trace.ran.append(PipelineStep.NORMALIZE_NAMES.value)

    drop_exclude = _protect_exclude(plan, row_id_col, names_normalized=names_normalized)

    if not _step_skipped(plan, PipelineStep.DROP_HIGH_MISSING):
        work = cleaners.drop_columns_by_missing(
            work,
            plan.drop_high_missing_threshold,
            exclude=drop_exclude,
        )
        trace.ran.append(PipelineStep.DROP_HIGH_MISSING.value)

    if not _step_skipped(plan, PipelineStep.STRIP_STRINGS):
        work = cleaners.strip_strings(work, exclude=drop_exclude)
        trace.ran.append(PipelineStep.STRIP_STRINGS.value)

    if not _step_skipped(plan, PipelineStep.REPLACE_PLACEHOLDERS):
        work = cleaners.replace_placeholders_with_na(work)
        trace.ran.append(PipelineStep.REPLACE_PLACEHOLDERS.value)

    if not _step_skipped(plan, PipelineStep.COERCE_DTYPES):
        dt_cols = _columns_present(work, plan.coerce_datetime_columns)
        num_cols = _columns_present(work, plan.coerce_numeric_columns)
        bool_cols = _columns_present(work, plan.coerce_bool_columns)
        if dt_cols:
            work = cleaners.coerce_datetime_columns(work, dt_cols)
        if num_cols:
            work = cleaners.coerce_numeric_columns(work, num_cols)
        if bool_cols:
            work = cleaners.coerce_bool_columns(work, bool_cols)
        trace.ran.append(PipelineStep.COERCE_DTYPES.value)

    if not _step_skipped(plan, PipelineStep.DROP_CONSTANT_COLUMNS):
        work = cleaners.drop_constant_columns(work, exclude=drop_exclude)
        trace.ran.append(PipelineStep.DROP_CONSTANT_COLUMNS.value)

    if not _step_skipped(plan, PipelineStep.DROP_ALL_NULL_COLUMNS):
        work = cleaners.drop_all_null_columns(work, exclude=drop_exclude)
        trace.ran.append(PipelineStep.DROP_ALL_NULL_COLUMNS.value)

    if not _step_skipped(plan, PipelineStep.IMPUTE):
        impute_skip = set(drop_exclude)
        for col in _columns_present(work, plan.impute_numeric_columns):
            if col in impute_skip:
                continue
            work[col] = cleaners.impute_numeric_median_or_mean(work[col])
        for col in _columns_present(work, plan.impute_categorical_columns):
            if col in impute_skip:
                continue
            work[col] = cleaners.impute_categorical_mode(work[col])
        trace.ran.append(PipelineStep.IMPUTE.value)

    if not _step_skipped(plan, PipelineStep.DROP_ALL_NULL_ROWS):
        work = _drop_all_null_data_rows(work, drop_exclude)
        trace.ran.append(PipelineStep.DROP_ALL_NULL_ROWS.value)

    if not _step_skipped(plan, PipelineStep.DROP_DUPLICATE_ROWS):
        dedupe_subset = tuple(
            col for col in work.columns if col not in set(drop_exclude)
        )
        work = cleaners.drop_duplicate_rows(work, subset=dedupe_subset)
        trace.ran.append(PipelineStep.DROP_DUPLICATE_ROWS.value)

    if not _step_skipped(plan, PipelineStep.RESET_INDEX):
        work = work.reset_index(drop=True)
        trace.ran.append(PipelineStep.RESET_INDEX.value)

    return work, trace
