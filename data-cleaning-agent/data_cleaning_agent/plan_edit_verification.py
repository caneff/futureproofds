"""Execution-based verification after plan-edit regeneration (scope B)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from data_cleaning_agent.utils import (
    APP_SYNTHETIC_ALIGN_ROW_ID_COLUMN,
    first_column_as_series,
    merged_plan_actions_by_column,
    normalize_cleaning_column_name,
)

CheckKind = Literal["impute", "drop", "dtype"]


def classify_removed_action(action: str) -> CheckKind | None:
    """
    Map a removed plan action string to a verification bucket.

    Order: drop > dtype > impute so compound phrases favor structural checks.
    """
    t = str(action).strip().lower()
    if not t:
        return None

    drop_pat = re.compile(
        r"\b(drop|dropping)\b.*\b(column|col)\b|"
        r"\b(remove|removing|delete|deleting)\b.*\b(column|col)\b|"
        r"\bdrop\s+column\b",
        re.I,
    )
    if drop_pat.search(t) or (re.search(r"\bdrop\b", t) and "column" in t):
        return "drop"

    dtype_pat = re.compile(
        r"\bdtype\b|astype|coerce|to_numeric|to_datetime|"
        r"\bconvert(ed)?\s+to\b|parse\s+as|"
        r"\btype\s+conversion\b|\bcasting\b",
        re.I,
    )
    if dtype_pat.search(t):
        return "dtype"

    # "retain missing values" contains "missing … values" but is not imputation.
    if re.search(r"\bretain\b\s+missing\b", t, re.I):
        return None

    impute_pat = re.compile(
        r"\bimpute\b|\bfillna\b|\bmedian\b|\bmean\b|\bmode\b|"
        r"\bbfill\b|\bffill\b|forward\s+fill|back(ward)?\s+fill|"
        r"fill\s+missing|missing\s+values?|"
        r"\binterpolation\b|\bfill\s+null\b",
        re.I,
    )
    if impute_pat.search(t):
        return "impute"

    return None


_RETAIN_MISSING_RE = re.compile(r"\bretain\b\s+missing\b", re.I)


def plan_column_lists_retain_missing_action(
    plan: dict | None, column_label: str
) -> bool:
    """
    True if merged ``columns`` include an explicit retain-missing wording for the column.

    Matches the same ``retain … missing`` branch used in :func:`classify_removed_action`
    so plan JSON that documents keeping nulls is not treated as undocumented missingness.
    """
    if plan is None or not isinstance(plan, dict):
        return False
    target = normalize_cleaning_column_name(column_label)
    if not target:
        return False
    merged = merged_plan_actions_by_column(plan.get("columns"))
    for plan_name, actions in merged.items():
        if normalize_cleaning_column_name(str(plan_name)) != target:
            continue
        for act in actions:
            if _RETAIN_MISSING_RE.search(str(act)):
                return True
    return False


def resolve_plan_column_name(plan_column: str, df: pd.DataFrame) -> str | None:
    """
    Resolve a plan ``name`` string to a column label in ``df`` using step-2
    normalization (see :func:`normalize_cleaning_column_name`).
    """
    target = normalize_cleaning_column_name(plan_column)
    if not target:
        return None
    for c in df.columns:
        if normalize_cleaning_column_name(str(c)) == target:
            return str(c)
    return None


@dataclass
class ClassifiedFailure:
    """One classified removed-step check that failed."""

    plan_column: str
    check_type: CheckKind
    reason: str


@dataclass
class VerificationResult:
    """Outcome of :func:`verify_removed_plan_steps`."""

    ok: bool
    classified_failures: list[ClassifiedFailure] = field(default_factory=list)
    unclassified_removed: list[tuple[str, str]] = field(default_factory=list)


def _null_counts_aligned(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    col_before: str,
    col_after: str,
    row_id_col: str | None,
) -> tuple[int, int, str]:
    """
    Return (null_before, null_after, mode).

    When ``row_id_col`` is present in both frames, count nulls on the inner join
    of row ids; otherwise compare the full column (positional / length mismatch
    risk if rows are dropped — conservative for v1).
    """
    if (
        row_id_col
        and row_id_col in df_before.columns
        and row_id_col in df_after.columns
    ):
        left = df_before[[row_id_col, col_before]].copy()
        right = df_after[[row_id_col, col_after]].copy()
        left = left.rename(columns={col_before: "_v_before"})
        right = right.rename(columns={col_after: "_v_after"})
        merged = left.merge(right, on=row_id_col, how="inner")
        if merged.empty and len(df_before) > 0:
            return (
                int(first_column_as_series(df_before, col_before).isna().sum()),
                int(first_column_as_series(df_after, col_after).isna().sum()),
                "row_id_inner_join_empty",
            )
        nb = int(merged["_v_before"].isna().sum())
        na = int(merged["_v_after"].isna().sum())
        return nb, na, "row_id_aligned"
    nb = int(first_column_as_series(df_before, col_before).isna().sum())
    na = int(first_column_as_series(df_after, col_after).isna().sum())
    return nb, na, "full_column"


def verify_removed_plan_steps(
    removed: list[tuple[str, str]],
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    *,
    row_id_col: str | None = APP_SYNTHETIC_ALIGN_ROW_ID_COLUMN,
) -> VerificationResult:
    """
    Run scope-B predicates for each removed (column, action) pair.

    Unclassified actions are recorded but do not fail verification.
    """
    failures: list[ClassifiedFailure] = []
    unclassified: list[tuple[str, str]] = []

    for plan_col, action in removed:
        kind = classify_removed_action(action)
        if kind is None:
            unclassified.append((plan_col, action))
            continue

        col_b = resolve_plan_column_name(plan_col, df_before)
        col_a = resolve_plan_column_name(plan_col, df_after)

        if kind == "drop":
            if col_b is None:
                failures.append(
                    ClassifiedFailure(
                        plan_col,
                        "drop",
                        f'Plan column "{plan_col}" not found in input DataFrame; '
                        "cannot verify column is retained.",
                    )
                )
                continue
            if col_a is None:
                failures.append(
                    ClassifiedFailure(
                        plan_col,
                        "drop",
                        f'Column "{col_b}" is missing from cleaner output; user removed '
                        "a drop-column step so the column must remain.",
                    )
                )
                continue

        if kind in ("impute", "dtype"):
            if col_b is None:
                failures.append(
                    ClassifiedFailure(
                        plan_col,
                        kind,
                        f'Plan column "{plan_col}" not found in input DataFrame.',
                    )
                )
                continue
            if col_a is None:
                failures.append(
                    ClassifiedFailure(
                        plan_col,
                        kind,
                        f'Column "{col_b}" missing after cleaning (cannot compare).',
                    )
                )
                continue

        if kind == "impute":
            nb, na, mode = _null_counts_aligned(
                df_before, df_after, col_b, col_a, row_id_col
            )
            if na < nb:
                mode_note = f" (comparison mode: {mode}; nulls before={nb}, after={na})"
                failures.append(
                    ClassifiedFailure(
                        plan_col,
                        "impute",
                        "Missing value count decreased after cleaning; user removed an "
                        f"imputation step so nulls must not decrease.{mode_note}",
                    )
                )

        if kind == "dtype":
            s_b = first_column_as_series(df_before, col_b)
            s_a = first_column_as_series(df_after, col_a)
            dt_b = s_b.dtype
            dt_a = s_a.dtype
            if dt_b != dt_a:
                failures.append(
                    ClassifiedFailure(
                        plan_col,
                        "dtype",
                        f"dtype changed from {dt_b!r} to {dt_a!r} for column "
                        f'"{col_b}"; user removed a dtype/coercion step.',
                    )
                )

    return VerificationResult(
        ok=len(failures) == 0,
        classified_failures=failures,
        unclassified_removed=unclassified,
    )


def compose_plan_regen_supplemental(
    base_instructions: str,
    plan_exclusion_block: str,
    *,
    prior_verification_feedback: str | None = None,
    automatic_retry_failure_block: str | None = None,
    user_follow_up: str | None = None,
) -> str:
    """
    Merge supplemental text for **Regenerate Code to Match Plan**.

    Order: base host notes → plan-edit exclusion → prior verification feedback
    (session) → automatic-retry failure (same click) → optional user notes.
    """
    parts: list[str] = [
        base_instructions.strip(),
        "---",
        "**Plan-edit exclusion (application UI; follow in addition "
        "to the host supplemental notes above):**",
        plan_exclusion_block.strip(),
    ]
    fb = (prior_verification_feedback or "").strip()
    if fb:
        parts.extend(["---", "**Prior automated verification findings:**", fb])
    ar = (automatic_retry_failure_block or "").strip()
    if ar:
        parts.extend([
            "---",
            "**Latest verification failure (this regenerate, first attempt):**",
            ar,
        ])
    uc = (user_follow_up or "").strip()
    if uc:
        parts.extend(["---", "**User follow-up (must honor):**", uc])
    return "\n\n".join(parts)


def format_verification_feedback_markdown(result: VerificationResult) -> str:
    """
    Host-facing markdown for LLM supplemental, read-only UI, and auto-retry.

    Includes imperative **DO NOT** / **must** language grouped by check type so
    the model gets the same clarity users often add in manual notes.
    """
    lines = [
        "The last regenerated cleaner **failed automated verification** (the code "
        "ran, but checks on removed plan steps did not pass).",
        "",
        "**Failing checks:**",
    ]
    for f in result.classified_failures:
        lines.append(f"- **`{f.plan_column}`** (`{f.check_type}`): {f.reason}")

    impute_cols = sorted({
        f.plan_column for f in result.classified_failures if f.check_type == "impute"
    })
    drop_cols = sorted({
        f.plan_column for f in result.classified_failures if f.check_type == "drop"
    })
    dtype_cols = sorted({
        f.plan_column for f in result.classified_failures if f.check_type == "dtype"
    })

    lines.append("")
    lines.append("**Host requirements for this revision (treat as hard constraints):**")
    lines.append("")

    if impute_cols:
        lines.append(
            "- **Imputation check:** The user **removed** an imputation step for at "
            "least one of these plan columns, but **missing-value counts still went "
            "down** after your function ran (so values were filled or otherwise "
            "replaced). **DO NOT** impute or fill those columns in the next revision:"
        )
        for c in impute_cols:
            lines.append(f"  - `{c}`")
        lines.append(
            "  Concretely: do **not** use `fillna`, `replace` to substitute for NaN, "
            "`bfill`/`ffill`, `interpolate`, mode/mean/median **fills**, or any "
            "assignment that reduces nulls on these columns (including inside **step "
            '9** or any loop over "columns to impute"). **Skip** these names entirely '
            "in imputation logic—same spirit as plan-edit exclusions from the host UI."
        )
        lines.append("")

    if drop_cols:
        lines.append(
            "- **Column-retention check:** The user **removed** a planned **column "
            "drop**, but the column is still missing from the output. These plan columns "
            "**must appear** in the returned DataFrame (use the normalized names your "
            "pipeline uses after step 2):"
        )
        for c in drop_cols:
            lines.append(f"  - `{c}`")
        lines.append(
            "  Do **not** `drop` these columns; do not `del` them; do not omit them when "
            "rebuilding `df` before `return`."
        )
        lines.append("")

    if dtype_cols:
        lines.append(
            "- **Dtype check:** The user **removed** a dtype/coercion step, but the "
            "pandas **dtype still changed** for that column. For the next revision, "
            "the cleaned column’s **dtype must match** the input column’s dtype at the "
            "point you read it (after any renames your code applies), for:"
        )
        for c in dtype_cols:
            lines.append(f"  - `{c}`")
        lines.append(
            "  Do **not** call `astype`, `pd.to_numeric`, `to_datetime`, or other "
            "coercions on these columns unless the main pipeline prompt already required "
            "them independently of the removed plan step."
        )
        lines.append("")

    lines.append(
        "**Instructions:** Implement the constraints above in Python only. Keep the "
        "rest of the pipeline coherent and deterministic. Return a JSON cleaning plan "
        "that matches what the revised function **actually** does."
    )
    return "\n".join(lines)


def format_unclassified_warning_markdown(
    pairs: list[tuple[str, str]],
) -> str:
    """Short markdown / plain text for ``st.warning``."""
    if not pairs:
        return ""
    bullets = "\n".join(f"- `{col}`: {act!r}" for col, act in pairs)
    return (
        "These removed steps are not covered by automatic checks yet; please "
        f"confirm the preview:\n{bullets}"
    )


def plan_column_lists_imputation_action(plan: dict | None, column_label: str) -> bool:
    """
    True if merged ``columns`` include an imputation-classified action for the column.

    Uses the same keyword rules as :func:`classify_removed_action` when that
    function returns the ``impute`` kind.
    """
    if plan is None or not isinstance(plan, dict):
        return False
    target = normalize_cleaning_column_name(column_label)
    if not target:
        return False
    merged = merged_plan_actions_by_column(plan.get("columns"))
    for plan_name, actions in merged.items():
        if normalize_cleaning_column_name(str(plan_name)) != target:
            continue
        for act in actions:
            if classify_removed_action(str(act)) == "impute":
                return True
    return False


def columns_where_missingness_dropped_without_plan_imputation(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    plan: dict | None,
    *,
    row_id_col: str | None = APP_SYNTHETIC_ALIGN_ROW_ID_COLUMN,
) -> list[str]:
    """
    Columns whose aligned missing-count decreased without plan text for that change.

    Skips a column when the merged plan lists an **imputation**-classified action or
    an explicit **retain missing** phrase, so documented intent is not flagged as a
    ghost imputation omission.

    Still flags columns where nulls drop but the plan has neither (model drift).
    """
    bcols = set(df_before.columns)
    acols = set(df_after.columns)
    shared = sorted(bcols & acols)
    flagged: list[str] = []
    for col in shared:
        if row_id_col and col == row_id_col:
            continue
        nb, na, _mode = _null_counts_aligned(df_before, df_after, col, col, row_id_col)
        if na >= nb:
            continue
        if plan_column_lists_imputation_action(plan, col):
            continue
        if plan_column_lists_retain_missing_action(plan, col):
            continue
        flagged.append(col)
    return sorted(flagged)


def columns_where_retain_missing_plan_violated_by_execution(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    plan: dict | None,
    *,
    row_id_col: str | None = APP_SYNTHETIC_ALIGN_ROW_ID_COLUMN,
) -> list[str]:
    """
    Columns where aligned missing-count decreased while the plan lists **retain
    missing** and does **not** list an **imputation**-classified action.

    Used to catch cleaners that still fill or normalize nulls away despite a retain
    line in the JSON plan. If the plan also lists imputation for the column, this
    returns false for that column (imputation is taken as authorizing a null drop).
    """
    bcols = set(df_before.columns)
    acols = set(df_after.columns)
    shared = sorted(bcols & acols)
    flagged: list[str] = []
    for col in shared:
        if row_id_col and col == row_id_col:
            continue
        nb, na, _mode = _null_counts_aligned(df_before, df_after, col, col, row_id_col)
        if na >= nb:
            continue
        if not plan_column_lists_retain_missing_action(plan, col):
            continue
        if plan_column_lists_imputation_action(plan, col):
            continue
        flagged.append(col)
    return sorted(flagged)


def format_retain_violation_host_fix_message(columns: list[str]) -> str:
    """
    Synthetic error text for :meth:`LightweightDataCleaningAgent.regenerate_plan_after_execute_error`.

    Drives the fix prompt when the host detects code that still fills columns
    whose JSON plan lists **retain missing values** without imputation.
    """
    cols = sorted({str(c) for c in columns})
    rtx = ", ".join(f"`{c}`" for c in cols)
    return (
        "Host verification (not a Python traceback): after running the cleaner on "
        "the upload, aligned missing-value counts decreased on "
        f"{rtx}, but the cleaning-plan JSON lists **retain missing values** for "
        "those column(s) without an imputation step—so the code still filled or "
        "removed nulls there. Revise the Python so step 9 and the rest of the "
        "pipeline **never** assign filled values to those columns (no fillna, "
        "mode/mean/median fill, bfill/ffill, or replace that reduces NA on them); "
        "leave them as NA. Update `columns[].actions` so it matches the revised "
        "code (keep the retain lines; do not add fake imputation for them)."
    )


def compose_host_pre_apply_blocked_message(
    ghost_columns: list[str],
    retain_violation_columns: list[str],
) -> str:
    """
    Single synthetic error for :meth:`LightweightDataCleaningAgent.regenerate_plan_after_execute_error`.

    Covers both “missingness dropped without imputation in the plan” and
    “retain missing in plan but code still reduces missingness” in one revision pass.
    """
    parts = [
        "Host dry-run verification failed (not a Python traceback). Revise the "
        "Python cleaner and JSON cleaning plan together so the next run satisfies "
        "all of the following that apply."
    ]
    if ghost_columns:
        g = ", ".join(f"`{c}`" for c in sorted(ghost_columns))
        parts.append(
            f"**Plan / missingness mismatch:** aligned missing-value counts decrease on "
            f"{g}, but `columns[].actions` has no **imputation**-classified step for those "
            "column(s). Add explicit "
            '`"impute missing values (mean)"`, `(median)`, or `(mode)` lines that match '
            "step 9, **or** change the code so missingness does not decrease there without "
            "that documentation."
        )
    if retain_violation_columns:
        r = ", ".join(f"`{c}`" for c in sorted(retain_violation_columns))
        parts.append(
            f"**Retain violated:** aligned missingness decreases on {r}, but the plan "
            "lists **retain missing values** (no imputation) for those column(s). "
            "Step 9 and the rest of the pipeline must **not** fill or coerce away nulls "
            "on those columns; leave NA as NA. Keep JSON aligned with the code."
        )
    return "\n\n".join(parts)
