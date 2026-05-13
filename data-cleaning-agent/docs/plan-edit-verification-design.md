# Plan-edit verification after regenerate (scope B, lenient unknowns)

> **Repo note:** `docs/superpowers/` is gitignored in this repository; this file is the
> **versioned** copy of the design. A duplicate may exist under `docs/superpowers/specs/`
> for local superpowers workflows.

## Summary

When the user unchecks cleaning-plan steps and clicks **Regenerate code to match plan**, the app must ensure the **new generated cleaner** does not still perform effects the user removed—at minimum for **imputation / missingness**, **column drops**, and **dtype / coercion** on the affected columns (scope **B**). Verification is **execution-based** (run the proposed cleaner on the current upload) plus a **small classifier** that maps removed `(column, action)` pairs to check types.

Removed steps whose action text **cannot** be classified are handled **leniently**: show a **warning**, skip dedicated predicates for that pair, and **still allow** the regenerated code to be accepted if all classified checks pass.

## Goals

1. After successful LLM `generate_cleaning_code` on plan-edit regen, **before** persisting new `pending_cleaner_code` (and related snapshot), run verification.
2. If verification **fails** on any **classified** removed step, **do not** accept the new code; show a clear error; keep prior code/plan state (see Rollback).
3. If verification **passes** but there were **unclassified** removed steps, show a **non-blocking warning** listing those pairs so users know enforcement was partial.
4. **Apply cleaning** continues to use accepted code only; optional future: lightweight re-check before apply (out of scope unless trivial).

## Non-goals (v1)

- Full semantic fidelity for every arbitrary English action string (only B + lenient unknowns).
- Row-level operations (duplicate removal, all-null row drops) as verification targets unless later extended.
- Replacing LLM codegen with a host compiler (future option).

## Definitions

- **Removed steps:** `removed_plan_actions(plan_snapshot_for_code.columns, pending_cleaning_plan.columns)` multiset pairs `(column_name, action_string)` the user removed relative to the snapshot used for regen (same source as today’s `plan_excl`).
- **Classified removed step:** `(col, action)` maps to exactly one of `impute`, `drop`, `dtype`, via keyword/pattern rules (implementation detail in code; table in tests).
- **Unclassified removed step:** no match; lenient path.

## Architecture

| Piece | Responsibility |
|--------|----------------|
| **Classifier** | `(action: str) -> Literal["impute","drop","dtype", None]` |
| **Predicates** | Given `df_before`, `df_after`, `column` (and optional row-id column), return pass/fail + human-readable reason |
| **Orchestrator** | For each removed pair: classify → run predicate or warn; aggregate result |
| **Host hook** | Streamlit regen handler: after `generate_cleaning_code`, run `run_cleaner_code_on_dataframe` on `preview_df_input`; call orchestrator; branch accept/rollback |

Reuse existing utilities where possible (`run_cleaner_code_on_dataframe`, `APP_SYNTHETIC_ALIGN_ROW_ID_COLUMN` / `AGENT_ROW_ID`, column name alignment after step-2 normalization—align with how plans name columns vs `df`).

## Predicates (scope B)

All comparisons use the **same** `df_before` / `df_after` as the verification run (upload with synthetic row id, consistent with apply).

### Imputation removed

**Intent:** User removed an imputation step; the cleaner must not **reduce** missingness on that column relative to its input for this run.

**Rule (v1):** Compare `null_before` and `null_after` (missing counts via `isna` on the resolved column; extend later for placeholder strings if needed). Align rows by synthetic row id when present in both frames; if alignment is ambiguous, document conservative behavior in the implementation plan.

**Pass:** `null_after >= null_before`.

**Edge:** Column renamed in cleaner step 2—map plan column string to `df_before` / `df_after` names consistently; if the column disappears from `df_after`, treat under **drop** rules as well.

### Drop column removed

**Intent:** User unchecked dropping a column; output must retain that column (under same name resolution as above).

**Pass:** Column present in `df_after`.

### Dtype / coercion removed

**Intent:** User unchecked a dtype step; dtype for that column should be unchanged across this cleaner run unless another retained step legitimately changes it (known false-positive risk).

**Pass:** `dtype(df_before[col])` equals `dtype(df_after[col])` (pandas dtype equality).

**Edge:** If another retained step changes dtype, verification may false-fail; error copy should show both dtypes.

## Lenient unknowns

- If classifier returns `None`: append to `unclassified_removed: list[tuple[str,str]]`.
- After all classified checks: if any classified check **failed** → **reject** (see Errors).
- If classified checks **pass** and `unclassified_removed` non-empty → `st.warning` with bullet list of pairs: “Could not automatically verify these removed steps; confirm in preview: …”
- If only unclassified removals exist (no classified predicates run): **accept** with the same warning (lenient).

## Rollback / session state

On **verification failure**:

- Do **not** update `pending_cleaner_code` from the failed attempt.
- Do **not** advance `plan_snapshot_for_code` / `plan_widget_nonce` as if regen succeeded.
- Leave `pending_cleaning_plan` as the user’s edited `cur_go` (already true today).
- Show error; optionally keep `plan_regen_exclusion_instructions` visible for retry.

On **success** (possibly with warnings):

- Persist code + snapshot as today; clear or keep regen exclusion panel per existing UX.

## UX copy (draft)

- **Failure (classified):** “Regenerated code still performs a removed step: **{column}** — **{reason}**. Try **Regenerate** again, or adjust the plan.”
- **Warning (unclassified):** “These removed steps are not covered by automatic checks yet; please confirm the preview: **{list}**.”

## Testing

- **Classifier:** table-driven tests for `impute` / `drop` / `dtype` / `None`.
- **Predicates:** small `DataFrame` fixtures for pass/fail per type.
- **Orchestrator:** mixed removed list → expect fail / pass + warnings.
- **App:** optional thin test of hook with mocked agent output (heavier; defer if costly).

## Open items for implementation plan

1. Exact column name resolution between plan strings and `df_before` columns (normalization helper reuse).
2. Null-count baseline for object columns with empty-string placeholders (align with pipeline “missing” definition or keep `isna` only for v1).
3. Maximum auto-retries (default **0**; user clicks regen again).

## Self-review

- No contradictory strict/lenient: classified strict, unknown lenient—explicit.
- Scope B only; row ops out of scope.
- Rollback behavior specified to avoid half-applied regen.
