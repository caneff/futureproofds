# Design: Read-only cleaning plan and instruction-driven regeneration

**Status:** Approved for implementation planning  
**Date:** 2026-05-15  
**Scope:** `data-cleaning-agent` Streamlit app and supporting `data_cleaning_agent` package

## 1. Problem

The cleaning UI has grown complex: interactive plan editing, dirty-state tracking, regeneration flows to keep code aligned with an edited plan, and pre-apply verification that compares dry-run execution to the plan (imputation, retain-missing, ghost missingness). This is hard to maintain and exceeds the product goal: users should steer cleaning via **instructions** and **full re-generation**, not by mutating structured plan JSON in the browser.

## 2. Goals

- **Remove** all ability to modify the cleaning plan in the UI (no widgets that change plan steps, no dirty state, no “regenerate code to match plan” / “reset plan” flows).
- **Show** a clear, **read-only** summary of intended work **per column**, derived from the latest plan JSON.
- **Remove** all checks that execution must match the plan for imputation, retain-missing, or related missingness semantics, and remove **LLM-driven auto-sync** before apply.
- **Keep** user **instructions** at the start; if the result is unsatisfactory, the user **adds or edits instructions** and runs **Generate again**, which **replaces** plan + code **in place** (no history of prior runs in the main UI).
- **Keep** a **non-LLM** optional **preview execute** of the cleaner on the in-memory dataframe before treating apply as successful, **only** to catch runtime/syntax/shape failures—not to compare outcomes to the plan.

## 3. Non-goals

- Deriving the column summary from static analysis of generated Python (plan JSON remains the summary source).
- Version history, diffing, or side-by-side comparison of past generations in the UI.
- Reintroducing plan–code parity enforcement in prompts or in the apply path.
- Automatic fix-agent / `regenerate_plan_after_execute_error` loops triggered from **Apply** (user recovers by editing instructions and generating again).

## 4. User flows

1. User uploads data (unchanged from current product expectations).
2. User enters **cleaning instructions** in a single persistent text area.
3. User clicks **Generate** (or equivalent). The agent returns **code + plan**; both are stored; the UI shows read-only per-column summary + collapsible code.
4. If the user is unhappy with the plan or code, they change **instructions** and click **Generate again**. The new result **fully replaces** the previous plan and code in session state and on screen.
5. User clicks **Apply**. The app may run the cleaner once on the preview dataframe to detect hard failures; on success, the app proceeds with the normal apply/persist behavior. **No** comparison of output missingness to plan steps.

## 5. UI behavior

| Element | Behavior |
|--------|----------|
| Instructions | Single `text_area` (or equivalent), same key across runs, editable any time before generate. |
| Generate / Generate again | Invokes full cleaning agent with current upload + instructions; replaces `pending_cleaning_plan` and `pending_cleaner_code` (and clears any caches tied to a prior generation). |
| Plan | Read-only: table or list per column (name + merged human-readable actions). Plan `notes` rendered as static text if present. |
| Code | Shown read-only in expander (existing pattern is fine). |
| Apply | Enabled whenever valid cleaner code exists; **not** gated on plan dirty flags. |

## 6. Session state

**Remove** keys and logic used only for: interactive plan edits, snapshot-vs-current diff, row-stats caches for editable widgets, regen supplemental state (exclusion text, verification feedback, cumulative removed actions, unclassified warnings), and pre-apply sync attempt counters.

**Retain** a minimal set: preview/upload dataframe, user instructions string, latest plan dict, latest code string, and apply/outcome state required for the rest of the app.

Exact key names are an implementation detail; the implementation plan should list current keys deleted vs kept after an audit of `app.py`.

## 7. Code and tests to remove or trim

| Artifact | Action |
|----------|--------|
| `data_cleaning_agent/plan_edit_verification.py` | Delete. |
| `tests/test_plan_edit_verification.py` | Delete. |
| `app.py` | Remove interactive plan renderer, `plan_dirty` / snapshot logic, regen-to-sync UI, imports and loops for pre-apply verification and host auto-sync. Simplify Apply per §5. |
| `data_cleaning_agent/utils.py` | Remove `removed_plan_actions`, `multiset_union_removed_plan_pairs`, `plan_step9_policy_host_supplement`, and any helpers exclusively used by removed flows. **Keep** `sanitize_cleaning_plan`, `coerce_cleaning_plan_columns`, `merged_plan_actions_by_column` for summary and sanitization. |
| `tests/test_plan_edit_helpers.py` | Remove tests for deleted multiset/removed-action helpers; **keep** tests for `coerce_cleaning_plan_columns` and `merged_plan_actions_by_column`. |
| `tests/test_utils.py` | Remove tests for `plan_step9_policy_host_supplement`. |
| `data_cleaning_agent/data_cleaning_agent.py` | Remove `LightweightDataCleaningAgent.regenerate_plan_after_execute_error` if no remaining callers. |
| Prompts (`data_cleaning_agent/prompts/data_cleaning.md`, `data_cleaning_fix.md`) | Remove or shorten passages whose sole purpose is enforcing JSON plan vs execution alignment for the old verification path; keep normal data-cleaning guidance. |

## 8. Read-only summary (implementation hint)

Add or reuse a small pure helper (module path chosen in the implementation plan; prefer `data_cleaning_agent` with no Streamlit imports) that:

1. Accepts a sanitized plan dict (or `columns` list).
2. Uses `coerce_cleaning_plan_columns` + `merged_plan_actions_by_column` to produce stable per-column rows.
3. Returns structures suitable for `st.dataframe` or markdown list (column name, actions as display string).

No Streamlit dependency inside the helper if practical, for easy unit testing.

## 9. Apply and preview execute

- **Allowed:** Run stored cleaner code on the in-memory preview dataframe before committing results, **only** to surface Python/runtime errors early (same conceptual role as a dry-run for “does this code run,” not “does it match the plan”).
- **Forbidden in this design:** Any check that compares resulting missingness or imputation behavior to plan text or JSON; any loop that calls the fix LLM to rewrite plan/code before apply based on such checks.

If preview execute fails, show an error; user fixes via instructions + **Generate again**.

## 10. Testing

- Delete tests tied to deleted modules and deleted helpers.
- Add focused unit tests for the read-only summary helper (plan in → display rows out), including empty plan, unknown column shapes, and merged duplicate column rows.
- Run `uv run pytest -q` for `data-cleaning-agent` after changes; all tests must pass.

## 11. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Users apply code that diverges semantically from the plan narrative | Accepted tradeoff; instructions + regenerate are the control surface. |
| Large `app.py` refactor introduces regressions | Implementation plan should stage internal extraction (e.g. summary renderer) and keep apply path tests where they exist; add summary tests. |
| Stale imports after deletions | Grep for removed symbols before merge. |

## 12. Relation to prior specs

This design **supersedes** the interactive plan-edit and pre-apply verification behavior described in `docs/superpowers/specs/2026-05-14-plan-edit-verification-design.md` for the shipped product. That file may remain in the repo for history if present; new work follows **this** document.
