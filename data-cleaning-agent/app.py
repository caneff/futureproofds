"""Streamlit interface for the Data Cleaning Agent."""

import copy
import hashlib

import pandas as pd
import streamlit as st
from data_cleaning_agent import LightweightDataCleaningAgent
from data_cleaning_agent.cleaning_outcome_summary import (
    build_cleaning_outcome_facts,
    format_outcome_summary_markdown,
    outcome_facts_show_any_change,
)
from data_cleaning_agent.plan_edit_verification import (
    VerificationResult,
    columns_where_missingness_dropped_without_plan_imputation,
    columns_where_retain_missing_plan_violated_by_execution,
    compose_host_pre_apply_blocked_message,
    compose_plan_regen_supplemental,
    format_unclassified_warning_markdown,
    format_verification_feedback_markdown,
    verify_removed_plan_steps,
)
from data_cleaning_agent.utils import (
    merged_plan_actions_by_column,
    multiset_union_removed_plan_pairs,
    plan_step9_policy_host_supplement,
    removed_plan_actions,
    run_cleaner_code_on_dataframe,
    sanitize_cleaning_plan,
)
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from preview_helpers import (
    AGENT_ROW_ID,
    preview_aligned_frames,
    reorder_cleaned_for_export,
    style_preview_pair,
)

load_dotenv()

st.set_page_config(
    page_title="Data Cleaning Agent",
    layout="wide",
)

st.title("🧹 Data Cleaning Agent")

MAX_EXECUTE_FIXES = 3
MAX_HOST_PRE_APPLY_SYNC_FIXES = 5


def _plan_step_checkbox_key(
    *,
    code_digest: str,
    widget_nonce: int,
    column_name: str,
    action: str,
    occurrence: int,
) -> str:
    """Stable widget key for one plan step (survives list shrink when others are removed)."""
    stable = hashlib.md5(
        f"{column_name}\x00{action}\x00{occurrence}".encode("utf-8")
    ).hexdigest()[:20]
    return f"plan_step_{code_digest}_{widget_nonce}_{stable}"


def _coerce_cumulative_removed(val: object) -> list[tuple[str, str]]:
    """Normalize ``plan_regen_cumulative_removed_actions`` session value."""
    if not isinstance(val, list):
        return []
    out: list[tuple[str, str]] = []
    for item in val:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((str(item[0]), str(item[1])))
    return out


def _build_plan_exclusion_supplement(
    snap: dict,
    cur: dict,
    *,
    removed_pairs: list[tuple[str, str]] | None = None,
) -> str:
    """Host-supplement text appended to ``supplemental_instructions`` on plan-edit regen."""
    removed = (
        removed_pairs
        if removed_pairs is not None
        else removed_plan_actions(snap.get("columns"), cur.get("columns"))
    )
    if not removed:
        return (
            "The user edited the cleaning plan but no steps were removed relative "
            "to the last generated code. Keep the standard pipeline."
        )
    lines = [
        "The user **removed** these planned cleaning steps (column + action text). "
        "Do **not** perform them in your Python function: skip or omit the "
        "corresponding logic only. Keep the rest of the pipeline coherent and "
        "deterministic. Return a JSON cleaning plan that matches what the revised "
        "code actually does (do not list removed steps).",
        "",
    ]
    for col, act in removed:
        lines.append(f'- Column "{col}": {act!r}')
    lines.append("")
    lines.append(
        "**Imputation / missing-value fills:** For any removed step whose action text "
        "suggests imputation or filling NaNs on a column (e.g. **impute**, **median**, "
        "**mean**, **mode**, **fillna**, **bfill**, **ffill**, or **missing values**), "
        "the revised Python must **not** assign filled values to that column—**skip "
        "that column** in imputation loops (step 9 and similar). Do **not** rely on "
        "`notes` alone to record an exclusion; the code must implement it."
    )
    lines.append("")
    lines.append(
        "If omitting a step makes a later step unsafe or meaningless, adapt the code "
        "minimally while still honoring the removals above; keep `notes` factual and "
        "consistent with what the function actually does."
    )
    return "\n".join(lines)


def _store_plan_snapshot_after_code_from_llm(plan: dict | None) -> None:
    """Call whenever pending_cleaner_code is replaced from the model (not user edits)."""
    if plan is not None:
        st.session_state["plan_snapshot_for_code"] = copy.deepcopy(plan)
    else:
        st.session_state.pop("plan_snapshot_for_code", None)
    st.session_state["plan_dirty"] = False
    st.session_state["plan_widget_nonce"] = (
        int(st.session_state.get("plan_widget_nonce") or 0) + 1
    )


def _invalidate_plan_row_stats_cache() -> None:
    st.session_state.pop("plan_row_stats", None)
    st.session_state.pop("_plan_row_stats_cache_key", None)


def _render_interactive_cleaning_plan_columns(
    plan: dict,
    *,
    code_digest: str,
    widget_nonce: int,
) -> None:
    """
    Per-column collapsible lists: ``st.expander`` per column with checkboxes inside.

    Checkbox rows come from ``plan_snapshot_for_code`` when available so unchecking
    only updates the pending plan and steps can be re-checked. Otherwise rows follow
    the current plan only.
    """
    cols_in = plan.get("columns")
    if not isinstance(cols_in, list) or not cols_in:
        return
    st.caption(
        "Expand a column to edit steps. Uncheck to drop. **Apply** stays off until "
        "**Regenerate Code to Match Plan**."
    )

    def _short_label(text: str, max_len: int = 72) -> str:
        t = str(text).replace("\n", " ").strip()
        return t if len(t) <= max_len else f"{t[: max_len - 1]}…"

    pend_by_name = merged_plan_actions_by_column(cols_in)

    snap = st.session_state.get("plan_snapshot_for_code")
    snap_by_name: dict[str, list[str]] = {}
    if isinstance(snap, dict):
        snap_by_name = merged_plan_actions_by_column(snap.get("columns"))

    names_ordered: list[str] = []
    seen: set[str] = set()
    for nm in snap_by_name.keys():
        if nm not in seen:
            names_ordered.append(nm)
            seen.add(nm)
    for nm in pend_by_name.keys():
        if nm not in seen:
            names_ordered.append(nm)
            seen.add(nm)

    new_columns: list[dict] = []
    for name in names_ordered:
        row = next(
            (
                r
                for r in cols_in
                if isinstance(r, dict) and str(r.get("name", "")).strip() == name
            ),
            None,
        )
        if row is None:
            row = {"name": name, "actions": snap_by_name.get(name, [])}

        if name in snap_by_name:
            display_actions = list(snap_by_name[name])
        else:
            display_actions = list(pend_by_name.get(name, []))

        if not display_actions:
            st.caption(f"`{name or '(unnamed)'}` — no steps")
            continue

        pending_actions = pend_by_name.get(name, [])
        # Stable key per column (not list index): if ``names_ordered`` shrinks or
        # reorders after edits, index-based keys would remount expanders and collapse them.
        name_slug = hashlib.md5(
            f"{code_digest}\x00{widget_nonce}\x00{name}".encode("utf-8")
        ).hexdigest()[:24]
        exp_key = f"plan_col_exp_{name_slug}"
        title = f"`{name or '(unnamed)'}` — {len(display_actions)} step(s)"
        with st.expander(title, expanded=False, key=exp_key):
            kept: list[str] = []
            dup_occ: dict[str, int] = {}
            for act in display_actions:
                occ = dup_occ.get(act, 0)
                dup_occ[act] = occ + 1
                ck = _plan_step_checkbox_key(
                    code_digest=code_digest,
                    widget_nonce=widget_nonce,
                    column_name=name,
                    action=act,
                    occurrence=occ,
                )
                pending_matches = sum(1 for a in pending_actions if a == act)
                initially_on = pending_matches > occ
                if st.checkbox(
                    _short_label(act),
                    value=initially_on,
                    key=ck,
                    help=str(act),
                ):
                    kept.append(act)
            if kept:
                new_columns.append({"name": row.get("name"), "actions": kept})
    plan["columns"] = new_columns


# Upload file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    upload_fp = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("_cleaning_upload_fp") != upload_fp:
        st.session_state.pop("preview_df_input", None)
        st.session_state.pop("preview_df_cleaned", None)
        st.session_state.pop("pending_cleaner_code", None)
        st.session_state.pop("pending_cleaning_plan", None)
        st.session_state.pop("pending_function_name", None)
        st.session_state.pop("execute_fix_count", None)
        st.session_state.pop("cleaning_apply_exhausted", None)
        st.session_state.pop("plan_row_stats", None)
        st.session_state.pop("_plan_row_stats_cache_key", None)
        st.session_state.pop("plan_snapshot_for_code", None)
        st.session_state.pop("plan_dirty", None)
        st.session_state.pop("plan_widget_nonce", None)
        st.session_state.pop("plan_regen_exclusion_instructions", None)
        st.session_state.pop("plan_regen_verification_feedback", None)
        st.session_state.pop("plan_regen_user_context", None)
        st.session_state.pop("plan_regen_cumulative_removed_actions", None)
        st.session_state["_cleaning_upload_fp"] = upload_fp

    df_uploaded = pd.read_csv(uploaded_file)

    supplemental_instructions = (
        f'The column "{AGENT_ROW_ID}" is a synthetic stable row identifier '
        "added by the application before cleaning. Do not drop it, rename it, "
        "or change its values. Carry it through unchanged for every row that "
        "remains in the returned DataFrame so before-and-after rows can be aligned."
    )

    def _cleaning_supplement_for_df(df: pd.DataFrame) -> str:
        return supplemental_instructions + plan_step9_policy_host_supplement(
            df, row_id_col=AGENT_ROW_ID
        )

    if st.button("Generate cleaning plan"):
        with st.spinner("Generating plan and code..."):
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            agent = LightweightDataCleaningAgent(model=llm, log=True)
            df_input = df_uploaded.copy()
            df_input.insert(
                0, AGENT_ROW_ID, pd.RangeIndex(stop=len(df_input), dtype="int64")
            )
            agent.generate_cleaning_code(
                source_df=df_input,
                supplemental_instructions=_cleaning_supplement_for_df(df_input),
            )
            st.session_state["pending_cleaner_code"] = agent.get_data_cleaner_function()
            st.session_state["pending_cleaning_plan"] = agent.get_cleaning_plan()
            st.session_state["pending_function_name"] = (
                agent.response.get("data_cleaner_function_name")
                if agent.response
                else "data_cleaner"
            )
            st.session_state["preview_df_input"] = df_input
            st.session_state.pop("preview_df_cleaned", None)
            st.session_state["execute_fix_count"] = 0
            st.session_state.pop("cleaning_apply_exhausted", None)
            st.session_state.pop("plan_snapshot_for_code", None)
            st.session_state.pop("plan_dirty", None)
            st.session_state.pop("plan_regen_exclusion_instructions", None)
            st.session_state.pop("plan_regen_verification_feedback", None)
            st.session_state.pop("plan_regen_user_context", None)
            st.session_state.pop("plan_regen_cumulative_removed_actions", None)
            st.session_state["plan_widget_nonce"] = (
                int(st.session_state.get("plan_widget_nonce") or 0) + 1
            )
            _invalidate_plan_row_stats_cache()
        st.success(
            "Plan generated. Review the plan and code below, then apply when ready."
        )

    pending_code = st.session_state.get("pending_cleaner_code")
    pending_plan = st.session_state.get("pending_cleaning_plan")
    df_input_stored = st.session_state.get("preview_df_input")

    if pending_plan is not None and df_input_stored is not None:
        resanitized = sanitize_cleaning_plan(pending_plan, df_input_stored)
        if resanitized is not None:
            st.session_state["pending_cleaning_plan"] = resanitized
            pending_plan = resanitized

    if (
        pending_code
        and pending_plan is not None
        and isinstance(pending_plan, dict)
        and st.session_state.get("plan_snapshot_for_code") is None
    ):
        st.session_state["plan_snapshot_for_code"] = copy.deepcopy(pending_plan)
        st.session_state["plan_dirty"] = False

    if pending_code and df_input_stored is not None:
        _uw = st.session_state.pop("_plan_regen_unclassified_warn", None)
        if isinstance(_uw, str) and _uw.strip():
            st.warning(_uw)
        if pending_plan is None:
            st.subheader("Cleaning plan")
            st.warning(
                "Structured plan JSON was missing or invalid. You can still run "
                "cleaning with Apply cleaning below—review the generated code first."
            )
            st.session_state["plan_dirty"] = False
        else:
            cols = pending_plan.get("columns")
            notes = pending_plan.get("notes") or ""
            with st.expander("Cleaning plan", expanded=False, key="cleaning_plan_main"):
                code_digest = hashlib.sha256(pending_code.encode("utf-8")).hexdigest()[
                    :24
                ]
                nonce = int(st.session_state.get("plan_widget_nonce") or 0)
                if cols:
                    if isinstance(cols, list):
                        _render_interactive_cleaning_plan_columns(
                            pending_plan,
                            code_digest=code_digest,
                            widget_nonce=nonce,
                        )
                    else:
                        st.dataframe(
                            pd.DataFrame(cols), width="stretch", hide_index=True
                        )
                        st.caption(
                            "Step editing needs a list-shaped ``columns`` plan; this plan "
                            "uses another shape."
                        )
                else:
                    st.caption("No per-column entries (columns list empty).")

                if notes:
                    st.markdown("**Notes**")
                    st.write(notes)

            snap = st.session_state.get("plan_snapshot_for_code")
            cur_plan = st.session_state.get("pending_cleaning_plan")
            if isinstance(snap, dict) and isinstance(cur_plan, dict):
                st.session_state["plan_dirty"] = bool(
                    removed_plan_actions(snap.get("columns"), cur_plan.get("columns"))
                )
            else:
                st.session_state["plan_dirty"] = False

            if st.session_state.get("plan_dirty"):
                st.warning(
                    "The cleaning plan no longer matches the generated code. "
                    "Regenerate code before applying, or reset the plan."
                )
                st.text_area(
                    "Notes for next regenerate (optional)",
                    height=96,
                    key="plan_regen_user_context",
                    help=(
                        "Free-text constraints or how to fix the last failure; "
                        "appended to supplemental instructions on the next "
                        "**Regenerate Code to Match Plan**."
                    ),
                )
                regen_col, reset_col = st.columns(2)
                with regen_col:
                    if st.button("Regenerate Code to Match Plan", type="primary"):
                        snap_go = st.session_state.get("plan_snapshot_for_code")
                        cur_go = st.session_state.get("pending_cleaning_plan")
                        if not isinstance(snap_go, dict) or not isinstance(
                            cur_go, dict
                        ):
                            st.error("Missing plan state for regeneration.")
                        else:
                            assert isinstance(df_input_stored, pd.DataFrame)
                            df_regen = df_input_stored
                            delta_removed = removed_plan_actions(
                                snap_go.get("columns"), cur_go.get("columns")
                            )
                            prior_removed = _coerce_cumulative_removed(
                                st.session_state.get(
                                    "plan_regen_cumulative_removed_actions"
                                )
                            )
                            removed = multiset_union_removed_plan_pairs(
                                prior_removed, delta_removed
                            )
                            plan_excl = _build_plan_exclusion_supplement(
                                snap_go, cur_go, removed_pairs=removed
                            )
                            prior_fb = st.session_state.get(
                                "plan_regen_verification_feedback"
                            )
                            if not isinstance(prior_fb, str):
                                prior_fb = None
                            user_notes = st.session_state.get("plan_regen_user_context")
                            if not isinstance(user_notes, str):
                                user_notes = None

                            def _generate_and_verify(
                                extra_retry_failure: str | None,
                                df_work: pd.DataFrame,
                            ) -> tuple[
                                str | None,
                                str | None,
                                VerificationResult | None,
                                str | None,
                            ]:
                                """Return ``(code, fn_name, verify_result, execute_err)``."""
                                sup = compose_plan_regen_supplemental(
                                    _cleaning_supplement_for_df(df_work),
                                    plan_excl,
                                    prior_verification_feedback=prior_fb,
                                    automatic_retry_failure_block=extra_retry_failure,
                                    user_follow_up=user_notes,
                                )
                                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                                agent = LightweightDataCleaningAgent(
                                    model=llm, log=True
                                )
                                agent.generate_cleaning_code(
                                    source_df=df_work,
                                    user_instructions=None,
                                    supplemental_instructions=sup,
                                )
                                code = agent.get_data_cleaner_function()
                                if not isinstance(code, str) or not code.strip():
                                    return None, None, None, "no cleaner code returned"
                                fn_name = (
                                    agent.response.get("data_cleaner_function_name")
                                    if agent.response
                                    else "data_cleaner"
                                )
                                if not isinstance(fn_name, str) or not fn_name.strip():
                                    fn_name = "data_cleaner"
                                if not removed:
                                    return (
                                        code,
                                        fn_name,
                                        VerificationResult(ok=True),
                                        None,
                                    )
                                df_out, err = run_cleaner_code_on_dataframe(
                                    code,
                                    df_work,
                                    function_name=fn_name,
                                )
                                if err:
                                    return None, None, None, err
                                if df_out is None:
                                    return None, None, None, "cleaner returned no frame"
                                vr = verify_removed_plan_steps(
                                    removed,
                                    df_work,
                                    df_out,
                                    row_id_col=AGENT_ROW_ID,
                                )
                                return code, fn_name, vr, None

                            with st.spinner(
                                "Regenerating code for your edited plan..."
                            ):
                                code1, fn1, vr1, err1 = _generate_and_verify(
                                    None, df_regen
                                )
                                accepted_code: str | None = None
                                accepted_fn: str | None = None
                                final_vr: VerificationResult | None = None

                                if err1:
                                    st.error(
                                        "Regenerated code failed to execute on the upload: "
                                        f"{err1}"
                                    )
                                elif vr1 is not None and vr1.ok:
                                    accepted_code, accepted_fn = code1, fn1
                                    final_vr = vr1
                                elif vr1 is not None and not vr1.ok:
                                    fb_block = format_verification_feedback_markdown(
                                        vr1
                                    )
                                    code2, fn2, vr2, err2 = _generate_and_verify(
                                        fb_block, df_regen
                                    )
                                    if err2:
                                        st.session_state[
                                            "plan_regen_verification_feedback"
                                        ] = (
                                            fb_block
                                            + "\n\n**Second attempt — code did not run:**\n"
                                            + str(err2)
                                        )
                                        st.error(
                                            "The automatic retry produced code that "
                                            f"failed to execute: {err2}"
                                        )
                                    elif vr2 is not None and vr2.ok:
                                        accepted_code, accepted_fn = code2, fn2
                                        final_vr = vr2
                                    elif vr2 is not None:
                                        st.session_state[
                                            "plan_regen_verification_feedback"
                                        ] = format_verification_feedback_markdown(vr2)
                                        fail0 = vr2.classified_failures[0]
                                        st.error(
                                            "Regenerated code still fails checks after an "
                                            "automatic retry — "
                                            f"**{fail0.plan_column}**: {fail0.reason}. "
                                            "Review the **automatic findings** below, add "
                                            "**optional notes** if needed, then **Regenerate "
                                            "Code to Match Plan** again."
                                        )

                                if accepted_code is not None and final_vr is not None:
                                    plan_to_store = copy.deepcopy(cur_go)
                                    sanitized_plan = sanitize_cleaning_plan(
                                        plan_to_store, df_input_stored
                                    )
                                    if sanitized_plan is not None:
                                        plan_to_store = sanitized_plan
                                    st.session_state["pending_cleaning_plan"] = (
                                        plan_to_store
                                    )
                                    st.session_state["pending_cleaner_code"] = (
                                        accepted_code
                                    )
                                    st.session_state["pending_function_name"] = (
                                        accepted_fn or "data_cleaner"
                                    )
                                    _store_plan_snapshot_after_code_from_llm(
                                        st.session_state.get("pending_cleaning_plan")
                                    )
                                    _invalidate_plan_row_stats_cache()
                                    st.session_state["execute_fix_count"] = 0
                                    st.session_state.pop(
                                        "cleaning_apply_exhausted", None
                                    )
                                    st.session_state[
                                        "plan_regen_exclusion_instructions"
                                    ] = plan_excl
                                    st.session_state.pop(
                                        "plan_regen_verification_feedback", None
                                    )
                                    st.session_state.pop(
                                        "plan_regen_user_context", None
                                    )
                                    st.session_state[
                                        "plan_regen_cumulative_removed_actions"
                                    ] = removed
                                    if final_vr.unclassified_removed:
                                        st.session_state[
                                            "_plan_regen_unclassified_warn"
                                        ] = format_unclassified_warning_markdown(
                                            final_vr.unclassified_removed
                                        )
                                    st.success(
                                        "Code and plan updated from your edits. "
                                        "Review, then apply."
                                    )
                                    st.rerun()
                with reset_col:
                    if st.button("Reset plan to last generated"):
                        snap_r = st.session_state.get("plan_snapshot_for_code")
                        if isinstance(snap_r, dict):
                            st.session_state["pending_cleaning_plan"] = copy.deepcopy(
                                snap_r
                            )
                            st.session_state["plan_dirty"] = False
                            st.session_state["plan_widget_nonce"] = (
                                int(st.session_state.get("plan_widget_nonce") or 0) + 1
                            )
                            st.session_state.pop(
                                "plan_regen_exclusion_instructions", None
                            )
                            st.session_state.pop(
                                "plan_regen_verification_feedback", None
                            )
                            st.session_state.pop("plan_regen_user_context", None)
                            st.session_state.pop(
                                "plan_regen_cumulative_removed_actions", None
                            )
                            st.rerun()

                fb_show = st.session_state.get("plan_regen_verification_feedback")
                if isinstance(fb_show, str) and fb_show.strip():
                    with st.expander(
                        "Automatic verification findings (sent on next regenerate)",
                        expanded=True,
                    ):
                        st.markdown(fb_show)

        regen_excl = st.session_state.get("plan_regen_exclusion_instructions")
        if isinstance(regen_excl, str) and regen_excl.strip():
            col_code, col_excl = st.columns(2, gap="medium")
            with col_code:
                with st.expander("Generated cleaning code", expanded=False):
                    st.code(pending_code, language="python")
            with col_excl:
                with st.expander("Build plan exclusion instructions", expanded=False):
                    st.markdown(regen_excl)
        else:
            with st.expander("Generated cleaning code"):
                st.code(pending_code, language="python")

        plan_dirty = bool(st.session_state.get("plan_dirty"))
        if plan_dirty:
            st.caption("Apply cleaning is disabled until code matches the plan.")
        if st.button("Apply Cleaning", disabled=plan_dirty):
            run_apply = True
            snap_apply = st.session_state.get("plan_snapshot_for_code")
            cur_apply = st.session_state.get("pending_cleaning_plan")
            if isinstance(snap_apply, dict) and isinstance(cur_apply, dict):
                if removed_plan_actions(
                    snap_apply.get("columns"), cur_apply.get("columns")
                ):
                    st.error(
                        "Cannot apply: the edited plan still does not match the "
                        "generated code. Use **Regenerate Code to Match Plan** first, "
                        "then apply again."
                    )
                    run_apply = False
            if run_apply:
                apply_regen_rerun = False
                apply_exhausted = False
                apply_exhausted_err: str | None = None
                apply_success = False
                apply_host_sync_notice: str | None = None
                ghost_apply_cols: list[str] = []
                retain_violation_cols: list[str] = []
                pre_apply_blocked = False
                host_pre_apply_sync_attempts = 0
                _host_pre_apply_sync_done = 0
                with st.spinner("Applying cleaning..."):
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                    agent = LightweightDataCleaningAgent(model=llm, log=True)
                    fn_work = (
                        st.session_state.get("pending_function_name") or "data_cleaner"
                    )
                    plan_work = st.session_state.get("pending_cleaning_plan")
                    code_work = pending_code
                    df_pre_apply: pd.DataFrame | None = None
                    pre_apply_err: str | None = None

                    while True:
                        agent.response = {
                            "data_cleaner_function": code_work,
                            "cleaning_plan": plan_work,
                            "data_cleaner_function_name": fn_work,
                            "retry_count": int(
                                st.session_state.get("execute_fix_count") or 0
                            ),
                        }
                        df_pre_apply, pre_apply_err = run_cleaner_code_on_dataframe(
                            code_work,
                            df_input_stored,
                            function_name=fn_work,
                        )
                        ghost_apply_cols = []
                        retain_violation_cols = []
                        if not pre_apply_err and df_pre_apply is not None:
                            ghost_apply_cols = columns_where_missingness_dropped_without_plan_imputation(
                                df_input_stored,
                                df_pre_apply,
                                plan_work,
                                row_id_col=AGENT_ROW_ID,
                            )
                            retain_violation_cols = (
                                columns_where_retain_missing_plan_violated_by_execution(
                                    df_input_stored,
                                    df_pre_apply,
                                    plan_work,
                                    row_id_col=AGENT_ROW_ID,
                                )
                            )
                        pre_apply_blocked = bool(ghost_apply_cols) or bool(
                            retain_violation_cols
                        )
                        if not pre_apply_blocked:
                            break
                        if pre_apply_err:
                            break
                        if (
                            host_pre_apply_sync_attempts
                            >= MAX_HOST_PRE_APPLY_SYNC_FIXES
                        ):
                            break
                        agent.regenerate_plan_after_execute_error(
                            compose_host_pre_apply_blocked_message(
                                ghost_apply_cols,
                                retain_violation_cols,
                            )
                        )
                        c2 = agent.get_data_cleaner_function()
                        if isinstance(c2, str) and c2.strip():
                            code_work = c2
                        p2 = agent.get_cleaning_plan()
                        if p2 is not None:
                            sp2 = sanitize_cleaning_plan(p2, df_input_stored)
                            plan_work = sp2 if sp2 is not None else p2
                        fn2 = (
                            agent.response.get("data_cleaner_function_name")
                            if agent.response
                            else None
                        )
                        if isinstance(fn2, str) and fn2.strip():
                            fn_work = fn2
                        host_pre_apply_sync_attempts += 1

                    err = None
                    if not pre_apply_blocked:
                        if pre_apply_err:
                            err = pre_apply_err
                        elif df_pre_apply is None:
                            err = "cleaner returned no result"
                        else:
                            agent.response = {
                                **(agent.response or {}),
                                "data_cleaner_function": code_work,
                                "cleaning_plan": plan_work,
                                "data_cleaner_function_name": fn_work,
                                "retry_count": int(
                                    st.session_state.get("execute_fix_count") or 0
                                ),
                                "data_cleaned": df_pre_apply.to_dict(),
                                "data_cleaner_error": None,
                                "source_df": df_input_stored.to_dict(),
                            }
                            if host_pre_apply_sync_attempts > 0:
                                apply_host_sync_notice = (
                                    f"Adjusted cleaning code and plan automatically "
                                    f"({host_pre_apply_sync_attempts}x) using host dry-run checks."
                                )
                                st.session_state["pending_cleaner_code"] = code_work
                                st.session_state["pending_cleaning_plan"] = plan_work
                                st.session_state["pending_function_name"] = fn_work
                                _invalidate_plan_row_stats_cache()

                    _host_pre_apply_sync_done = host_pre_apply_sync_attempts

                    if not pre_apply_blocked:
                        if err:
                            fc = int(st.session_state.get("execute_fix_count") or 0)
                            if fc < MAX_EXECUTE_FIXES:
                                agent.regenerate_plan_after_execute_error(err)
                                st.session_state["pending_cleaner_code"] = (
                                    agent.get_data_cleaner_function()
                                )
                                st.session_state["pending_cleaning_plan"] = (
                                    agent.get_cleaning_plan()
                                )
                                plan_fix = st.session_state["pending_cleaning_plan"]
                                if plan_fix is not None:
                                    sp_fix = sanitize_cleaning_plan(
                                        plan_fix, df_input_stored
                                    )
                                    if sp_fix is not None:
                                        st.session_state["pending_cleaning_plan"] = (
                                            sp_fix
                                        )
                                        plan_fix = sp_fix
                                _store_plan_snapshot_after_code_from_llm(
                                    st.session_state.get("pending_cleaning_plan")
                                )
                                _invalidate_plan_row_stats_cache()
                                st.session_state["execute_fix_count"] = fc + 1
                                st.session_state["cleaning_apply_exhausted"] = False
                                st.session_state.pop(
                                    "plan_regen_exclusion_instructions", None
                                )
                                st.session_state.pop(
                                    "plan_regen_verification_feedback", None
                                )
                                st.session_state.pop("plan_regen_user_context", None)
                                st.session_state.pop(
                                    "plan_regen_cumulative_removed_actions", None
                                )
                                apply_regen_rerun = True
                            else:
                                st.session_state["cleaning_apply_exhausted"] = True
                                apply_exhausted = True
                                apply_exhausted_err = str(err)
                        else:
                            st.session_state["cleaning_apply_exhausted"] = False
                            st.session_state["preview_df_cleaned"] = (
                                agent.get_data_cleaned()
                            )
                            st.session_state["pending_cleaner_code"] = (
                                agent.get_data_cleaner_function()
                            )
                            st.session_state["pending_cleaning_plan"] = (
                                agent.get_cleaning_plan()
                            )
                            plan_ok = st.session_state["pending_cleaning_plan"]
                            if plan_ok is not None:
                                sp_ok = sanitize_cleaning_plan(plan_ok, df_input_stored)
                                if sp_ok is not None:
                                    st.session_state["pending_cleaning_plan"] = sp_ok
                                    plan_ok = sp_ok
                            _store_plan_snapshot_after_code_from_llm(
                                st.session_state.get("pending_cleaning_plan")
                            )
                            _invalidate_plan_row_stats_cache()
                            st.session_state["execute_fix_count"] = 0
                            st.session_state.pop(
                                "plan_regen_exclusion_instructions", None
                            )
                            st.session_state.pop(
                                "plan_regen_verification_feedback", None
                            )
                            st.session_state.pop("plan_regen_user_context", None)
                            st.session_state.pop(
                                "plan_regen_cumulative_removed_actions", None
                            )
                            apply_success = True

                if pre_apply_blocked:
                    parts: list[str] = []
                    if _host_pre_apply_sync_done:
                        parts.append(
                            "**Cannot apply cleaning** — host dry-run checks still fail after "
                            f"**{_host_pre_apply_sync_done}** automatic model revision(s)."
                        )
                    else:
                        parts.append(
                            "**Cannot apply cleaning** — host dry-run checks failed."
                        )
                    if ghost_apply_cols:
                        gtxt = ", ".join(f"`{c}`" for c in ghost_apply_cols)
                        parts.append(
                            f"**Imputation gap:** missing values decrease on {gtxt} (aligned on "
                            "the synthetic row id), but the plan JSON has no matching "
                            "**imputation** action for those columns."
                        )
                    if retain_violation_cols:
                        rtx = ", ".join(f"`{c}`" for c in retain_violation_cols)
                        parts.append(
                            f"**Retain violated:** aligned missingness decreases on {rtx}, but "
                            "the plan lists **retain missing values** without imputation—code "
                            "must not fill nulls there."
                        )
                    parts.append(
                        "Try **Regenerate Code to Match Plan** or **Regenerate plan from scratch**."
                    )
                    st.error("\n\n".join(parts))
                elif apply_regen_rerun:
                    st.warning(
                        "Cleaning failed; the model produced a revised plan and code. "
                        "Review the update, then try Apply cleaning again."
                    )
                    st.rerun()
                elif apply_exhausted and apply_exhausted_err is not None:
                    st.error(
                        f"Cleaning still failing after {MAX_EXECUTE_FIXES} automatic "
                        f"fix(es): {apply_exhausted_err}"
                    )
                elif apply_success:
                    if apply_host_sync_notice:
                        st.info(apply_host_sync_notice)
                    st.success("Cleaning complete.")

        if st.session_state.get("cleaning_apply_exhausted"):
            if st.button("Regenerate plan from scratch"):
                st.session_state.pop("pending_cleaner_code", None)
                st.session_state.pop("pending_cleaning_plan", None)
                st.session_state.pop("pending_function_name", None)
                st.session_state.pop("execute_fix_count", None)
                st.session_state.pop("preview_df_cleaned", None)
                st.session_state["cleaning_apply_exhausted"] = False
                _invalidate_plan_row_stats_cache()
                st.session_state.pop("plan_snapshot_for_code", None)
                st.session_state.pop("plan_dirty", None)
                st.session_state.pop("plan_widget_nonce", None)
                st.session_state.pop("plan_regen_exclusion_instructions", None)
                st.session_state.pop("plan_regen_verification_feedback", None)
                st.session_state.pop("plan_regen_user_context", None)
                st.session_state.pop("plan_regen_cumulative_removed_actions", None)
                st.rerun()

    df_cleaned_stored = st.session_state.get("preview_df_cleaned")

    if "preview_df_input" in st.session_state:
        st.subheader("Cleaned Data")
        if df_cleaned_stored is None:
            st.info("Generate a plan and apply cleaning to see results here.")
            st.download_button(
                "Download Cleaned Data",
                data="",
                file_name="cleaned_data.csv",
                mime="text/csv",
                disabled=True,
            )
        else:
            st.write(
                f"Shape: {df_cleaned_stored.shape[0]} rows × "
                f"{df_cleaned_stored.shape[1]} columns"
            )
            n_clean = len(df_cleaned_stored)
            if n_clean > 0:
                max_k = min(50, n_clean)
                default_k = min(10, max_k)
            else:
                max_k = 1
                default_k = 1
            _preview_k_state = "cleaning_preview_k"
            if _preview_k_state not in st.session_state:
                st.session_state[_preview_k_state] = default_k
            else:
                try:
                    cur = int(st.session_state[_preview_k_state])
                except TypeError, ValueError:
                    cur = default_k
                st.session_state[_preview_k_state] = min(max(cur, 1), max_k)
            k_preview = st.slider(
                "Preview rows (k)",
                min_value=1,
                max_value=max_k,
                step=1,
                help=(
                    "Up to this many preview rows: every differing row is "
                    "listed first (most changed columns first); if fewer than "
                    "k rows differ, matching rows are added to reach k when "
                    "possible."
                ),
                key=_preview_k_state,
            )
            to_export = reorder_cleaned_for_export(
                st.session_state["preview_df_input"],
                df_cleaned_stored,
                AGENT_ROW_ID,
            )
            csv = to_export.to_csv(index=False)
            st.download_button(
                "Download Cleaned Data",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv",
            )
            preview = preview_aligned_frames(
                st.session_state["preview_df_input"],
                df_cleaned_stored,
                AGENT_ROW_ID,
                k=k_preview,
            )
            if (
                preview.before_view.empty
                and preview.after_view.empty
                and preview.only_in_before.empty
                and preview.only_in_after.empty
                and not df_cleaned_stored.empty
                and k_preview > 0
            ):
                st.info(
                    "No differing rows in the preview window (no column mismatches "
                    "on overlapping rows, and no rows only in upload or only in "
                    "cleaned)."
                )
            if not preview.aligned:
                st.warning(
                    "Could not align rows on the **synthetic row id** this app adds "
                    "for matching; previews compare rows **by position**, show "
                    "up to **k** rows with the most column changes (ties: earlier "
                    "row first). Rows may not correspond to the same logical record."
                )
            facts = build_cleaning_outcome_facts(
                st.session_state["preview_df_input"],
                df_cleaned_stored,
                row_id_col=AGENT_ROW_ID,
            )
            plan_warn = st.session_state.get("pending_cleaning_plan")
            if isinstance(plan_warn, dict):
                ghost_impute = (
                    columns_where_missingness_dropped_without_plan_imputation(
                        st.session_state["preview_df_input"],
                        df_cleaned_stored,
                        plan_warn,
                        row_id_col=AGENT_ROW_ID,
                    )
                )
                retain_violation = (
                    columns_where_retain_missing_plan_violated_by_execution(
                        st.session_state["preview_df_input"],
                        df_cleaned_stored,
                        plan_warn,
                        row_id_col=AGENT_ROW_ID,
                    )
                )
                if ghost_impute:
                    cols_txt = ", ".join(f"`{c}`" for c in ghost_impute)
                    st.warning(
                        f"Missing values decreased on {cols_txt} (matched rows on the "
                        "synthetic row id), but the cleaning plan does **not** list an "
                        "**imputation** step for "
                        f"{'those columns' if len(ghost_impute) > 1 else 'that column'}. "
                        "The generated code likely filled values without recording them "
                        "in the plan JSON. Try **Regenerate plan from scratch** or "
                        "**Regenerate Code to Match Plan** after editing the plan."
                    )
                if retain_violation:
                    rtx = ", ".join(f"`{c}`" for c in retain_violation)
                    st.warning(
                        f"Missing values decreased on {rtx} (matched rows on the synthetic "
                        "row id), but the plan lists **retain missing values** for "
                        f"{'those columns' if len(retain_violation) > 1 else 'that column'} "
                        "without **imputation**—the cleaner likely still filled or removed "
                        "nulls there. Try **Regenerate Code to Match Plan** or **Regenerate plan "
                        "from scratch**."
                    )
            if outcome_facts_show_any_change(facts):
                with st.expander("What Actually Changed", expanded=False):
                    st.markdown(format_outcome_summary_markdown(facts))
            st.subheader("Preview")
            st.caption(
                "Mismatching rows first (most changed columns), then matching "
                "rows to fill up to k when fewer than k rows differ. "
                "Shading marks differing cells; numbers rounded to 2 decimals."
            )
            before_disp, after_disp = style_preview_pair(
                preview.before_view, preview.after_view
            )
            col_before, col_after = st.columns(2)
            with col_before:
                st.caption("Before (upload)")
                st.dataframe(
                    before_disp,
                    width="stretch",
                    height="content",
                    hide_index=True,
                )
            with col_after:
                st.caption("After (cleaned)")
                st.dataframe(
                    after_disp,
                    width="stretch",
                    height="content",
                    hide_index=True,
                )
            if preview.aligned and (
                not preview.only_in_before.empty or not preview.only_in_after.empty
            ):
                st.subheader("Added / removed rows (by id)")
                c_rm, c_add = st.columns(2)
                with c_rm:
                    st.caption("Only in upload (removed from cleaned)")
                    st.dataframe(
                        preview.only_in_before,
                        width="stretch",
                        height="content",
                        hide_index=True,
                    )
                with c_add:
                    st.caption("Only in cleaned (new rows)")
                    st.dataframe(
                        preview.only_in_after,
                        width="stretch",
                        height="content",
                        hide_index=True,
                    )
