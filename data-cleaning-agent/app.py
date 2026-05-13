"""Streamlit interface for the Data Cleaning Agent."""

import copy
import hashlib
from hashlib import md5

import pandas as pd
import streamlit as st
from data_cleaning_agent import LightweightDataCleaningAgent
from data_cleaning_agent.utils import (
    merged_plan_actions_by_column,
    removed_plan_actions,
    run_cleaner_code_on_dataframe,
    sanitize_cleaning_plan,
    summarize_cleaning_row_effects,
)
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from preview_helpers import (
    AGENT_ROW_ID,
    preview_aligned_frames,
    reorder_cleaned_for_export,
    style_preview_pair,
)
from cleaning_outcome_summary import (
    build_cleaning_outcome_facts,
    format_outcome_summary_markdown,
)
from row_stats_narrative import glossary_bullets, verified_row_stats_strip_items

load_dotenv()

st.set_page_config(
    page_title="Data Cleaning Agent",
    layout="wide",
)

st.title("🧹 Data Cleaning Agent")

MAX_EXECUTE_FIXES = 3


def _plan_step_checkbox_key(
    *,
    code_digest: str,
    widget_nonce: int,
    column_name: str,
    action: str,
    occurrence: int,
) -> str:
    """Stable widget key for one plan step (survives list shrink when others are removed)."""
    stable = md5(
        f"{column_name}\x00{action}\x00{occurrence}".encode("utf-8")
    ).hexdigest()[:20]
    return f"plan_step_{code_digest}_{widget_nonce}_{stable}"


def _build_plan_exclusion_supplement(snap: dict, cur: dict) -> str:
    """Host-supplement text appended to ``supplemental_instructions`` on plan-edit regen."""
    removed = removed_plan_actions(snap.get("columns"), cur.get("columns"))
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
        "If omitting a step makes a later step unsafe or meaningless, adapt minimally "
        "and document in plan notes."
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
        "**Regenerate code to match plan**."
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
    for exp_i, name in enumerate(names_ordered):
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
        exp_key = f"plan_col_exp_{code_digest}_{widget_nonce}_{exp_i}"
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


def _refresh_plan_row_stats_if_needed(
    pending_code: str,
    function_name: str,
    df_input: pd.DataFrame,
    upload_fp: tuple[str, int],
) -> None:
    """Run cleaner once (cached) and store row-effect summary for the plan UI."""
    digest = hashlib.sha256(pending_code.encode("utf-8")).hexdigest()
    cache_key = (upload_fp, digest, function_name)
    if st.session_state.get("_plan_row_stats_cache_key") == cache_key:
        return
    with st.spinner("Measuring row-level effects…"):
        df_out, err = run_cleaner_code_on_dataframe(
            pending_code,
            df_input,
            function_name=function_name,
        )
        if err is not None or df_out is None:
            st.session_state["plan_row_stats"] = {"error": err or "unknown error"}
        else:
            st.session_state["plan_row_stats"] = summarize_cleaning_row_effects(
                df_input,
                df_out,
                row_id_col=AGENT_ROW_ID,
            )
        st.session_state["_plan_row_stats_cache_key"] = cache_key


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
        st.session_state["_cleaning_upload_fp"] = upload_fp

    df_uploaded = pd.read_csv(uploaded_file)

    supplemental_instructions = (
        f'The column "{AGENT_ROW_ID}" is a synthetic stable row identifier '
        "added by the application before cleaning. Do not drop it, rename it, "
        "or change its values. Carry it through unchanged for every row that "
        "remains in the returned DataFrame so before-and-after rows can be aligned."
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
                supplemental_instructions=supplemental_instructions,
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
        _refresh_plan_row_stats_if_needed(
            pending_code,
            st.session_state.get("pending_function_name") or "data_cleaner",
            df_input_stored,
            upload_fp,
        )
        if pending_plan is None:
            st.subheader("Cleaning plan")
            st.warning(
                "Structured plan JSON was missing or invalid. You can still run "
                "cleaning with Apply cleaning below—review the generated code first."
            )
            st.session_state["plan_dirty"] = False
        else:
            cols = pending_plan.get("columns")
            row_ops = pending_plan.get("row_ops") or []
            notes = pending_plan.get("notes") or ""
            with st.expander("Cleaning plan", expanded=False):
                if cols:
                    if isinstance(cols, list):
                        code_digest = hashlib.sha256(
                            pending_code.encode("utf-8")
                        ).hexdigest()[:24]
                        nonce = int(st.session_state.get("plan_widget_nonce") or 0)
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
                stats = st.session_state.get("plan_row_stats")
                show_row_ops = bool(row_ops)
                if isinstance(stats, dict) and not stats.get("error"):
                    if stats.get("removed_total", 0) == 0 and "n_in" in stats:
                        show_row_ops = False

                st.markdown("**Plan vs verified run**")
                st.caption(
                    "Plan text = intent. Metrics = one run of the current generated "
                    "code on this upload."
                )
                st.markdown("**Verified row counts**")
                st.caption("One run of the current generated code on this upload.")
                strip_items = (
                    verified_row_stats_strip_items(stats)
                    if isinstance(stats, dict)
                    else None
                )
                if strip_items is not None:
                    cols_strip = st.columns(len(strip_items))
                    for col_slot, (lbl, val) in zip(
                        cols_strip, strip_items, strict=True
                    ):
                        with col_slot:
                            st.metric(label=lbl, value=val)
                elif isinstance(stats, dict) and stats.get("error"):
                    st.warning(f"Counts unavailable: {stats['error']}")

                if show_row_ops:
                    st.markdown("**Row operations**")
                    st.caption("From the plan JSON (model intent).")
                    for op in row_ops:
                        st.write(f"- {op}")

                with st.expander("What do these numbers mean?", expanded=False):
                    for line in glossary_bullets():
                        st.markdown(line)

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
                rc1, rc2 = st.columns(2)
                with rc1:
                    if st.button("Regenerate code to match plan", type="primary"):
                        snap_go = st.session_state.get("plan_snapshot_for_code")
                        cur_go = st.session_state.get("pending_cleaning_plan")
                        if not isinstance(snap_go, dict) or not isinstance(
                            cur_go, dict
                        ):
                            st.error("Missing plan state for regeneration.")
                        else:
                            plan_excl = _build_plan_exclusion_supplement(
                                snap_go, cur_go
                            )
                            supplemental_for_regen = (
                                f"{supplemental_instructions}\n\n"
                                "---\n\n"
                                "**Plan-edit exclusion (application UI; follow in addition "
                                "to the host supplemental notes above):**\n\n"
                                f"{plan_excl}"
                            )
                            with st.spinner(
                                "Regenerating code for your edited plan..."
                            ):
                                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                                agent = LightweightDataCleaningAgent(
                                    model=llm, log=True
                                )
                                agent.generate_cleaning_code(
                                    source_df=df_input_stored,
                                    user_instructions=None,
                                    supplemental_instructions=supplemental_for_regen,
                                )
                                new_plan = agent.get_cleaning_plan()
                                if new_plan is not None:
                                    sp = sanitize_cleaning_plan(
                                        new_plan, df_input_stored
                                    )
                                    if sp is not None:
                                        new_plan = sp
                                st.session_state["pending_cleaning_plan"] = new_plan
                                st.session_state["pending_cleaner_code"] = (
                                    agent.get_data_cleaner_function()
                                )
                                st.session_state["pending_function_name"] = (
                                    agent.response.get("data_cleaner_function_name")
                                    if agent.response
                                    else "data_cleaner"
                                )
                                _store_plan_snapshot_after_code_from_llm(
                                    st.session_state.get("pending_cleaning_plan")
                                )
                                _invalidate_plan_row_stats_cache()
                                st.session_state["execute_fix_count"] = 0
                                st.session_state.pop("cleaning_apply_exhausted", None)
                                st.session_state[
                                    "plan_regen_exclusion_instructions"
                                ] = plan_excl
                            st.success(
                                "Code and plan updated from your edits. Review, then apply."
                            )
                            st.rerun()
                with rc2:
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
                            st.rerun()

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
            with st.spinner("Applying cleaning..."):
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                agent = LightweightDataCleaningAgent(model=llm, log=True)
                fn = st.session_state.get("pending_function_name") or "data_cleaner"
                plan_for_exec = st.session_state.get("pending_cleaning_plan")
                agent.response = {
                    "data_cleaner_function": pending_code,
                    "cleaning_plan": plan_for_exec,
                    "data_cleaner_function_name": fn,
                    "retry_count": int(st.session_state.get("execute_fix_count") or 0),
                }
                exec_out = agent.execute_stored_cleaning(df_input_stored)
                err = exec_out.get("data_cleaner_error")

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
                            sp_fix = sanitize_cleaning_plan(plan_fix, df_input_stored)
                            if sp_fix is not None:
                                st.session_state["pending_cleaning_plan"] = sp_fix
                                plan_fix = sp_fix
                        _store_plan_snapshot_after_code_from_llm(
                            st.session_state.get("pending_cleaning_plan")
                        )
                        _invalidate_plan_row_stats_cache()
                        st.session_state["execute_fix_count"] = fc + 1
                        st.session_state["cleaning_apply_exhausted"] = False
                        st.session_state.pop("plan_regen_exclusion_instructions", None)
                        st.warning(
                            "Cleaning failed; the model produced a revised plan and code. "
                            "Review the update, then try Apply cleaning again."
                        )
                        st.rerun()
                    else:
                        st.session_state["cleaning_apply_exhausted"] = True
                        st.error(
                            f"Cleaning still failing after {MAX_EXECUTE_FIXES} automatic "
                            f"fix(es): {err}"
                        )
                else:
                    st.session_state["cleaning_apply_exhausted"] = False
                    st.session_state["preview_df_cleaned"] = agent.get_data_cleaned()
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
                    st.session_state.pop("plan_regen_exclusion_instructions", None)
                    st.success("Cleaning complete.")

        if st.session_state.get("cleaning_apply_exhausted"):
            if st.button("Regenerate plan from scratch"):
                st.session_state.pop("pending_cleaner_code", None)
                st.session_state.pop("pending_cleaning_plan", None)
                st.session_state.pop("pending_function_name", None)
                st.session_state.pop("execute_fix_count", None)
                st.session_state.pop("preview_df_cleaned", None)
                st.session_state["cleaning_apply_exhausted"] = False
                st.session_state.pop("plan_row_stats", None)
                st.session_state.pop("_plan_row_stats_cache_key", None)
                st.session_state.pop("plan_snapshot_for_code", None)
                st.session_state.pop("plan_dirty", None)
                st.session_state.pop("plan_widget_nonce", None)
                st.session_state.pop("plan_regen_exclusion_instructions", None)
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
            if _preview_k_state in st.session_state:
                try:
                    cur = int(st.session_state[_preview_k_state])
                except (TypeError, ValueError):
                    cur = default_k
                st.session_state[_preview_k_state] = min(max(cur, 1), max_k)
            k_preview = st.slider(
                "Preview rows (k)",
                min_value=1,
                max_value=max_k,
                value=default_k,
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
                    "Could not align rows on the synthetic id column "
                    f"({AGENT_ROW_ID}); previews compare rows **by position**, show "
                    "up to **k** rows with the most column changes (ties: earlier "
                    "row first). Rows may not correspond to the same logical record."
                )
            facts = build_cleaning_outcome_facts(
                st.session_state["preview_df_input"],
                df_cleaned_stored,
                row_id_col=AGENT_ROW_ID,
            )
            st.markdown("### What actually changed (verified run)")
            st.markdown(
                format_outcome_summary_markdown(facts, row_id_label=AGENT_ROW_ID)
            )
            stats_for_warn = st.session_state.get("plan_row_stats")
            if isinstance(stats_for_warn, dict) and stats_for_warn.get("error"):
                st.caption(
                    "Row-level subset stats in the plan area may be unavailable; "
                    "this summary still reflects this upload vs cleaned output."
                )
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
