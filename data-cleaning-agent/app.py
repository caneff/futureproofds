"""Streamlit interface for the Data Cleaning Agent."""

import hashlib
import html

import pandas as pd
import streamlit as st
from data_cleaning_agent import LightweightDataCleaningAgent
from data_cleaning_agent.utils import (
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

load_dotenv()

st.set_page_config(
    page_title="Data Cleaning Agent",
    layout="wide",
)

st.title("🧹 Data Cleaning Agent")

MAX_EXECUTE_FIXES = 3


def _render_cleaning_plan_columns(columns: object) -> None:
    """Render the plan table with chip actions; colors match Streamlit dark theme."""
    if not isinstance(columns, list) or not columns:
        return

    # Streamlit dark surfaces (~ #0E1117 app, #262730 cards); light text for contrast.
    _chip = (
        "display:inline-block;margin:0 8px 6px 0;padding:4px 11px;border-radius:9999px;"
        "font-size:0.8125rem;line-height:1.4;color:rgba(245,247,255,0.95);"
        "background:rgba(120,132,170,0.28);border:1px solid rgba(180,190,230,0.28);"
        "box-shadow:0 1px 0 rgba(255,255,255,0.06) inset"
    )
    _chips_wrap = (
        "display:flex;flex-wrap:wrap;gap:6px 8px;align-items:center;line-height:1.45"
    )

    def _chips_cell(raw_actions: object) -> str:
        dash = "<span style='color:rgba(250,250,250,0.38);font-size:0.85em'>—</span>"
        if isinstance(raw_actions, list):
            if not raw_actions:
                return dash
            chips = "".join(
                f"<span style='{_chip}'>{html.escape(str(a))}</span>"
                for a in raw_actions
            )
            return f"<div style='{_chips_wrap}'>{chips}</div>"
        if raw_actions is None:
            return dash
        inner = f"<span style='{_chip}'>{html.escape(str(raw_actions))}</span>"
        return f"<div style='{_chips_wrap}'>{inner}</div>"

    _th = (
        "text-align:left;padding:10px 14px;font-weight:600;letter-spacing:0.02em;"
        "color:rgba(250,250,250,0.94);"
        "background:linear-gradient(180deg,#32343e 0%,#262730 100%);"
        "border-bottom:1px solid rgba(255,255,255,0.1)"
    )
    _name_td = (
        "vertical-align:top;padding:12px 14px;border-bottom:1px solid rgba(255,255,255,0.08);"
        "white-space:nowrap;width:22%;font-weight:600;color:rgba(230,234,240,0.98);"
        "font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;font-size:0.88em"
    )
    _act_td = (
        "vertical-align:top;padding:10px 14px 12px;border-bottom:1px solid rgba(255,255,255,0.08);"
        "width:78%;color:rgba(250,250,250,0.9)"
    )

    rows_html: list[str] = []
    rendered = 0
    for row in columns:
        if not isinstance(row, dict):
            continue
        name = html.escape(str(row.get("name", "")))
        actions_cell = _chips_cell(row.get("actions"))
        row_bg = "rgba(255,255,255,0.045)" if rendered % 2 else "transparent"
        rendered += 1
        rows_html.append(
            f"<tr style='background:{row_bg}'>"
            f"<td style='{_name_td}'>{name}</td>"
            f"<td style='{_act_td}'>{actions_cell}</td>"
            "</tr>"
        )

    if not rows_html:
        return

    wrap = (
        "border-radius:12px;overflow:hidden;"
        "border:1px solid rgba(255,255,255,0.12);"
        "box-shadow:0 2px 12px rgba(0,0,0,0.35);"
        "background:rgba(38,39,48,0.92)"
    )
    tbl = (
        "width:100%;border-collapse:collapse;table-layout:fixed;font-size:0.92rem;"
        "color:rgba(250,250,250,0.92)"
    )

    table = (
        f"<div style='{wrap}'>"
        f"<table style='{tbl}'>"
        "<colgroup><col style='width:22%' /><col style='width:78%' /></colgroup>"
        "<thead><tr>"
        f"<th style='{_th}'>Column</th>"
        f"<th style='{_th}'>Actions</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table></div>"
    )
    st.markdown(table, unsafe_allow_html=True)


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
        st.session_state["_cleaning_upload_fp"] = upload_fp

    df_raw = pd.read_csv(uploaded_file)

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
            df_input = df_raw.copy()
            df_input.insert(
                0, AGENT_ROW_ID, pd.RangeIndex(stop=len(df_input), dtype="int64")
            )
            agent.generate_cleaning_code(
                data_raw=df_input,
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

    if pending_code and df_input_stored is not None:
        _refresh_plan_row_stats_if_needed(
            pending_code,
            st.session_state.get("pending_function_name") or "data_cleaner",
            df_input_stored,
            upload_fp,
        )
        st.subheader("Cleaning plan")
        if pending_plan is None:
            st.warning(
                "Structured plan JSON was missing or invalid. You can still run "
                "cleaning with Apply cleaning below—review the generated code first."
            )
        else:
            cols = pending_plan.get("columns")
            row_ops = pending_plan.get("row_ops") or []
            notes = pending_plan.get("notes") or ""
            if cols:
                if isinstance(cols, list):
                    _render_cleaning_plan_columns(cols)
                else:
                    st.dataframe(pd.DataFrame(cols), width="stretch", hide_index=True)
            else:
                st.caption("No per-column entries (columns list empty).")
            stats = st.session_state.get("plan_row_stats")
            show_row_ops = bool(row_ops)
            if isinstance(stats, dict) and not stats.get("error"):
                if stats.get("removed_total", 0) == 0 and "n_in" in stats:
                    show_row_ops = False
            if show_row_ops:
                st.markdown("**Row operations**")
                for op in row_ops:
                    st.write(f"- {op}")
            if isinstance(stats, dict):
                if stats.get("error"):
                    st.caption(
                        "Could not verify row counts (cleaner failed when run for "
                        f"measurement): {stats['error']}"
                    )
                elif "n_in" in stats and "n_out" in stats:
                    if stats.get("removed_total", 0) > 0:
                        st.caption(
                            "Verified for this dataset (one execution of the generated "
                            f"cleaner): {stats['n_in']:,} → {stats['n_out']:,} rows "
                            f"({stats['removed_total']:,} removed in total)."
                        )
                        rnull = stats.get("removed_all_null_raw_user_cols")
                        if rnull is not None:
                            st.caption(
                                f"Of removed rows, {rnull:,} were all-null on original "
                                "data columns (excluding the synthetic row id) before cleaning."
                            )
            if notes:
                st.markdown("**Notes**")
                st.write(notes)

        with st.expander("Generated cleaning code"):
            st.code(pending_code, language="python")

        if st.button("Apply cleaning"):
            with st.spinner("Applying cleaning..."):
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                agent = LightweightDataCleaningAgent(model=llm, log=True)
                fn = st.session_state.get("pending_function_name") or "data_cleaner"
                agent.response = {
                    "data_cleaner_function": pending_code,
                    "cleaning_plan": pending_plan,
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
                        st.session_state["execute_fix_count"] = fc + 1
                        st.session_state["cleaning_apply_exhausted"] = False
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
                    st.session_state["execute_fix_count"] = 0
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
            k_preview = int(
                st.number_input(
                    "Preview rows (k)",
                    min_value=1,
                    max_value=max_k,
                    value=default_k,
                    step=1,
                    width=220,
                    help=(
                        "Up to this many differing rows, preferring those with the "
                        "**most** changed columns (ties: earlier id in cleaned order, "
                        "or earlier row index when not aligned)."
                    ),
                )
                or default_k
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
            st.subheader("Preview")
            st.caption(
                "Sorted by most changed columns, up to k above. "
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
