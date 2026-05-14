"""Streamlit interface for the Data Cleaning Agent."""

import pandas as pd
import streamlit as st
from data_cleaning_agent import LightweightDataCleaningAgent
from data_cleaning_agent.cleaning_outcome_summary import (
    build_cleaning_outcome_facts,
    format_outcome_summary_markdown,
    outcome_facts_show_any_change,
)
from data_cleaning_agent.plan_column_summary import plan_columns_to_summary_rows
from data_cleaning_agent.utils import run_cleaner_code_on_dataframe, sanitize_cleaning_plan
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

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    upload_fp = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("_cleaning_upload_fp") != upload_fp:
        for k in (
            "preview_df_input",
            "preview_df_cleaned",
            "pending_cleaner_code",
            "pending_cleaning_plan",
            "pending_function_name",
            "cleaning_user_instructions",
        ):
            st.session_state.pop(k, None)
        st.session_state["_cleaning_upload_fp"] = upload_fp

    df_uploaded = pd.read_csv(uploaded_file)

    supplemental_instructions = (
        f'The column "{AGENT_ROW_ID}" is a synthetic stable row identifier '
        "added by the application before cleaning. Do not drop it, rename it, "
        "or change its values. Carry it through unchanged for every row that "
        "remains in the returned DataFrame so before-and-after rows can be aligned."
    )

    st.text_area(
        "Cleaning instructions",
        height=120,
        key="cleaning_user_instructions",
        help=(
            "Describe how you want the data cleaned. Edit these instructions and "
            "click Generate again for a new plan and code."
        ),
    )

    gen_label = (
        "Generate cleaning plan"
        if not st.session_state.get("pending_cleaner_code")
        else "Generate again"
    )

    if st.button(gen_label):
        with st.spinner("Generating plan and code..."):
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            agent = LightweightDataCleaningAgent(model=llm, log=True)
            df_input = df_uploaded.copy()
            df_input.insert(
                0, AGENT_ROW_ID, pd.RangeIndex(stop=len(df_input), dtype="int64")
            )
            raw_ui = st.session_state.get("cleaning_user_instructions")
            user_instructions = (
                raw_ui.strip()
                if isinstance(raw_ui, str) and raw_ui.strip()
                else None
            )
            agent.generate_cleaning_code(
                source_df=df_input,
                user_instructions=user_instructions,
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
        st.success(
            "Plan generated. Review the summary and code below, then apply when ready."
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
        if pending_plan is None:
            st.subheader("Cleaning plan")
            st.warning(
                "Structured plan JSON was missing or invalid. You can still run "
                "cleaning with Apply below—review the generated code first."
            )
        else:
            cols = pending_plan.get("columns")
            notes = pending_plan.get("notes") or ""
            with st.expander(
                "Cleaning plan (read-only)", expanded=False, key="cleaning_plan_main"
            ):
                if isinstance(cols, list):
                    summary_rows = plan_columns_to_summary_rows(pending_plan)
                    if summary_rows:
                        st.dataframe(
                            pd.DataFrame(summary_rows),
                            width="stretch",
                            hide_index=True,
                        )
                    else:
                        st.caption("No per-column entries (columns list empty).")
                else:
                    st.dataframe(
                        pd.DataFrame(cols) if cols is not None else pd.DataFrame(),
                        width="stretch",
                        hide_index=True,
                    )
                    st.caption(
                        "Expected a list-shaped ``columns`` plan for the step summary."
                    )
                if notes:
                    st.markdown("**Notes**")
                    st.write(notes)

        with st.expander("Generated cleaning code", expanded=False):
            st.code(pending_code, language="python")

        if st.button("Apply Cleaning"):
            apply_err: str | None = None
            with st.spinner("Applying cleaning..."):
                fn_work = (
                    st.session_state.get("pending_function_name") or "data_cleaner"
                )
                code_work = pending_code
                df_pre, err = run_cleaner_code_on_dataframe(
                    code_work,
                    df_input_stored,
                    function_name=fn_work,
                )
                if err:
                    apply_err = str(err)
                elif df_pre is None:
                    apply_err = "cleaner returned no result"
                else:
                    st.session_state["preview_df_cleaned"] = df_pre
            if apply_err is not None:
                st.error(f"Cleaning failed: {apply_err}")
            else:
                st.success("Cleaning complete.")

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
                except (TypeError, ValueError):
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
