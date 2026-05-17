"""Streamlit interface for the Data Cleaning Agent."""

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from data_cleaning_agent import LightweightDataCleaningAgent
from data_cleaning_agent.cleaning_outcome_summary import (
    build_cleaning_outcome_facts,
    format_outcome_summary_markdown,
    outcome_facts_show_any_change,
)
from data_cleaning_agent.cleaning_plan import format_plan_summary_markdown
from preview_helpers import (
    AGENT_ROW_ID,
    preview_aligned_frames,
    reorder_cleaned_for_export,
    style_preview_pair,
)


def _synthetic_row_id_series(index: pd.Index) -> pd.Series:
    """Stable row keys as pandas string dtype (safe for naive ``.str`` loops in generated code)."""
    return pd.Series([str(i) for i in range(len(index))], index=index, dtype="string")


def _normalize_cleaned_row_id(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``AGENT_ROW_ID`` to str so joins match the upload frame after numeric cleaners."""
    if AGENT_ROW_ID not in df.columns:
        return df
    out = df.copy()
    out[AGENT_ROW_ID] = out[AGENT_ROW_ID].astype(str)
    return out


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
            "pending_cleaning_plan",
            "cleaning_user_instructions",
        ):
            st.session_state.pop(k, None)
        st.session_state["_cleaning_upload_fp"] = upload_fp

    df_uploaded = pd.read_csv(uploaded_file)

    with st.expander("Cleaning instructions", expanded=False):
        st.text_area(
            "cleaning_instructions_editor",
            height=120,
            key="cleaning_user_instructions",
            label_visibility="collapsed",
            placeholder="Optional. Describe how you want the data cleaned.",
            help=(
                "Edit these instructions and click Generate again for a new cleaning plan."
            ),
        )

    gen_label = (
        "Generate cleaning plan"
        if not st.session_state.get("pending_cleaning_plan")
        else "Generate again"
    )

    if st.button(gen_label):
        with st.spinner("Generating cleaning plan..."):
            llm = ChatOpenAI(**{"model": "gpt-4o-mini", "temperature": 0})
            agent = LightweightDataCleaningAgent(model=llm)
            df_input = df_uploaded.copy()
            df_input.insert(0, AGENT_ROW_ID, _synthetic_row_id_series(df_input.index))
            raw_ui = st.session_state.get("cleaning_user_instructions")
            user_instructions = (
                raw_ui.strip() if isinstance(raw_ui, str) and raw_ui.strip() else None
            )
            agent.generate_cleaning_plan(
                source_df=df_input,
                user_instructions=user_instructions,
            )
            plan = agent.get_cleaning_plan()
            if plan is None:
                st.error("Failed to generate a cleaning plan.")
            else:
                st.session_state["pending_cleaning_plan"] = plan
                st.session_state["preview_df_input"] = df_input
                st.session_state.pop("preview_df_cleaned", None)
        if st.session_state.get("pending_cleaning_plan") is not None:
            st.success(
                "Cleaning plan generated. Expand the plan section below to "
                "review, then apply when ready."
            )

    pending_plan = st.session_state.get("pending_cleaning_plan")
    df_input_stored = st.session_state.get("preview_df_input")

    if pending_plan and df_input_stored is not None:
        with st.expander("Generated cleaning plan", expanded=False):
            st.markdown(
                format_plan_summary_markdown(
                    pending_plan,
                    row_id_col=AGENT_ROW_ID,
                )
            )

        if st.button("Apply Cleaning"):
            apply_err: str | None = None
            with st.spinner("Applying cleaning..."):
                from data_cleaning_agent.cleaning_pipeline import run_cleaning_pipeline

                try:
                    df_pre, _trace = run_cleaning_pipeline(
                        df_input_stored,
                        pending_plan,
                        row_id_col=AGENT_ROW_ID,
                    )
                except Exception as exc:
                    apply_err = str(exc)
                else:
                    st.session_state["preview_df_cleaned"] = _normalize_cleaned_row_id(
                        df_pre
                    )
            if apply_err is not None:
                st.error(f"Cleaning failed: {apply_err}")
            else:
                st.success("Cleaning complete.")

    df_cleaned_stored = st.session_state.get("preview_df_cleaned")

    if "preview_df_input" in st.session_state:
        st.subheader("Cleaned Data")
        if df_cleaned_stored is None:
            st.info("Generate a cleaning plan and apply to see results here.")
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
