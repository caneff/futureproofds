"""Streamlit interface for the Data Cleaning Agent."""

import pandas as pd
import streamlit as st
from data_cleaning_agent import LightweightDataCleaningAgent
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

# Upload file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    upload_fp = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("_cleaning_upload_fp") != upload_fp:
        st.session_state.pop("preview_df_input", None)
        st.session_state.pop("preview_df_cleaned", None)
        st.session_state["_cleaning_upload_fp"] = upload_fp

    # Load data
    df_raw = pd.read_csv(uploaded_file)

    if st.button("Clean Data"):
        with st.spinner("Cleaning..."):
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            agent = LightweightDataCleaningAgent(model=llm, log=True)
            df_input = df_raw.copy()
            df_input.insert(
                0, AGENT_ROW_ID, pd.RangeIndex(stop=len(df_input), dtype="int64")
            )
            supplemental_instructions = (
                f'The column "{AGENT_ROW_ID}" is a synthetic stable row identifier '
                "added by the application before cleaning. Do not drop it, rename it, "
                "or change its values. Carry it through unchanged for every row that "
                "remains in the returned DataFrame so before-and-after rows can be aligned."
            )
            agent.invoke_agent(
                data_raw=df_input,
                supplemental_instructions=supplemental_instructions,
            )
            df_cleaned = agent.get_data_cleaned()
            st.session_state["preview_df_input"] = df_input
            st.session_state["preview_df_cleaned"] = df_cleaned
        st.success("Done!")

    df_cleaned_stored = st.session_state.get("preview_df_cleaned")

    if "preview_df_input" in st.session_state:
        st.subheader("Cleaned Data")
        if df_cleaned_stored is None:
            st.error("Cleaning did not return a dataframe.")
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
            k_preview = st.slider(
                "Preview rows (k)",
                min_value=1,
                max_value=max_k,
                value=default_k,
                help=(
                    "Up to this many differing rows, preferring those with the "
                    "**most** changed columns (ties: earlier id in cleaned order, "
                    "or earlier row index when not aligned)."
                ),
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
