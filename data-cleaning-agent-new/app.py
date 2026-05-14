"""Streamlit interface for the Data Cleaning Agent."""

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from data_cleaning_agent import LightweightDataCleaningAgent

load_dotenv()

st.title("🧹 Data Cleaning Agent")

# Upload file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    # Load data
    df_raw = pd.read_csv(uploaded_file)
    
    # Clean button
    if st.button("Clean Data"):
        with st.spinner("Cleaning..."):
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            agent = LightweightDataCleaningAgent(model=llm, log=True)
            agent.invoke_agent(data_raw=df_raw)
            df_cleaned = agent.get_data_cleaned()
            
            st.success("Done!")
            
            st.subheader("Cleaned Data")
            st.write(f"Shape: {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")
            st.dataframe(df_cleaned.head())
            
            # Download
            csv = df_cleaned.to_csv(index=False)
            st.download_button(
                "Download Cleaned Data",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )
