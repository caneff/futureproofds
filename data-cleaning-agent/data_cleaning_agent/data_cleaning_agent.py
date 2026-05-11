# Libraries
from typing import Required, TypedDict
import os
import logging

import pandas as pd

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.types import Checkpointer
from langgraph.graph import StateGraph, END

from .utils import (
    PythonOutputParser,
    execute_agent_code,
    fix_agent_code,
    format_dataframe_summary,
    get_dataframe_summary,
)

# Setup
logger = logging.getLogger(__name__)
AGENT_NAME = "lightweight_data_cleaning_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")


# State schema for the workflow graph. total=False because nodes incrementally
# populate keys (data_cleaner_function, data_cleaned, data_cleaner_error, ...)
# during the run rather than requiring all keys up front.
class GraphState(TypedDict, total=False):
    user_instructions: Required[str | None]
    data_raw: Required[dict]
    max_retries: Required[int]
    retry_count: Required[int]
    data_cleaned: dict
    data_cleaner_function: str
    data_cleaner_function_path: str
    data_cleaner_function_name: str
    data_cleaner_error: str


class LightweightDataCleaningAgent:
    """
    LLM-powered agent that generates and executes Python code to clean pandas DataFrames.
    
    Uses an LLM to create data cleaning functions based on user instructions. The agent
    automatically retries with error correction if the generated code fails.
    
    Parameters
    ----------
    model : LLM
        Language model for generating cleaning code (e.g., ChatOpenAI).
    log : bool, default=False
        Whether to save generated code to a file.
    log_path : str, optional
        Directory for log files. Defaults to './logs/' if log=True and not specified.
    file_name : str, default="data_cleaner.py"
        Name of the log file when log=True.
    function_name : str, default="data_cleaner"
        Name of the generated cleaning function.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving agent state.
    
    Attributes
    ----------
    response : dict or None
        Stores the full response after invoke_agent() is called.
    """
    
    def __init__(
        self, 
        model, 
        log=False, 
        log_path=None, 
        file_name="data_cleaner.py", 
        function_name="data_cleaner",
        checkpointer: Checkpointer = None
    ):
        self.model = model
        self.log = log
        self.log_path = log_path
        self.file_name = file_name
        self.function_name = function_name
        self.checkpointer = checkpointer
        self.response = None
        # Build the LangGraph workflow with code generation, execution, and error fixing nodes
        self._compiled_graph = make_lightweight_data_cleaning_agent(
            model=model,
            log=log,
            log_path=log_path,
            file_name=file_name,
            function_name=function_name,
            checkpointer=checkpointer
        )
    
    def invoke_agent(
        self,
        data_raw: pd.DataFrame,
        user_instructions: str | None = None,
        max_retries: int = 3,
        retry_count: int = 0,
        config: RunnableConfig | None = None,
    ) -> None:
        """
        Generate and execute data cleaning code on the provided DataFrame.

        Parameters
        ----------
        data_raw : pd.DataFrame
            Raw dataset to clean.
        user_instructions : str, optional
            Custom cleaning instructions. If None, applies default cleaning steps:
            removing columns with >40% missing values, imputing missing values,
            and removing duplicates.
        max_retries : int, default=3
            Maximum number of retry attempts if generated code fails.
        retry_count : int, default=0
            Starting retry count (typically left at 0).
        config : RunnableConfig, optional
            LangChain runtime config (e.g. ``{"configurable": {"thread_id": ...}}``)
            forwarded to the underlying graph ``invoke`` call.

        Returns
        -------
        None
            Results are stored in self.response and accessed via getter methods.
        """
        initial_state: GraphState = {
            "user_instructions": user_instructions,
            "data_raw": data_raw.to_dict(),
            "max_retries": max_retries,
            "retry_count": retry_count,
        }
        self.response = self._compiled_graph.invoke(initial_state, config=config)
    
    def get_data_cleaned(self):
        """
        Retrieves the cleaned data stored after running invoke_agent.
        """
        if self.response:
            return pd.DataFrame(self.response.get("data_cleaned"))
        
    def get_data_raw(self):
        """
        Retrieves the raw data.
        """
        if self.response:
            return pd.DataFrame(self.response.get("data_raw"))
    
    def get_data_cleaner_function(self):
        """
        Retrieves the agent's cleaning function code.
        """
        if self.response:
            return self.response.get("data_cleaner_function")


# Agent Factory Function

def make_lightweight_data_cleaning_agent(
    model, 
    log=False, 
    log_path=None, 
    file_name="data_cleaner.py",
    function_name="data_cleaner",
    checkpointer: Checkpointer = None
):
    """
    Factory function that creates a compiled LangGraph workflow for data cleaning.
    
    Builds a state graph with three nodes: code generation, execution, and error fixing.
    The workflow automatically retries with corrections if generated code fails.
    
    Parameters
    ----------
    model : LLM
        Language model for generating cleaning code.
    log : bool, default=False
        Whether to save generated code to a file.
    log_path : str, optional
        Directory for log files. Defaults to './logs/' if log=True and not specified.
    file_name : str, default="data_cleaner.py"
        Name of the log file when log=True.
    function_name : str, default="data_cleaner"
        Name of the generated cleaning function.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.
    
    Returns
    -------
    CompiledStateGraph
        Compiled LangGraph workflow ready to process cleaning requests.
    """
    # Setup Log Directory
    log_dir: str = log_path if log_path is not None else LOG_PATH
    if log and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    def create_data_cleaner_code(state: GraphState):
        """
        Generate the data cleaning code based on user instructions.
        """
        logger.info("Creating data cleaner code")

        data_raw = state["data_raw"]
        df = pd.DataFrame.from_dict(data_raw)

        summary = get_dataframe_summary(df)
        dataset_summary = format_dataframe_summary(summary)


        data_cleaning_prompt = PromptTemplate(
            template="""
            You are a Data Cleaning Agent. Create a {function_name}(data_raw) function that
            returns a cleaned pandas DataFrame.

            Follow these rules strictly. Do not reorder steps. Do not skip steps.

            Hard constraints:
            - Start with: df = data_raw.copy(). Never mutate data_raw.
            - Be deterministic. Do not use randomness. If you must, seed it with 0.
            - Never drop or destructively transform any column named in User Instructions.
              Treat those as protected (target/id columns).
            - Preserve original column order except for columns that are dropped.
            - Reset the index at the end after any row drops.

            Pipeline (in order):
            1. df = data_raw.copy().
            2. Normalize column names: lowercase, strip, replace non-alphanumeric runs with
               a single underscore.
            3. For object/string columns, strip leading/trailing whitespace. For columns
               that look like categorical labels (not free text), also casefold values.
            4. Replace placeholder strings with NaN in object columns:
               "", "N/A", "n/a", "NA", "null", "NULL", "None", "?", "missing", "-", "unknown".
            5. Coerce dtypes where the column clearly fits:
               - Date-like strings: pd.to_datetime(col, errors="coerce").
               - Numeric-looking strings (currency, percent, thousands separators): strip
                 "$", ",", "%" then pd.to_numeric(col, errors="coerce").
               - Boolean-like strings ("yes"/"no", "true"/"false", "t"/"f", "0"/"1"): map to bool.
            6. Drop columns with more than 40% missing values, EXCEPT any column listed
               in User Instructions.
            7. Drop columns that are constant (one unique non-null value) or 100% NaN.
            8. Identify ID-like columns (cardinality == len(df), name ends with "id" or
               "uuid", or strictly monotonically increasing integers). Exempt them from
               steps 9, 10, and 11 (categorical detection, rare-bucketing, imputation).
               Do not drop them.
            9. Convert columns with fewer than 10 unique values (after step 3
               canonicalization) into pd.Categorical with the observed categories.
            10. For each categorical column, bucket categories whose frequency is below
                1% into a single "other" category. Keep a "_raw" version of any
                categorical variable where you make an "other" category so original
                categories are not lost.
            11. Impute missing values:
                - Numeric columns: use median if abs(skew) > 1, otherwise mean.
                - Categorical/object columns: use mode if missing fraction <= 20%,
                  otherwise add and use an "unknown" sentinel category.
            12. Drop rows that are entirely NaN: df.dropna(how="all").
            13. Drop exact duplicate rows: df.drop_duplicates().
            14. df = df.reset_index(drop=True).

            User Instructions:
            {user_instructions}

            Dataset Summary:
            {all_datasets_summary}

            Return Python code in ```python``` format with a single function:

            def {function_name}(data_raw):
                import pandas as pd
                import numpy as np
                # Your cleaning code here, following the pipeline above in order.
                return data_cleaned

            Important: when fit_transform()-style outputs need to be assigned to a
            DataFrame column, flatten with .ravel() first.
            """,
            input_variables=["user_instructions", "all_datasets_summary", "function_name"]
        )

        data_cleaning_agent = data_cleaning_prompt | model | PythonOutputParser()
        
        response = data_cleaning_agent.invoke({
            "user_instructions": state.get("user_instructions") or "Follow the basic cleaning steps.",
            "all_datasets_summary": dataset_summary,
            "function_name": function_name
        })
        
        # Simple logging if enabled
        file_path = None
        if log:
            file_path = os.path.join(log_dir, file_name)
            with open(file_path, 'w') as f:
                f.write(response)
            logger.info(f"Code saved to: {file_path}")
   
        return {
            "data_cleaner_function": response,
            "data_cleaner_function_path": file_path,
            "data_cleaner_function_name": function_name,
        }
        
    def execute_data_cleaner_code(state):
        """
        Execute the generated cleaning code on the data.
        """
        return execute_agent_code(
            state=state,
            data_key="data_raw",
            result_key="data_cleaned",
            error_key="data_cleaner_error",
            code_snippet_key="data_cleaner_function",
            agent_function_name=state.get("data_cleaner_function_name")
        )
        
    def fix_data_cleaner_code(state: GraphState):
        """
        Fix errors in the generated data cleaning code.
        """
        data_cleaner_prompt = """
        You are a Data Cleaning Agent. Fix the broken {function_name}() function.
        
        Return Python code in ```python``` format with the corrected function definition.
        
        Broken code: 
        {code_snippet}

        Error:
        {error}
        """

        return fix_agent_code(
            state=state,
            code_snippet_key="data_cleaner_function",
            error_key="data_cleaner_error",
            llm=model,  
            prompt_template=data_cleaner_prompt,
            function_name=state.get("data_cleaner_function_name"),
        )

    # Build the workflow graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("create_data_cleaner_code", create_data_cleaner_code)
    workflow.add_node("execute_data_cleaner_code", execute_data_cleaner_code)
    workflow.add_node("fix_data_cleaner_code", fix_data_cleaner_code)
    
    # Set entry point
    workflow.set_entry_point("create_data_cleaner_code")
    
    # Add edges
    workflow.add_edge("create_data_cleaner_code", "execute_data_cleaner_code")
    workflow.add_edge("fix_data_cleaner_code", "execute_data_cleaner_code")
    
    # Conditional routing: retry with fixes if error occurs and retries remain
    def should_retry(state):
        has_error = state.get("data_cleaner_error") is not None
        can_retry = (
            state.get("retry_count") is not None
            and state.get("max_retries") is not None
            and state["retry_count"] < state["max_retries"]
        )
        return "fix_code" if (has_error and can_retry) else "end"
    
    workflow.add_conditional_edges(
        "execute_data_cleaner_code",
        should_retry,
        {
            "fix_code": "fix_data_cleaner_code",
            "end": END,
        },
    )
    
    # Compile the workflow
    app = workflow.compile(checkpointer=checkpointer, name=AGENT_NAME)
    
    return app
