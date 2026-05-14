# Libraries
import logging
import os
from pathlib import Path
from typing import Required, TypedDict

import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.types import Checkpointer

from .utils import (
    DataCleaningOutputParser,
    execute_agent_code,
    fix_agent_code,
    format_dataframe_summary,
    get_dataframe_summary,
    sanitize_cleaning_plan,
)

# Setup
logger = logging.getLogger(__name__)
AGENT_NAME = "lightweight_data_cleaning_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")


_PROMPT_PATH = Path(__file__).parent / "prompts" / "data_cleaning.md"
_DATA_CLEANING_PROMPT_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")

_FIX_PROMPT_PATH = Path(__file__).parent / "prompts" / "data_cleaning_fix.md"
_FIX_DATA_CLEANER_PROMPT_TEMPLATE = _FIX_PROMPT_PATH.read_text(encoding="utf-8")


def _data_cleaning_generation_chain(model):
    """Prompt | LLM | dual-output parser for create/regenerate flows."""
    data_cleaning_prompt = PromptTemplate(
        template=_DATA_CLEANING_PROMPT_TEMPLATE,
        input_variables=[
            "user_instructions",
            "supplemental_instructions",
            "all_datasets_summary",
            "function_name",
        ],
    )
    return data_cleaning_prompt | model | DataCleaningOutputParser()


def _run_data_cleaning_generation(
    model,
    *,
    user_instructions: str | None,
    supplemental_instructions: str | None,
    source_df: pd.DataFrame,
    function_name: str,
    log: bool,
    log_dir: str,
    file_name: str,
) -> dict:
    """
    Invoke the cleaning LLM once and return graph-shaped keys for code + plan.

    Parameters
    ----------
    model
        LangChain chat model.
    user_instructions : str or None
        End-user cleaning instructions.
    supplemental_instructions : str or None
        Host-provided instructions.
    source_df : pd.DataFrame
        Data used only for summary statistics in the prompt.
    function_name : str
        Generated function name.
    log : bool
        Whether to write code to disk.
    log_dir : str
        Directory for log files when ``log`` is True.
    file_name : str
        File name when ``log`` is True.

    Returns
    -------
    dict
        Keys: ``data_cleaner_function``, ``cleaning_plan``, ``data_cleaner_function_path``,
        ``data_cleaner_function_name``.
    """
    summary = get_dataframe_summary(source_df)
    dataset_summary = format_dataframe_summary(summary)
    chain = _data_cleaning_generation_chain(model)
    llm_out = chain.invoke({
        "user_instructions": user_instructions or "Follow the basic cleaning steps.",
        "supplemental_instructions": supplemental_instructions or "(none)",
        "all_datasets_summary": dataset_summary,
        "function_name": function_name,
    })
    code = llm_out["code"]
    plan = llm_out["cleaning_plan"]
    if plan is not None:
        plan = sanitize_cleaning_plan(plan, source_df)
    file_path = None
    if log:
        file_path = os.path.join(log_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        logger.info("Code saved to: %s", file_path)
    return {
        "data_cleaner_function": code,
        "cleaning_plan": plan,
        "data_cleaner_function_path": file_path,
        "data_cleaner_function_name": function_name,
    }


# State schema for the workflow graph. total=False because nodes incrementally
# populate keys (data_cleaner_function, data_cleaned, data_cleaner_error, ...)
# during the run rather than requiring all keys up front.
class GraphState(TypedDict, total=False):
    user_instructions: Required[str | None]
    supplemental_instructions: str | None
    source_df: Required[dict]
    max_retries: Required[int]
    retry_count: Required[int]
    data_cleaned: dict
    data_cleaner_function: str
    data_cleaner_function_path: str
    data_cleaner_function_name: str
    data_cleaner_error: str
    cleaning_plan: dict | None


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
        checkpointer: Checkpointer = None,
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
            checkpointer=checkpointer,
        )

    def invoke_agent(
        self,
        source_df: pd.DataFrame,
        user_instructions: str | None = None,
        supplemental_instructions: str | None = None,
        max_retries: int = 3,
        retry_count: int = 0,
        config: RunnableConfig | None = None,
    ) -> None:
        """
        Generate and execute data cleaning code on the provided DataFrame.

        Parameters
        ----------
        source_df : pd.DataFrame
            Raw dataset to clean.
        user_instructions : str, optional
            Free-form cleaning instructions from the end user. Columns named
            here are treated as protected and exempt from drops and destructive
            transforms. When None, the agent applies its full default pipeline.
            The pipeline is defined in ``data_cleaning_agent/prompts/data_cleaning.md``.
        supplemental_instructions : str, optional
            Additional instructions injected by the application (for example
            stable row identifiers). Shown to the model in a separate prompt
            section from ``user_instructions``; columns named here are protected
            the same way as in User Instructions.
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
            "supplemental_instructions": supplemental_instructions,
            "source_df": source_df.to_dict(),
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

    def get_input_dataframe(self):
        """
        Retrieves the input (pre-cleaning) dataframe from the last response.
        """
        if self.response:
            return pd.DataFrame(self.response.get("source_df"))

    def get_data_cleaner_function(self):
        """
        Retrieves the agent's cleaning function code.
        """
        if self.response:
            return self.response.get("data_cleaner_function")

    def get_cleaning_plan(self):
        """
        Return the structured cleaning plan dict from the last generation or fix.

        Returns None if no plan was parsed or ``invoke_agent`` / ``generate_cleaning_code``
        has not been run yet.
        """
        if self.response:
            return self.response.get("cleaning_plan")

    def generate_cleaning_code(
        self,
        source_df: pd.DataFrame,
        user_instructions: str | None = None,
        supplemental_instructions: str | None = None,
    ) -> None:
        """
        Run the LLM once to produce cleaning code and a structured plan (no execution).

        Stores partial results on ``self.response`` (code, plan, function name). Call
        :meth:`execute_stored_cleaning` when the host app is ready to run the code.
        """
        log_dir = self.log_path if self.log_path is not None else LOG_PATH
        if self.log and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        gen = _run_data_cleaning_generation(
            self.model,
            user_instructions=user_instructions,
            supplemental_instructions=supplemental_instructions,
            source_df=source_df,
            function_name=self.function_name,
            log=self.log,
            log_dir=log_dir,
            file_name=self.file_name,
        )
        self.response = {
            **gen,
            "retry_count": 0,
        }

    def execute_stored_cleaning(self, source_df: pd.DataFrame) -> dict:
        """
        Execute the code from the last ``generate_cleaning_code`` (or graph output).

        Returns
        -------
        dict
            Keys ``data_cleaned`` and ``data_cleaner_error`` as returned by
            :func:`execute_agent_code`.
        """
        if not self.response or not self.response.get("data_cleaner_function"):
            msg = "Call generate_cleaning_code (or invoke_agent) before execute_stored_cleaning."
            raise ValueError(msg)
        fn = self.response.get("data_cleaner_function_name") or self.function_name
        state = {
            "source_df": source_df.to_dict(),
            "data_cleaner_function": self.response["data_cleaner_function"],
            "data_cleaner_function_name": fn,
        }
        exec_out = execute_agent_code(
            state=state,
            data_key="source_df",
            result_key="data_cleaned",
            error_key="data_cleaner_error",
            code_snippet_key="data_cleaner_function",
            agent_function_name=fn,
        )
        self.response = {
            **self.response,
            **exec_out,
            "source_df": state["source_df"],
            "data_cleaner_function_name": fn,
        }
        return exec_out


# Agent Factory Function


def make_lightweight_data_cleaning_agent(
    model,
    log=False,
    log_path=None,
    file_name="data_cleaner.py",
    function_name="data_cleaner",
    checkpointer: Checkpointer = None,
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

        source_df = state["source_df"]
        df = pd.DataFrame.from_dict(source_df)

        return _run_data_cleaning_generation(
            model,
            user_instructions=state.get("user_instructions"),
            supplemental_instructions=state.get("supplemental_instructions"),
            source_df=df,
            function_name=function_name,
            log=log,
            log_dir=log_dir,
            file_name=file_name,
        )

    def execute_data_cleaner_code(state):
        """
        Execute the generated cleaning code on the data.
        """
        return execute_agent_code(
            state=state,
            data_key="source_df",
            result_key="data_cleaned",
            error_key="data_cleaner_error",
            code_snippet_key="data_cleaner_function",
            agent_function_name=state.get("data_cleaner_function_name"),
        )

    def fix_data_cleaner_code(state: GraphState):
        """
        Fix errors in the generated data cleaning code.
        """
        out = fix_agent_code(
            state=state,
            code_snippet_key="data_cleaner_function",
            error_key="data_cleaner_error",
            llm=model,
            prompt_template=_FIX_DATA_CLEANER_PROMPT_TEMPLATE,
            function_name=state.get("data_cleaner_function_name"),
            output_parser=DataCleaningOutputParser(),
        )
        raw = state.get("source_df")
        if out.get("cleaning_plan") is not None and raw is not None:
            out = {
                **out,
                "cleaning_plan": sanitize_cleaning_plan(
                    out["cleaning_plan"],
                    pd.DataFrame.from_dict(raw),
                ),
            }
        return out

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
