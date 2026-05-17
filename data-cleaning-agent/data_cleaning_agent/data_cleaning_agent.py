# Libraries

import logging
from dataclasses import asdict
from typing import Required, TypedDict

import langchain_core.runnables as langchain_runnables
import langgraph.graph as langgraph_graph
import langgraph.types as langgraph_types
import pandas as pd

import data_cleaning_agent.cleaning_pipeline as cleaning_pipeline
import data_cleaning_agent.cleaning_plan as cleaning_plan
import data_cleaning_agent.plan_generation as plan_generation

# Setup

logger = logging.getLogger(__name__)

AGENT_NAME = "lightweight_data_cleaning_agent"


# State schema for the workflow graph. total=False because nodes incrementally

# populate keys (cleaning_plan, data_cleaned, data_cleaner_error, ...)

# during the run rather than requiring all keys up front.


class GraphState(TypedDict, total=False):
    user_instructions: Required[str | None]

    source_df: Required[dict]

    max_retries: Required[int]

    retry_count: Required[int]

    data_cleaned: dict

    cleaning_plan: dict

    cleaning_plan_error: str

    data_cleaner_error: str


class LightweightDataCleaningAgent:
    """

    LLM-powered agent that produces a JSON cleaning plan and runs the hybrid pipeline.



    Uses an LLM to build a :class:`~data_cleaning_agent.cleaning_plan.CleaningPlan`

    from user instructions, then executes fixed-order steps via

    :func:`~data_cleaning_agent.cleaning_pipeline.run_cleaning_pipeline`. The graph

    retries with plan correction on validation or runtime errors.



    Prompt sources: ``data_cleaning_plan.md``, ``data_cleaning_plan_fix.md``.



    Parameters

    ----------

    model : LLM

        Language model for generating cleaning plans (e.g., ChatOpenAI).

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
        checkpointer: langgraph_types.Checkpointer = None,
    ):

        self.model = model

        self.checkpointer = checkpointer

        self.response = None

        self._compiled_graph = make_lightweight_data_cleaning_agent(
            model=model,
            checkpointer=checkpointer,
        )

    def invoke_agent(
        self,
        source_df: pd.DataFrame,
        user_instructions: str | None = None,
        max_retries: int = 3,
        retry_count: int = 0,
        config: langchain_runnables.RunnableConfig | None = None,
    ) -> None:
        """

        Generate a cleaning plan and run the hybrid pipeline on the provided DataFrame.



        Parameters

        ----------

        source_df : pd.DataFrame

            Raw dataset to clean (should include ``DEFAULT_ROW_ID_COL`` when used

            from the Streamlit app).

        user_instructions : str, optional

            Free-form cleaning instructions from the end user. Columns named

            here are treated as protected and exempt from drops and destructive

            transforms. When None, the agent applies its full default pipeline.

        max_retries : int, default=3

            Maximum number of retry attempts if plan execution fails.

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

    def get_cleaning_plan(self) -> cleaning_plan.CleaningPlan | None:
        """Return the parsed cleaning plan from the last run, if any."""

        plan_dict = self.response.get("cleaning_plan") if self.response else None

        if plan_dict is None:
            return None

        return cleaning_plan.CleaningPlan(**plan_dict)

    def generate_cleaning_plan(
        self,
        source_df: pd.DataFrame,
        user_instructions: str | None = None,
    ) -> None:
        """

        Run the LLM once to produce a cleaning plan (no pipeline execution).



        Stores partial results on ``self.response``. Call

        :meth:`execute_stored_cleaning` when the host app is ready to apply the plan.

        """

        plan = plan_generation.generate_cleaning_plan(
            self.model,
            source_df,
            user_instructions,
            row_id_col=cleaning_plan.DEFAULT_ROW_ID_COL,
        )

        self.response = {
            "cleaning_plan": asdict(plan),
            "retry_count": 0,
        }

    def execute_stored_cleaning(self, source_df: pd.DataFrame) -> dict:
        """

        Run the hybrid pipeline from the last :meth:`generate_cleaning_plan` step.



        Returns

        -------

        dict

            Keys ``data_cleaned`` and ``data_cleaner_error``.

        """

        if not self.response:
            msg = (
                "Call generate_cleaning_plan or invoke_agent "
                "before execute_stored_cleaning."
            )

            raise ValueError(msg)

        plan_dict = self.response.get("cleaning_plan")

        if plan_dict is None:
            msg = (
                "Call generate_cleaning_plan or invoke_agent "
                "before execute_stored_cleaning."
            )

            raise ValueError(msg)

        plan = cleaning_plan.CleaningPlan(**plan_dict)

        try:
            cleaned, _trace = cleaning_pipeline.run_cleaning_pipeline(
                source_df,
                plan,
                row_id_col=cleaning_plan.DEFAULT_ROW_ID_COL,
            )

            exec_out = {
                "data_cleaned": cleaned.to_dict(),
                "data_cleaner_error": None,
            }

        except Exception as exc:
            exec_out = {"data_cleaner_error": str(exc)}

        self.response = {
            **self.response,
            **exec_out,
            "source_df": source_df.to_dict(),
        }

        return exec_out


# Agent Factory Function


def make_lightweight_data_cleaning_agent(
    model,
    checkpointer: langgraph_types.Checkpointer = None,
):
    """

    Factory function that creates a compiled LangGraph workflow for data cleaning.



    Builds a state graph with three nodes: plan generation, pipeline execution, and

    plan correction. The workflow automatically retries with corrections on failure.



    Parameters

    ----------

    model : LLM

        Language model for generating cleaning plans.

    checkpointer : Checkpointer, optional

        LangGraph checkpointer for saving workflow state.



    Returns

    -------

    CompiledStateGraph

        Compiled LangGraph workflow ready to process cleaning requests.

    """

    def create_cleaning_plan(state: GraphState):
        """Generate a validated cleaning plan from user instructions."""

        logger.info("Creating cleaning plan")

        df = pd.DataFrame.from_dict(state["source_df"])

        try:
            plan = plan_generation.generate_cleaning_plan(
                model,
                df,
                state.get("user_instructions"),
                row_id_col=cleaning_plan.DEFAULT_ROW_ID_COL,
            )

        except Exception as exc:
            logger.exception("Failed to generate cleaning plan")

            return {"cleaning_plan_error": str(exc)}

        return {
            "cleaning_plan": asdict(plan),
            "cleaning_plan_error": None,
        }

    def execute_cleaning_plan(state: GraphState):
        """Run the hybrid pipeline using the plan in state."""

        logger.info("Executing cleaning plan")

        if state.get("cleaning_plan_error"):
            return {"data_cleaner_error": state["cleaning_plan_error"]}

        plan_dict = state.get("cleaning_plan")

        if not plan_dict:
            return {"data_cleaner_error": "missing cleaning plan"}

        df = pd.DataFrame.from_dict(state["source_df"])

        plan = cleaning_plan.CleaningPlan(**plan_dict)

        try:
            cleaned, _trace = cleaning_pipeline.run_cleaning_pipeline(
                df,
                plan,
                row_id_col=cleaning_plan.DEFAULT_ROW_ID_COL,
            )

        except Exception as exc:
            logger.exception("Failed to execute cleaning plan")

            return {"data_cleaner_error": str(exc)}

        return {
            "data_cleaned": cleaned.to_dict(),
            "data_cleaner_error": None,
        }

    def fix_cleaning_plan(state: GraphState):
        """Fix a broken cleaning plan using the LLM."""

        logger.info("Fixing cleaning plan")

        df = pd.DataFrame.from_dict(state["source_df"])

        error = state.get("data_cleaner_error") or state.get("cleaning_plan_error")

        if not error:
            return {"retry_count": state.get("retry_count", 0) + 1}

        try:
            plan = plan_generation.fix_cleaning_plan(
                model,
                df,
                broken_plan=state.get("cleaning_plan"),
                error=error,
                user_instructions=state.get("user_instructions"),
                row_id_col=cleaning_plan.DEFAULT_ROW_ID_COL,
            )

        except Exception as exc:
            logger.exception("Failed to fix cleaning plan")

            return {
                "cleaning_plan_error": str(exc),
                "data_cleaner_error": str(exc),
                "retry_count": state.get("retry_count", 0) + 1,
            }

        return {
            "cleaning_plan": asdict(plan),
            "cleaning_plan_error": None,
            "data_cleaner_error": None,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    workflow = langgraph_graph.StateGraph(GraphState)

    workflow.add_node("create_cleaning_plan", create_cleaning_plan)

    workflow.add_node("execute_cleaning_plan", execute_cleaning_plan)

    workflow.add_node("fix_cleaning_plan", fix_cleaning_plan)

    workflow.set_entry_point("create_cleaning_plan")

    workflow.add_edge("create_cleaning_plan", "execute_cleaning_plan")

    workflow.add_edge("fix_cleaning_plan", "execute_cleaning_plan")

    def should_retry(state):

        has_error = state.get("data_cleaner_error") is not None

        can_retry = (
            state.get("retry_count") is not None
            and state.get("max_retries") is not None
            and state["retry_count"] < state["max_retries"]
        )

        return "fix_plan" if (has_error and can_retry) else "end"

    workflow.add_conditional_edges(
        "execute_cleaning_plan",
        should_retry,
        {
            "fix_plan": "fix_cleaning_plan",
            "end": langgraph_graph.END,
        },
    )

    return workflow.compile(checkpointer=checkpointer, name=AGENT_NAME)
