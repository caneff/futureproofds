"""Tests for LangGraph wiring and LightweightDataCleaningAgent plan path."""

from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from data_cleaning_agent.cleaning_plan import (
    DEFAULT_ROW_ID_COL,
    CleaningPlan,
    default_plan_from_summary,
)
from data_cleaning_agent.data_cleaning_agent import (
    LightweightDataCleaningAgent,
    make_lightweight_data_cleaning_agent,
)
from data_cleaning_agent.plan_generation import FIX_PLAN_PROMPT_TEMPLATE

_ROW_ID = DEFAULT_ROW_ID_COL


def _df_with_row_id(mixed_df: pd.DataFrame) -> pd.DataFrame:
    out = mixed_df.copy()
    out.insert(0, _ROW_ID, [str(i) for i in range(len(out))])
    return out


def _valid_plan_payload(summary) -> dict:
    example = default_plan_from_summary(summary, row_id_col=_ROW_ID)
    return {
        **asdict(example),
        "protected_columns": [_ROW_ID, "country"],
    }


@pytest.mark.unit
def test_fix_plan_prompt_formats_with_expected_placeholders() -> None:
    rendered = FIX_PLAN_PROMPT_TEMPLATE.format(
        user_instructions="protect country",
        all_datasets_summary="Rows: 5",
        pipeline_step_ids="normalize_names, impute",
        plan_snippet='{"skip_steps": []}',
        error="ValueError: unknown skip_steps",
        row_id_col=_ROW_ID,
    )
    assert "CleaningPlan" in rendered
    assert "```json" in rendered
    assert "unknown skip_steps" in rendered
    assert _ROW_ID in rendered


@pytest.mark.unit
def test_invoke_agent_runs_pipeline_with_mock_llm(
    mixed_df, summary, monkeypatch
) -> None:
    payload = _valid_plan_payload(summary)
    fake_model = RunnableLambda(
        lambda _prompt: AIMessage(content=json.dumps(payload)),
    )
    monkeypatch.setattr(
        "data_cleaning_agent.data_cleaning_agent.plan_generation.generate_cleaning_plan",
        lambda model, source_df, user_instructions=None, **kwargs: CleaningPlan(
            **payload
        ),
    )

    agent = LightweightDataCleaningAgent(model=fake_model)
    df = _df_with_row_id(mixed_df)
    agent.invoke_agent(df, user_instructions="protect country", max_retries=0)

    cleaned = agent.get_data_cleaned()
    assert cleaned is not None
    assert len(cleaned) == len(df)
    assert agent.response.get("data_cleaner_error") is None
    assert agent.response.get("cleaning_plan") is not None


@pytest.mark.unit
def test_generate_and_execute_stored_cleaning_plan(
    mixed_df, summary, monkeypatch
) -> None:
    payload = _valid_plan_payload(summary)
    fake_model = RunnableLambda(
        lambda _prompt: AIMessage(content=json.dumps(payload)),
    )
    monkeypatch.setattr(
        "data_cleaning_agent.data_cleaning_agent.plan_generation.generate_cleaning_plan",
        lambda model, source_df, user_instructions=None, **kwargs: CleaningPlan(
            **payload
        ),
    )

    agent = LightweightDataCleaningAgent(model=fake_model)
    df = _df_with_row_id(mixed_df)
    agent.generate_cleaning_plan(df, user_instructions="protect country")

    plan = agent.get_cleaning_plan()
    assert plan is not None
    assert "country" in plan.protected_columns

    out = agent.execute_stored_cleaning(df)
    assert out.get("data_cleaner_error") is None
    assert agent.get_data_cleaned() is not None


@pytest.mark.unit
def test_graph_retries_fix_on_pipeline_error(mixed_df, summary, monkeypatch) -> None:
    payload = _valid_plan_payload(summary)
    calls: list[str] = []

    def fake_generate(model, source_df, user_instructions=None, **kwargs):
        calls.append("generate")
        return CleaningPlan(**payload)

    def fake_repair(model, source_df, *, broken_plan, error, **kwargs):
        calls.append("repair")
        return CleaningPlan(**payload)

    execute_count = {"n": 0}

    def fake_run_cleaning_pipeline(df, plan, *, row_id_col=DEFAULT_ROW_ID_COL):
        execute_count["n"] += 1
        if execute_count["n"] == 1:
            msg = "simulated pipeline failure"
            raise RuntimeError(msg)
        from data_cleaning_agent.cleaning_pipeline import PipelineTrace

        return df.copy(), PipelineTrace()

    monkeypatch.setattr(
        "data_cleaning_agent.data_cleaning_agent.plan_generation.generate_cleaning_plan",
        fake_generate,
    )
    monkeypatch.setattr(
        "data_cleaning_agent.data_cleaning_agent.plan_generation.fix_cleaning_plan",
        fake_repair,
    )
    monkeypatch.setattr(
        "data_cleaning_agent.data_cleaning_agent.cleaning_pipeline.run_cleaning_pipeline",
        fake_run_cleaning_pipeline,
    )

    graph = make_lightweight_data_cleaning_agent(model=object())
    df = _df_with_row_id(mixed_df)
    result = graph.invoke({
        "user_instructions": "protect country",
        "source_df": df.to_dict(),
        "max_retries": 1,
        "retry_count": 0,
    })

    assert calls == ["generate", "repair"]
    assert result.get("data_cleaner_error") is None
    assert execute_count["n"] == 2
