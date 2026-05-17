"""Tests for cleaning-plan prompt rendering and JSON parsing."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict

import pytest
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda

import data_cleaning_agent.cleaning_plan as cleaning_plan
import data_cleaning_agent.pipeline_steps as pipeline_steps
import data_cleaning_agent.plan_generation as plan_generation

_ROW_ID_COL = "__agent_row_id__"
_JSON_FENCE_RE = re.compile(r"```json\s*\n.*?\n```", re.DOTALL)
_JSON_PARSER = JsonOutputParser()


def _json_blocks_in_prompt(prompt: str) -> list[dict]:
    return [
        _JSON_PARSER.parse(match.group(0)) for match in _JSON_FENCE_RE.finditer(prompt)
    ]


@pytest.mark.unit
def test_render_plan_prompt_embeds_example_and_runtime_inputs() -> None:
    example = cleaning_plan.CleaningPlan(
        protected_columns=[_ROW_ID_COL],
        coerce_datetime_columns=("signup_date",),
        coerce_numeric_columns=("income_str",),
        coerce_bool_columns=("is_active",),
    )
    rendered = plan_generation.render_plan_prompt(
        user_instructions="protect country",
        dataset_summary="Rows: 5",
        example_plan=example,
        row_id_col=_ROW_ID_COL,
    )

    (example_json,) = _json_blocks_in_prompt(rendered)
    assert example_json == json.loads(json.dumps(asdict(example)))

    expected_step_ids = ", ".join(
        step.value for step in pipeline_steps.PIPELINE_STEP_ORDER
    )
    assert expected_step_ids in rendered

    user_section = rendered.split("User Instructions:\n", 1)[1]
    assert user_section.split("Dataset Summary:")[0].strip() == "protect country"

    summary_section = rendered.split("Dataset Summary:\n", 1)[1]
    assert summary_section.split("Return **only**")[0].strip() == "Rows: 5"


@pytest.mark.unit
def test_parse_cleaning_plan_json_valid() -> None:
    raw = '{"skip_steps": [], "protected_columns": ["country"]}'
    plan = plan_generation.parse_cleaning_plan_json(raw)
    assert "country" in plan.protected_columns


@pytest.mark.unit
def test_parse_cleaning_plan_json_strips_fenced_block() -> None:
    raw = 'Sure.\n```json\n{"skip_steps": ["impute"], "protected_columns": ["a"]}\n```'
    plan = plan_generation.parse_cleaning_plan_json(raw)
    assert plan.skip_steps == ["impute"]
    assert "a" in plan.protected_columns


@pytest.mark.unit
def test_parse_cleaning_plan_json_rejects_unknown_skip() -> None:
    with pytest.raises(ValueError, match="unknown skip_steps"):
        plan_generation.parse_cleaning_plan_json('{"skip_steps": ["not_a_step"]}')


@pytest.mark.unit
def test_validate_cleaning_plan_requires_row_id(summary) -> None:
    plan = cleaning_plan.CleaningPlan(protected_columns=["country"])
    with pytest.raises(ValueError, match="protected_columns must include row id"):
        plan_generation.validate_cleaning_plan(
            plan,
            summary,
            row_id_col=_ROW_ID_COL,
        )


@pytest.mark.unit
def test_validate_cleaning_plan_warns_on_missing_coerce(summary, caplog) -> None:
    plan = cleaning_plan.CleaningPlan(
        protected_columns=[_ROW_ID_COL],
        coerce_datetime_columns=(),
        coerce_numeric_columns=(),
        coerce_bool_columns=(),
    )
    with caplog.at_level(logging.WARNING):
        plan_generation.validate_cleaning_plan(
            plan,
            summary,
            row_id_col=_ROW_ID_COL,
        )
    assert "coerce_datetime_columns" in caplog.text
    assert "signup_date" in caplog.text


@pytest.mark.unit
def test_generate_cleaning_plan_uses_mock_llm(mixed_df, summary) -> None:
    example = cleaning_plan.default_plan_from_summary(summary, row_id_col=_ROW_ID_COL)
    payload = {
        "skip_steps": list(example.skip_steps),
        "protected_columns": [_ROW_ID_COL, "country"],
        "drop_high_missing_threshold": example.drop_high_missing_threshold,
        "coerce_datetime_columns": list(example.coerce_datetime_columns),
        "coerce_numeric_columns": list(example.coerce_numeric_columns),
        "coerce_bool_columns": list(example.coerce_bool_columns),
        "impute_numeric_columns": list(example.impute_numeric_columns),
        "impute_categorical_columns": list(example.impute_categorical_columns),
    }

    fake_model = RunnableLambda(
        lambda _prompt: AIMessage(content=json.dumps(payload)),
    )

    plan = plan_generation.generate_cleaning_plan(
        fake_model,
        mixed_df,
        user_instructions="protect country",
    )
    assert "country" in plan.protected_columns
    assert _ROW_ID_COL in plan.protected_columns
    assert "signup_date" in set(plan.coerce_datetime_columns)
