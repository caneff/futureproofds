import pytest

from data_cleaning_agent.cleaning_plan import CleaningPlan, default_plan_from_summary


@pytest.mark.unit
def test_default_plan_seeds_coerce_columns_from_summary(mixed_df, summary) -> None:
    plan = default_plan_from_summary(summary, row_id_col="__agent_row_id__")
    assert "signup_date" in plan.coerce_datetime_columns
    assert "income_str" in plan.coerce_numeric_columns
    assert "is_active" in plan.coerce_bool_columns


@pytest.mark.unit
def test_default_plan_protects_row_id_only(mixed_df, summary) -> None:
    plan = default_plan_from_summary(summary, row_id_col="__agent_row_id__")
    assert plan.protected_columns == ["__agent_row_id__"]


@pytest.mark.unit
def test_cleaning_plan_rejects_unknown_skip_step() -> None:
    with pytest.raises(ValueError, match="unknown skip_steps"):
        CleaningPlan(skip_steps=["not_a_step"])
