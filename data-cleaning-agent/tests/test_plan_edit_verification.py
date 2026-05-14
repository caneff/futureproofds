"""Tests for plan-edit regeneration verification (scope B)."""

import pandas as pd
import pytest
from data_cleaning_agent.plan_edit_verification import (
    classify_removed_action,
    columns_where_missingness_dropped_without_plan_imputation,
    columns_where_retain_missing_plan_violated_by_execution,
    compose_host_pre_apply_blocked_message,
    compose_plan_regen_supplemental,
    format_retain_violation_host_fix_message,
    format_verification_feedback_markdown,
    plan_column_lists_imputation_action,
    plan_column_lists_retain_missing_action,
    resolve_plan_column_name,
    verify_removed_plan_steps,
)
from preview_helpers import AGENT_ROW_ID


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("impute missing values (median)", "impute"),
        ("Fill NaNs with mode", "impute"),
        ("forward fill gaps", "impute"),
        ("bfill then mean", "impute"),
        ("drop column (>40% missing)", "drop"),
        ("Remove column X entirely", "drop"),
        ("Delete column foo", "drop"),
        ("astype to float64", "dtype"),
        ("Coerce to datetime", "dtype"),
        ("dtype conversion to int", "dtype"),
        ("strip whitespace only", None),
        ("normalize labels", None),
        ("drop duplicate rows", None),
        ("retain missing values", None),
        ("Retain Missing Values", None),
    ],
)
def test_classify_removed_action(action: str, expected: CheckKind | None) -> None:
    assert classify_removed_action(action) == expected


def test_resolve_plan_column_name_case_and_space() -> None:
    df = pd.DataFrame({"My Col": [1, 2], "b": [3, None]})
    assert resolve_plan_column_name("my col", df) == "My Col"
    assert resolve_plan_column_name("B", df) == "b"


def test_compose_plan_regen_supplemental_order() -> None:
    out = compose_plan_regen_supplemental(
        "BASE",
        "EXCL",
        prior_verification_feedback="PRIOR",
        automatic_retry_failure_block="RETRY",
        user_follow_up="USER",
    )
    bi = out.index("BASE")
    ei = out.index("EXCL")
    pi = out.index("PRIOR")
    ri = out.index("RETRY")
    ui = out.index("USER")
    assert bi < ei < pi < ri < ui


def test_compose_plan_regen_supplemental_minimal() -> None:
    out = compose_plan_regen_supplemental("BASE", "EXCL")
    assert "BASE" in out and "EXCL" in out
    assert "PRIOR" not in out


def test_verify_impute_passes_when_nulls_non_decreasing() -> None:
    rid = AGENT_ROW_ID
    df_b = pd.DataFrame({rid: [0, 1, 2], "x": [1.0, None, 3.0]})
    df_a = pd.DataFrame({rid: [0, 1, 2], "x": [1.0, None, 3.0]})
    removed = [("x", "impute with median")]
    r = verify_removed_plan_steps(removed, df_b, df_a, row_id_col=rid)
    assert r.ok
    assert not r.unclassified_removed


def test_verify_impute_fails_when_nulls_decrease() -> None:
    rid = AGENT_ROW_ID
    df_b = pd.DataFrame({rid: [0, 1], "x": [None, 2.0]})
    df_a = pd.DataFrame({rid: [0, 1], "x": [1.0, 2.0]})
    removed = [("x", "impute missing (mean)")]
    r = verify_removed_plan_steps(removed, df_b, df_a, row_id_col=rid)
    assert not r.ok
    assert r.classified_failures[0].check_type == "impute"


def test_verify_drop_removed_column_must_exist() -> None:
    rid = AGENT_ROW_ID
    df_b = pd.DataFrame({rid: [0], "keep_me": [1]})
    df_a = pd.DataFrame({rid: [0]})
    removed = [("keep_me", "drop column")]
    r = verify_removed_plan_steps(removed, df_b, df_a, row_id_col=rid)
    assert not r.ok
    assert r.classified_failures[0].check_type == "drop"


def test_verify_dtype_fails_on_change() -> None:
    rid = AGENT_ROW_ID
    df_b = pd.DataFrame({rid: [0, 1], "z": ["1", "2"]})
    df_a = pd.DataFrame({rid: [0, 1], "z": [1, 2]})
    removed = [("z", "coerce to numeric safely")]
    r = verify_removed_plan_steps(removed, df_b, df_a, row_id_col=rid)
    assert not r.ok
    assert r.classified_failures[0].check_type == "dtype"


def test_verify_unclassified_lenient() -> None:
    rid = AGENT_ROW_ID
    df_b = pd.DataFrame({rid: [0], "a": [1]})
    df_a = pd.DataFrame({rid: [0], "a": [2]})
    removed = [("a", "custom obscure transform")]
    r = verify_removed_plan_steps(removed, df_b, df_a, row_id_col=rid)
    assert r.ok
    assert r.unclassified_removed == [("a", "custom obscure transform")]


def test_verify_mixed_fail_and_unclassified() -> None:
    rid = AGENT_ROW_ID
    df_b = pd.DataFrame({rid: [0, 1], "x": [None, 1.0], "y": [1, 2]})
    df_a = pd.DataFrame({rid: [0, 1], "x": [0.0, 1.0], "y": [1, 2]})
    removed = [
        ("x", "fill missing values"),
        ("y", "obscure step"),
    ]
    r = verify_removed_plan_steps(removed, df_b, df_a, row_id_col=rid)
    assert not r.ok
    assert len(r.classified_failures) == 1
    assert len(r.unclassified_removed) == 1


def test_format_verification_feedback_markdown() -> None:
    from data_cleaning_agent.plan_edit_verification import (
        ClassifiedFailure,
        VerificationResult,
    )

    vr = VerificationResult(
        ok=False,
        classified_failures=[
            ClassifiedFailure("c", "dtype", "dtype changed"),
        ],
    )
    md = format_verification_feedback_markdown(vr)
    assert "failed automated verification" in md
    assert "dtype" in md
    assert "dtype changed" in md
    assert "Host requirements" in md
    assert "Dtype check" in md
    assert "`c`" in md


def test_format_verification_feedback_imputation_opinionated() -> None:
    from data_cleaning_agent.plan_edit_verification import (
        ClassifiedFailure,
        VerificationResult,
    )

    vr = VerificationResult(
        ok=False,
        classified_failures=[
            ClassifiedFailure(
                "city",
                "impute",
                "nulls decreased",
            ),
            ClassifiedFailure(
                "experience",
                "impute",
                "nulls decreased",
            ),
        ],
    )
    md = format_verification_feedback_markdown(vr)
    assert "**DO NOT** impute" in md
    assert "`city`" in md
    assert "`experience`" in md
    assert "fillna" in md.lower()
    assert "step 9" in md.lower()


def test_compose_host_pre_apply_blocked_message_ghost_and_retain() -> None:
    msg = compose_host_pre_apply_blocked_message(
        ["salary", "age"],
        ["city"],
    )
    assert "salary" in msg and "age" in msg
    assert "city" in msg
    assert "imputation" in msg.lower()
    assert "retain" in msg.lower()


def test_format_retain_violation_host_fix_message() -> None:
    msg = format_retain_violation_host_fix_message(["city", "town"])
    assert "`city`" in msg
    assert "`town`" in msg
    assert "Host verification" in msg


def test_plan_column_lists_imputation_false_for_retain_only() -> None:
    plan = {
        "columns": [
            {"name": "city", "actions": ["normalize name", "retain missing values"]}
        ]
    }
    assert plan_column_lists_imputation_action(plan, "city") is False


def test_plan_column_lists_retain_missing_action() -> None:
    plan = {
        "columns": [
            {"name": "city", "actions": ["normalize name", "retain missing values"]}
        ]
    }
    assert plan_column_lists_retain_missing_action(plan, "city") is True
    assert plan_column_lists_retain_missing_action(plan, "zip") is False


def test_columns_where_missingness_no_flag_when_plan_lists_retain() -> None:
    """Ghost imputation-omission list stays empty when only retain documents the column."""
    rid = AGENT_ROW_ID
    df_b = pd.DataFrame({rid: [0, 1], "city": [None, None]})
    df_a = pd.DataFrame({rid: [0, 1], "city": ["a", "b"]})
    plan = {"columns": [{"name": "city", "actions": ["retain missing values"]}]}
    assert plan_column_lists_imputation_action(plan, "city") is False
    assert (
        columns_where_missingness_dropped_without_plan_imputation(
            df_b, df_a, plan, row_id_col=rid
        )
        == []
    )


def test_columns_where_retain_missing_plan_violated_when_nulls_drop() -> None:
    rid = AGENT_ROW_ID
    df_b = pd.DataFrame({rid: [0, 1], "city": [None, None]})
    df_a = pd.DataFrame({rid: [0, 1], "city": ["a", "b"]})
    plan = {"columns": [{"name": "city", "actions": ["retain missing values"]}]}
    assert columns_where_retain_missing_plan_violated_by_execution(
        df_b, df_a, plan, row_id_col=rid
    ) == ["city"]


def test_columns_where_retain_missing_plan_not_violated_when_impute_listed() -> None:
    rid = AGENT_ROW_ID
    df_b = pd.DataFrame({rid: [0, 1], "city": [None, None]})
    df_a = pd.DataFrame({rid: [0, 1], "city": ["a", "b"]})
    plan = {
        "columns": [
            {
                "name": "city",
                "actions": ["retain missing values", "impute missing values (median)"],
            }
        ]
    }
    assert (
        columns_where_retain_missing_plan_violated_by_execution(
            df_b, df_a, plan, row_id_col=rid
        )
        == []
    )


def test_columns_where_retain_missing_plan_not_violated_when_nulls_stable() -> None:
    rid = AGENT_ROW_ID
    df_b = pd.DataFrame({rid: [0, 1], "city": [None, "x"]})
    df_a = pd.DataFrame({rid: [0, 1], "city": [None, "x"]})
    plan = {"columns": [{"name": "city", "actions": ["retain missing values"]}]}
    assert (
        columns_where_retain_missing_plan_violated_by_execution(
            df_b, df_a, plan, row_id_col=rid
        )
        == []
    )


def test_plan_column_lists_imputation_action() -> None:
    p_strip = {"columns": [{"name": "city", "actions": ["strip whitespace"]}]}
    assert plan_column_lists_imputation_action(p_strip, "city") is False
    p_imp = {"columns": [{"name": "city", "actions": ["impute missing values (mode)"]}]}
    assert plan_column_lists_imputation_action(p_imp, "city") is True


def test_columns_where_missingness_dropped_without_plan_imputation() -> None:
    rid = AGENT_ROW_ID
    df_b = pd.DataFrame({rid: [0, 1], "city": [None, None]})
    df_a = pd.DataFrame({rid: [0, 1], "city": ["a", "b"]})
    plan = {"columns": [{"name": "city", "actions": ["strip whitespace"]}]}
    assert columns_where_missingness_dropped_without_plan_imputation(
        df_b, df_a, plan, row_id_col=rid
    ) == ["city"]


def test_columns_where_missingness_no_flag_when_plan_lists_impute() -> None:
    rid = AGENT_ROW_ID
    df_b = pd.DataFrame({rid: [0, 1], "city": [None, None]})
    df_a = pd.DataFrame({rid: [0, 1], "city": ["a", "b"]})
    plan = {
        "columns": [{"name": "city", "actions": ["impute missing values (median)"]}]
    }
    assert (
        columns_where_missingness_dropped_without_plan_imputation(
            df_b, df_a, plan, row_id_col=rid
        )
        == []
    )
