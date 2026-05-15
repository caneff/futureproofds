import numpy as np
import pandas as pd
import pytest
from data_cleaning_agent import cleaners


@pytest.mark.unit
@pytest.mark.parametrize(
    "data,expected_columns",
    [
        pytest.param(
            {"Customer Name": [1]},
            ["customer_name"],
            id="mixed_case_and_spaces_to_snake_case",
        ),
        pytest.param(
            {"Foo!!!Bar": [1]},
            ["foo_bar"],
            id="punctuation_runs_collapsed",
        ),
        pytest.param({0: [1, 2]}, ["0"], id="integer_column_label_as_string"),
        pytest.param({"École": [1]}, ["école"], id="unicode_letters_preserved"),
        pytest.param(
            {"X": [1], " x ": [2]},
            ["x", "x"],
            id="distinct_originals_duplicate_normalized_names",
        ),
        pytest.param(
            {"a___b": [1]},
            ["a_b"],
            id="literal_underscore_runs_collapsed",
        ),
    ],
)
def test_normalize_column_names(data: dict, expected_columns: list[str]) -> None:
    df = pd.DataFrame(data)
    original_id = id(df)
    original_cols = list(df.columns)
    out = cleaners.normalize_column_names(df)
    assert id(df) == original_id
    assert list(df.columns) == original_cols
    assert list(out.columns) == expected_columns


@pytest.mark.unit
@pytest.mark.parametrize(
    "data,cols,treat_blank,expected",
    [
        pytest.param(
            {"x": [1.0, np.nan, 3.0, np.nan]},
            None,
            True,
            pd.Series({"x": 0.5}),
            id="half_nan_numeric",
        ),
        pytest.param(
            {"s": ["a", "N/A", "c", "d"]},
            None,
            True,
            pd.Series({"s": 0.25}),
            id="object_column_placeholder_token",
        ),
        pytest.param(
            {"s": ["a", "b", "   ", "d"]},
            None,
            False,
            pd.Series({"s": 0.0}),
            id="whitespace_only_not_missing_when_no_treat_blank",
        ),
        pytest.param(
            {"s": ["a", "b", "   ", "d"]},
            None,
            True,
            pd.Series({"s": 0.25}),
            id="whitespace_only_counts_when_treat_blank",
        ),
        pytest.param(
            {"a": [1.0, np.nan], "b": [np.nan, np.nan]},
            ["b", "a"],
            True,
            pd.Series({"b": 1.0, "a": 0.5}),
            id="cols_subset_preserves_parameter_order",
        ),
        pytest.param(
            {"x": pd.Series([], dtype=object)},
            None,
            True,
            pd.Series({"x": float("nan")}),
            id="zero_rows_all_nan_fractions",
        ),
    ],
)
def test_missing_share(
    data: dict,
    cols: list[str] | None,
    treat_blank: bool,
    expected: pd.Series,
) -> None:
    df = pd.DataFrame(data)
    result = cleaners.missing_share(df, cols=cols, treat_blank_as_missing=treat_blank)
    pd.testing.assert_series_equal(
        result, expected, check_names=True, check_dtype=False
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "data,threshold,exclude,expected_columns",
    [
        pytest.param(
            {"hi": [np.nan, np.nan, np.nan, 1.0], "lo": [1, 2, 3, 4]},
            0.4,
            (),
            ["lo"],
            id="drops_column_strictly_above_threshold",
        ),
        pytest.param(
            {"x": [1.0, np.nan, 3.0, np.nan], "y": [1, 2, 3, 4]},
            0.5,
            (),
            ["y"],
            id="drops_column_at_threshold_inclusive",
        ),
        pytest.param(
            {"prot": [np.nan] * 5, "ok": [0, 1, 2, 3, 4]},
            0.4,
            ("prot",),
            ["prot", "ok"],
            id="exclude_protects_high_missing_column",
        ),
        pytest.param(
            {"prot": [np.nan] * 5, "ok": [0, 1, 2, 3, 4]},
            0.4,
            ("prot", "ghost_col"),
            ["prot", "ok"],
            id="exclude_unknown_label_ignored_no_keyerror",
        ),
        pytest.param(
            {"lo": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]},
            0.4,
            (),
            ["lo"],
            id="keeps_column_share_strictly_below_threshold",
        ),
        pytest.param(
            {"a": [1, 2, 3], "b": [4, 5, 6]},
            0.99,
            (),
            ["a", "b"],
            id="no_op_when_all_shares_below_threshold",
        ),
    ],
)
def test_drop_columns_by_missing(
    data: dict,
    threshold: float,
    exclude: tuple[str, ...],
    expected_columns: list[str],
) -> None:
    df = pd.DataFrame(data)
    out = cleaners.drop_columns_by_missing(df, threshold, exclude=exclude)
    assert list(out.columns) == expected_columns


@pytest.mark.unit
def test_drop_columns_by_missing_rejects_threshold_out_of_range() -> None:
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="threshold"):
        cleaners.drop_columns_by_missing(df, -0.01)
    with pytest.raises(ValueError, match="threshold"):
        cleaners.drop_columns_by_missing(df, 1.01)


@pytest.mark.unit
@pytest.mark.parametrize(
    "data,cols,exclude,expected",
    [
        pytest.param(
            {"s": ["  a  ", "b"], "t": ["  Sales  ", "marketing"]},
            None,
            (),
            {"s": ["a", "b"], "t": ["Sales", "marketing"]},
            id="object_columns_strip_and_preserve_case",
        ),
        pytest.param(
            {"raw": ["  keep  "], "other": [" z "]},
            None,
            ("raw",),
            {"raw": ["  keep  "], "other": ["z"]},
            id="exclude_column_not_stripped",
        ),
        pytest.param(
            {"a": [" x "], "b": [" y "]},
            ["a"],
            (),
            {"a": ["x"], "b": [" y "]},
            id="cols_subset_only_strips_listed_string_column",
        ),
        pytest.param(
            {"n": [1, 2], "s": [" a ", " b "]},
            None,
            (),
            {"n": [1, 2], "s": ["a", "b"]},
            id="numeric_column_unchanged",
        ),
        pytest.param(
            {"s": pd.Series(["  u  "], dtype="string")},
            None,
            (),
            {"s": ["u"]},
            id="nullable_string_dtype_stripped",
        ),
    ],
)
def test_strip_strings(
    data: dict,
    cols: list[str] | None,
    exclude: tuple[str, ...],
    expected: dict,
) -> None:
    df = pd.DataFrame(data)
    out = cleaners.strip_strings(df, cols=cols, exclude=exclude)
    expected_df = pd.DataFrame(expected)
    pd.testing.assert_frame_equal(
        out[sorted(expected_df.columns)],
        expected_df[sorted(expected_df.columns)],
        check_dtype=False,
        check_column_type=False,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "data,cols,placeholders_spec,expected",
    [
        pytest.param(
            {"s": ["  N/A  ", "ok", "  null  "], "n": [1, 2, 3]},
            None,
            None,
            {"s": [np.nan, "ok", np.nan], "n": [1, 2, 3]},
            id="defaults_strip_and_replace_numeric_unchanged",
        ),
        pytest.param(
            {"s": ["", "   ", "x"]},
            None,
            None,
            {"s": [np.nan, np.nan, "x"]},
            id="empty_and_whitespace_only_become_na",
        ),
        pytest.param(
            {"a": ["N/A"], "b": ["N/A"]},
            ["a"],
            None,
            {"a": [np.nan], "b": ["N/A"]},
            id="cols_subset_only_scored_column_replaced",
        ),
        pytest.param(
            {"s": ["  TBD  ", "N/A"]},
            None,
            ("TBD",),
            {"s": [np.nan, "N/A"]},
            id="custom_placeholder_only",
        ),
        pytest.param(
            {"s": ["N/A"]},
            None,
            (),
            {"s": ["N/A"]},
            id="empty_placeholder_iterable_no_op",
        ),
    ],
)
def test_replace_placeholders_with_na(
    data: dict,
    cols: list[str] | None,
    placeholders_spec: tuple[str, ...] | None,
    expected: dict,
) -> None:
    df = pd.DataFrame(data)
    out = cleaners.replace_placeholders_with_na(
        df, placeholders=placeholders_spec, cols=cols
    )

    expected_df = pd.DataFrame(expected)
    pd.testing.assert_frame_equal(
        out[sorted(expected_df.columns)],
        expected_df[sorted(expected_df.columns)],
        check_dtype=False,
        check_column_type=False,
    )
