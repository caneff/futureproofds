import numpy as np
import pandas as pd
import pytest
from data_cleaning_agent.cleaners import missing_share, normalize_column_names


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
    out = normalize_column_names(df)
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
    result = missing_share(df, cols=cols, treat_blank_as_missing=treat_blank)
    pd.testing.assert_series_equal(
        result, expected, check_names=True, check_dtype=False
    )
