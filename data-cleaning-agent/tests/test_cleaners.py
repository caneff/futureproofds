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
def test_normalize_column_label() -> None:
    assert cleaners.normalize_column_label("Customer Name") == "customer_name"


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


@pytest.mark.unit
def test_coerce_datetime_columns_parses_strings_and_skips_missing_labels() -> None:
    df = pd.DataFrame({
        "d": ["2020-01-01", "not a date", "2021-12-31"],
        "x": [1, 2, 3],
    })
    out = cleaners.coerce_datetime_columns(df, ["d", "ghost"])
    assert pd.api.types.is_datetime64_any_dtype(out["d"])
    assert pd.isna(out["d"].iloc[1])
    assert not pd.isna(out["d"].iloc[0])
    assert out["x"].tolist() == [1, 2, 3]
    assert df["d"].dtype == object


@pytest.mark.unit
def test_coerce_numeric_columns_strips_currency_and_skips_non_targets() -> None:
    df = pd.DataFrame({
        "money": ["$1,234.5%", np.nan, "42"],
        "n": [1, 2, 3],
        "s": ["a", "b", "c"],
    })
    out = cleaners.coerce_numeric_columns(df, ["money", "n", "ghost"])
    assert out["money"].iloc[0] == pytest.approx(1234.5)
    assert pd.isna(out["money"].iloc[1])
    assert out["money"].iloc[2] == pytest.approx(42.0)
    assert out["n"].tolist() == [1, 2, 3]
    assert out["s"].tolist() == ["a", "b", "c"]


@pytest.mark.unit
def test_coerce_numeric_columns_leaves_datetime_column_unchanged() -> None:
    df = pd.DataFrame({"dt": pd.to_datetime(["2020-01-01", "2020-06-15"])})
    out = cleaners.coerce_numeric_columns(df, ["dt"])
    pd.testing.assert_series_equal(out["dt"], df["dt"], check_dtype=True)


@pytest.mark.unit
def test_coerce_bool_columns_maps_tokens_and_unknown_to_na() -> None:
    df = pd.DataFrame({
        "b": ["Yes", "NO", " t ", "maybe", np.nan],
        "x": [1, 2, 3, 4, 5],
    })
    out = cleaners.coerce_bool_columns(df, ["b", "ghost"])
    expected_b = pd.Series(
        [True, False, True, pd.NA, pd.NA],
        dtype="boolean",
        name="b",
    )
    pd.testing.assert_series_equal(out["b"], expected_b, check_dtype=True)
    assert out["x"].tolist() == [1, 2, 3, 4, 5]


@pytest.mark.unit
@pytest.mark.parametrize(
    "df,exclude,expected_columns",
    [
        pytest.param(
            pd.DataFrame(
                {"c": [7, 7, 7], "v": [1, 2, 1], "n": [1.0, np.nan, 1.0]},
            ),
            (),
            ["v"],
            id="single_non_null_level",
        ),
        pytest.param(
            pd.DataFrame({"keep": [1, 1], "drop": [2, 2]}),
            ("keep", "ghost"),
            ["keep"],
            id="exclude_keeps_constant_unknown_exclude_ignored",
        ),
    ],
)
def test_drop_constant_columns(
    df: pd.DataFrame,
    exclude: tuple[str, ...],
    expected_columns: list[str],
) -> None:
    out = cleaners.drop_constant_columns(df, exclude=exclude)
    assert list(out.columns) == expected_columns


@pytest.mark.unit
@pytest.mark.parametrize(
    "df,exclude,expected_columns,expect_copy_not_identity",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "all_nan": [np.nan, np.nan],
                    "ok": [1.0, 2.0],
                    "prot": [np.nan, np.nan],
                },
            ),
            ("prot",),
            ["ok", "prot"],
            False,
            id="drops_all_nan_respects_exclude",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "a": pd.Series([], dtype=float),
                    "b": pd.Series([], dtype=float),
                },
            ),
            (),
            ["a", "b"],
            True,
            id="zero_rows_returns_copy_columns_unchanged",
        ),
    ],
)
def test_drop_all_null_columns(
    df: pd.DataFrame,
    exclude: tuple[str, ...],
    expected_columns: list[str],
    expect_copy_not_identity: bool,
) -> None:
    out = cleaners.drop_all_null_columns(df, exclude=exclude)
    assert list(out.columns) == expected_columns
    if expect_copy_not_identity:
        assert out is not df


@pytest.mark.unit
def test_impute_numeric_median_or_mean_skewed_uses_median() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0, np.nan])
    out = cleaners.impute_numeric_median_or_mean(s)
    assert out.isna().sum() == 0
    assert out.iloc[-1] == pytest.approx(3.0)


@pytest.mark.unit
def test_impute_numeric_median_or_mean_symmetric_uses_mean() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
    out = cleaners.impute_numeric_median_or_mean(s)
    assert out.iloc[-1] == pytest.approx(3.0)


@pytest.mark.unit
def test_impute_numeric_median_or_mean_non_numeric_unchanged() -> None:
    s = pd.Series(["a", np.nan, "b"])
    out = cleaners.impute_numeric_median_or_mean(s)
    pd.testing.assert_series_equal(out, s, check_dtype=True)


@pytest.mark.unit
def test_impute_categorical_mode_fills_with_first_mode() -> None:
    s = pd.Series(["a", "b", "a", np.nan], dtype=object)
    out = cleaners.impute_categorical_mode(s)
    assert out.tolist() == ["a", "b", "a", "a"]


@pytest.mark.unit
def test_impute_categorical_mode_tie_breaks_sorted_first() -> None:
    s = pd.Series(["x", "x", "y", "y", np.nan], dtype=object)
    out = cleaners.impute_categorical_mode(s)
    assert out.iloc[-1] == "x"


@pytest.mark.unit
def test_impute_categorical_mode_all_na_returns_copy() -> None:
    s = pd.Series([np.nan, np.nan, np.nan], dtype=object)
    out = cleaners.impute_categorical_mode(s)
    assert out.isna().all()
    assert out is not s


@pytest.mark.unit
def test_impute_categorical_mode_non_string_like_unchanged() -> None:
    s = pd.Series([1.0, np.nan, 3.0])
    out = cleaners.impute_categorical_mode(s)
    pd.testing.assert_series_equal(out, s, check_dtype=True)


@pytest.mark.unit
@pytest.mark.parametrize(
    "df,expected_len,expect_copy_not_identity",
    [
        pytest.param(
            pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, np.nan]}),
            1,
            False,
            id="drops_all_null_row",
        ),
        pytest.param(
            pd.DataFrame({"x": pd.Series([], dtype=float)}),
            0,
            True,
            id="empty_frame_returns_copy",
        ),
    ],
)
def test_drop_all_null_rows(
    df: pd.DataFrame,
    expected_len: int,
    expect_copy_not_identity: bool,
) -> None:
    out = cleaners.drop_all_null_rows(df)
    assert len(out) == expected_len
    if expect_copy_not_identity:
        assert out is not df


@pytest.mark.unit
@pytest.mark.parametrize(
    "df,subset,expected_len",
    [
        pytest.param(
            pd.DataFrame({"a": [1, 1], "b": [2, 2]}),
            None,
            1,
            id="dedupe_all_columns",
        ),
        pytest.param(
            pd.DataFrame({"id": [1, 1], "value": [10, 20]}),
            ("id",),
            1,
            id="subset_id_dedupes_same_id",
        ),
        pytest.param(
            pd.DataFrame({"id": [1, 1], "value": [10, 20]}),
            None,
            2,
            id="all_columns_keeps_rows_with_different_values",
        ),
        pytest.param(
            pd.DataFrame({"a": [1, 2]}),
            ("ghost",),
            2,
            id="unknown_subset_unchanged",
        ),
    ],
)
def test_drop_duplicate_rows(
    df: pd.DataFrame,
    subset: tuple[str, ...] | None,
    expected_len: int,
) -> None:
    out = cleaners.drop_duplicate_rows(df, subset=subset)
    assert len(out) == expected_len
