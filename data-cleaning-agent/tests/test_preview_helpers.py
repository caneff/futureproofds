"""Unit tests for ``preview_helpers.preview_aligned_frames``."""

import pandas as pd
import pytest

from preview_helpers import (
    AGENT_ROW_ID,
    diff_cell_mask,
    preview_aligned_frames,
    reorder_cleaned_for_export,
    round_numeric_preview,
    style_preview_pair,
)


@pytest.mark.unit
def test_round_numeric_preview_coerces_all_numeric_object_column():
    df = pd.DataFrame({"x": ["1.234", "2.0", None]})
    out = round_numeric_preview(df, 2)
    assert out["x"].dtype == float or str(out["x"].dtype).startswith("float")
    assert abs(out["x"].iloc[0] - 1.23) < 1e-9
    assert pd.isna(out["x"].iloc[2])


@pytest.mark.unit
def test_round_numeric_preview_leaves_non_numeric():
    df = pd.DataFrame({"s": ["a"], "x": [1.2345], "y": [7]})
    out = round_numeric_preview(df, 2)
    assert out["s"].tolist() == ["a"]
    assert abs(out["x"].iloc[0] - 1.23) < 1e-9
    assert out["y"].iloc[0] == 7


@pytest.mark.unit
def test_diff_cell_mask_nan_safe():
    before = pd.DataFrame({"a": [1, 2], "b": [3.0, float("nan")]})
    after = pd.DataFrame({"a": [1, 9], "b": [3.0, float("nan")]})
    m = diff_cell_mask(before, after)
    assert m is not None
    assert m.iloc[0].tolist() == [False, False]
    assert m.iloc[1].tolist() == [True, False]


@pytest.mark.unit
def test_diff_cell_mask_no_na_in_result_for_nullable_string():
    before = pd.DataFrame({"a": pd.array(["x", None], dtype="string")})
    after = pd.DataFrame({"a": pd.array(["y", None], dtype="string")})
    mask = diff_cell_mask(before, after)
    assert mask is not None
    assert mask.dtypes.eq(bool).all()
    assert not mask.isna().any().any()


@pytest.mark.unit
def test_style_preview_pair_does_not_raise_on_nullable_na():
    before = pd.DataFrame({"a": pd.array(["x", None], dtype="string")})
    after = pd.DataFrame({"a": pd.array(["y", None], dtype="string")})
    style_preview_pair(before, after)


@pytest.mark.unit
def test_style_preview_pair_highlights_differing_cells():
    before = pd.DataFrame({"x": [1.0, 2.0]})
    after = pd.DataFrame({"x": [1.0, 3.0]})
    before_disp, _after_disp = style_preview_pair(before, after)
    html = before_disp.to_html()
    assert "rgba(255, 230, 120" in html


@pytest.mark.unit
def test_aligned_first_k_cleaned_ids_only_changed_rows():
    """Includes only differing rows; order tie-breaks on cleaned id order when counts tie."""
    rid = AGENT_ROW_ID
    raw = pd.DataFrame({rid: [0, 1, 2], "a": ["x", "y", "z"], "b": [1, 2, 3]})
    cleaned = pd.DataFrame({
        rid: [2, 0],
        "a": ["z", "x"],
        "b": [300, 99],
        "extra": [99, 100],
    })
    result = preview_aligned_frames(raw, cleaned, rid, k=2)
    assert result.aligned is True
    assert rid not in result.before_view.columns
    assert list(result.before_view["a"]) == ["z", "x"]
    assert list(result.before_view["b"]) == [3, 1]
    assert list(result.after_view["b"]) == [300, 99]
    assert "extra" not in result.before_view.columns
    assert list(result.before_view.columns) == ["a", "b"]


@pytest.mark.unit
def test_aligned_empty_preview_when_no_cells_changed():
    rid = AGENT_ROW_ID
    raw = pd.DataFrame({rid: [0, 1], "a": [1, 2]})
    cleaned = pd.DataFrame({rid: [0, 1], "a": [1, 2]})
    result = preview_aligned_frames(raw, cleaned, rid, k=5)
    assert result.aligned is True
    assert len(result.before_view) == 0
    assert list(result.before_view.columns) == ["a"]


@pytest.mark.unit
def test_fallback_skips_unchanged_leading_rows():
    rid = AGENT_ROW_ID
    raw = pd.DataFrame({rid: [0, 1, 2], "a": [1, 2, 3]})
    cleaned = pd.DataFrame({"a": [1, 20, 30]})
    result = preview_aligned_frames(raw, cleaned, rid, k=2)
    assert result.aligned is False
    assert list(result.before_view["a"]) == [2, 3]
    assert list(result.after_view["a"]) == [20, 30]


@pytest.mark.unit
def test_aligned_only_surviving_ids_when_cleaned_drops_rows():
    """Preview is only overlapping ids; tie on one column each uses cleaned id order."""
    rid = AGENT_ROW_ID
    raw = pd.DataFrame({rid: [0, 1, 2, 3], "v": [10, 20, 30, 40]})
    cleaned = pd.DataFrame({rid: [2, 0], "v": [300, 100]})
    result = preview_aligned_frames(raw, cleaned, rid, k=10)
    assert result.aligned is True
    assert rid not in result.before_view.columns
    # Both rows have one column different; tie-break: cleaned order ids 2 then 0
    assert list(result.before_view["v"]) == [30, 10]
    assert list(result.after_view["v"]) == [300, 100]
    assert len(result.before_view) == 2
    assert len(result.after_view) == 2


@pytest.mark.unit
def test_unaligned_when_row_id_missing_from_cleaned():
    """If ``row_id`` is absent from cleaned, use fallback and ``aligned`` is False."""
    rid = AGENT_ROW_ID
    raw = pd.DataFrame({rid: [0, 1], "b": [3, 4], "a": [1, 2]})
    cleaned = pd.DataFrame({"a": [1, 2], "b": [30, 40]})
    result = preview_aligned_frames(raw, cleaned, rid, k=2)
    assert result.aligned is False
    assert rid not in result.before_view.columns
    assert rid not in result.after_view.columns
    assert list(result.before_view.columns) == list(result.after_view.columns)
    assert list(result.before_view.columns) == ["b", "a"]


@pytest.mark.unit
def test_unaligned_when_row_id_missing_from_input_frame():
    rid = AGENT_ROW_ID
    raw = pd.DataFrame({"a": [1, 2]})
    cleaned = pd.DataFrame({rid: [0], "a": [1]})
    result = preview_aligned_frames(raw, cleaned, rid, k=5)
    assert result.aligned is False


@pytest.mark.unit
def test_preview_pads_to_k_with_matching_rows_when_mismatches_sparse():
    """When fewer than k rows differ, pad with matching intersection rows."""
    rid = AGENT_ROW_ID
    raw = pd.DataFrame({rid: list(range(10)), "v": list(range(10))})
    cleaned = raw.copy()
    cleaned.loc[0, "v"] = 999
    result = preview_aligned_frames(raw, cleaned, rid, k=5)
    assert result.aligned is True
    assert len(result.before_view) == 5
    assert len(result.after_view) == 5


@pytest.mark.unit
def test_top_k_prefers_more_column_mismatches():
    """k=2 picks rows with 3 then 2 column diffs before the row with 1 diff."""
    rid = AGENT_ROW_ID
    raw = pd.DataFrame({
        rid: [0, 1, 2],
        "a": [1, 1, 1],
        "b": [2, 2, 2],
        "c": [3, 3, 3],
    })
    cleaned = pd.DataFrame({
        rid: [0, 1, 2],
        "a": [9, 1, 1],
        "b": [8, 2, 9],
        "c": [7, 3, 9],
    })
    result = preview_aligned_frames(raw, cleaned, rid, k=2)
    assert result.aligned is True
    assert list(result.before_view["a"]) == [1, 1]
    assert list(result.before_view["b"]) == [2, 2]
    assert list(result.before_view["c"]) == [3, 3]
    assert list(result.after_view["a"]) == [9, 1]
    assert list(result.after_view["b"]) == [8, 9]
    assert list(result.after_view["c"]) == [7, 9]


@pytest.mark.unit
def test_distinct_ids_tiebreak_cleaned_order_when_same_mismatch_count():
    rid = AGENT_ROW_ID
    raw = pd.DataFrame({rid: [0, 1, 2], "x": [1, 2, 3]})
    cleaned = pd.DataFrame({rid: [1, 1, 2, 0], "x": [20, 20, 30, 10]})
    result = preview_aligned_frames(raw, cleaned, rid, k=2)
    assert result.aligned is True
    # Distinct id order in cleaned: 1, 2, 0 — each has one x mismatch; k=2 -> ord 0,1
    assert list(result.before_view["x"]) == [2, 3]
    assert list(result.after_view["x"]) == [20, 30]


@pytest.mark.unit
def test_preview_excludes_row_id_when_aligned():
    rid = AGENT_ROW_ID
    raw = pd.DataFrame({"z": [1], rid: [0]})
    cleaned = pd.DataFrame({rid: [0], "z": [10]})
    result = preview_aligned_frames(raw, cleaned, rid, k=1)
    assert result.aligned is True
    assert rid not in result.before_view.columns
    assert list(result.before_view.columns) == ["z"]


@pytest.mark.unit
def test_preview_column_order_follows_upload_csv_order():
    """Common columns appear in raw column order, not sorted alphabetically."""
    rid = AGENT_ROW_ID
    raw = pd.DataFrame({
        "m": [1, 2],
        rid: [0, 1],
        "z": [3, 4],
        "a": [5, 6],
    })
    cleaned = pd.DataFrame({
        "z": [30, 40],
        rid: [0, 1],
        "a": [50, 60],
        "m": [10, 20],
    })
    result = preview_aligned_frames(raw, cleaned, rid, k=2)
    assert result.aligned is True
    assert list(result.before_view.columns) == ["m", "z", "a"]
    assert list(result.after_view.columns) == ["m", "z", "a"]


@pytest.mark.unit
def test_aligned_surfaces_rows_only_in_after():
    rid = AGENT_ROW_ID
    raw = pd.DataFrame({rid: [0, 1], "v": [1, 2]})
    cleaned = pd.DataFrame({rid: [0, 1, 2], "v": [1, 2, 99]})
    result = preview_aligned_frames(raw, cleaned, rid, k=5)
    assert result.aligned is True
    assert rid not in result.only_in_before.columns
    assert result.only_in_before.empty
    assert not result.only_in_after.empty
    assert list(result.only_in_after["v"]) == [99]


@pytest.mark.unit
def test_aligned_surfaces_rows_only_in_before():
    rid = AGENT_ROW_ID
    raw = pd.DataFrame({rid: [0, 1, 2], "v": [1, 2, 3]})
    cleaned = pd.DataFrame({rid: [0, 1], "v": [10, 20]})
    result = preview_aligned_frames(raw, cleaned, rid, k=5)
    assert result.aligned is True
    assert rid not in result.only_in_before.columns
    assert not result.only_in_before.empty
    assert list(result.only_in_before["v"]) == [3]
    assert result.only_in_after.empty


@pytest.mark.unit
def test_preview_city_shows_strings_when_cleaned_uses_category_dtype():
    """Regression: dtype/null representation differences must not hide city values in preview."""
    rid = AGENT_ROW_ID
    raw = pd.read_csv("data/sample_data.csv").reset_index(drop=True)
    raw[rid] = raw.index.astype("int64")
    cleaned = raw.copy()
    cleaned["city"] = cleaned["city"].astype("category")
    cleaned.loc[0, "department"] = "SalesX"
    result = preview_aligned_frames(raw, cleaned, rid, k=5)
    assert result.aligned is True
    assert not result.after_view.empty
    city_after = result.after_view["city"].tolist()
    assert any(pd.notna(v) and str(v) != "nan" for v in city_after)
    assert "New York" in city_after or "Boston" in city_after


@pytest.mark.unit
def test_reorder_cleaned_for_export_upload_order_then_new_columns():
    rid = AGENT_ROW_ID
    raw = pd.DataFrame({
        "m": [1, 2],
        rid: [0, 1],
        "z": [3, 4],
        "a": [5, 6],
    })
    cleaned = pd.DataFrame({
        "z": [30, 40],
        "extra": [7, 8],
        rid: [0, 1],
        "a": [50, 60],
        "m": [10, 20],
    })
    out = reorder_cleaned_for_export(raw, cleaned, rid)
    assert rid not in out.columns
    assert list(out.columns) == ["m", "z", "a", "extra"]


@pytest.mark.unit
def test_reorder_cleaned_for_export_empty_after_drop_row_id():
    rid = AGENT_ROW_ID
    raw = pd.DataFrame({rid: [0]})
    cleaned = pd.DataFrame({rid: [0]})
    out = reorder_cleaned_for_export(raw, cleaned, rid)
    assert out.shape == (1, 0)
