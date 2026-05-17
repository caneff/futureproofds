"""Pure helpers for aligned before/after DataFrame previews in the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from datacompy.core import Compare

# Must match data_cleaning_agent.utils.APP_SYNTHETIC_ALIGN_ROW_ID_COLUMN
AGENT_ROW_ID = "__agent_row_id__"


@dataclass(frozen=True)
class AlignedPreview:
    """Result of ``preview_aligned_frames``.

    Attributes
    ----------
    before_view
        Left-hand preview suitable for ``st.dataframe`` (no ``row_id`` column;
        default RangeIndex).
    after_view
        Right-hand preview; same column order as ``before_view`` when aligned.
    aligned
        True when ``row_id`` was present in both frames and previews list up to
        ``k`` intersection rows that differ, chosen by **most differing
        columns** (tie: cleaned id order when aligned, else row index).
    only_in_before
        Rows present only in ``raw`` by join key (aligned path), without
        ``row_id``, column order derived from ``raw``.
    only_in_after
        Rows present only in ``cleaned`` by join key (aligned path), without
        ``row_id``, column order derived from ``cleaned``.
    """

    before_view: pd.DataFrame
    after_view: pd.DataFrame
    aligned: bool
    only_in_before: pd.DataFrame = field(default_factory=pd.DataFrame)
    only_in_after: pd.DataFrame = field(default_factory=pd.DataFrame)


def _all_distinct_in_order(series: pd.Series) -> list:
    """Distinct values of ``series`` in first-appearance order."""
    seen: set = set()
    out: list = []
    for val in series:
        if val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def _preview_columns_in_upload_order(
    upload: pd.DataFrame,
    cleaned: pd.DataFrame,
    row_id: str,
) -> list[str]:
    """Names common to upload and cleaned (excluding ``row_id``), in upload column order."""
    common = (set(upload.columns) & set(cleaned.columns)) - {row_id}
    return [c for c in upload.columns if c in common]


def _ordered_drop_row_id(
    df: pd.DataFrame, ref: pd.DataFrame, row_id: str
) -> pd.DataFrame:
    """Drop ``row_id`` and order remaining columns like ``ref`` then trailing names."""
    if df.empty:
        return pd.DataFrame()
    d = df.drop(columns=[row_id], errors="ignore")
    ref_cols = [c for c in ref.columns if c != row_id and c in d.columns]
    rest = [c for c in d.columns if c not in ref_cols]
    return d.loc[:, ref_cols + rest].reset_index(drop=True).copy()


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    """Empty DataFrame with named columns (typed-friendly for pyright)."""
    if not columns:
        return pd.DataFrame()
    return pd.DataFrame({c: [] for c in columns})


def _empty_pair(cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    e = _empty_frame(cols)
    return e, e


def _normalize_object_nulls_for_compare(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy safe to pass to DataComPy ``Compare``.

    ``groupby(...).first()`` often stores missing values in ``object`` columns as
    Python ``None``, while ``Categorical`` / numeric columns use ``float(nan)``.
    DataComPy then treats ``None`` and ``nan`` as different and can surface
    ``*_df2`` cells as missing in mismatch rows even when the cleaned data is
    correct—Streamlit previews look wrong while CSV export matches.

    Coerce missing-like values in object columns to ``numpy.nan`` so comparison
    matches the on-disk CSV semantics.
    """
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if s.dtype != object:
            continue
        out[c] = s.where(s.notna(), np.nan)
    return out


_PREVIEW_DIFF_BG = "background-color: rgba(255, 230, 120, 0.45)"


def round_numeric_preview(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Copy ``df`` with numeric columns rounded to ``decimals`` places (preview display).

    ``object`` columns are converted to numeric when every non-null value parses,
    then rounded; otherwise left unchanged.
    """
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            out[col] = s.round(decimals)
            continue
        if s.dtype != object:
            continue
        num = cast(pd.Series, pd.to_numeric(s, errors="coerce"))
        if (s.isna() | num.notna()).all() and num.notna().any():
            out[col] = num.round(decimals)
    return out


def _preview_decimal_formatters(df: pd.DataFrame) -> dict[str, str]:
    """Column -> format string for two decimal places where appropriate."""
    fmt: dict[str, str] = {}
    for col in df.columns:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_integer_dtype(s):
            continue
        fmt[col] = "{:.2f}"
    return fmt


def _apply_preview_display(
    df: pd.DataFrame,
    highlight: pd.io.formats.style.Styler | None,
) -> pd.DataFrame | pd.io.formats.style.Styler:
    """Apply two-decimal display formatting; optional pre-built highlight styler."""
    if df.empty:
        return df
    fmt = _preview_decimal_formatters(df)
    if highlight is not None:
        if not fmt:
            return highlight
        return cast(
            pd.io.formats.style.Styler,
            highlight.format(cast(Any, fmt), na_rep=""),
        )
    if not fmt:
        return df
    return cast(
        pd.io.formats.style.Styler,
        df.style.format(cast(Any, fmt), na_rep=""),
    )


def diff_cell_mask(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame | None:
    """Return boolean DataFrame where ``before`` and ``after`` differ (NaN-safe).

    Returns ``None`` if frames are empty or not aligned on shape/columns.
    """
    if before.empty or after.empty:
        return None
    if before.shape != after.shape or not before.columns.equals(after.columns):
        return None
    same = before.eq(after) | (before.isna() & after.isna())
    # ``eq`` can yield NA on nullable/object columns; treat NA as "not equal".
    return (~same.fillna(False)).astype(bool)


def style_preview_pair(
    before: pd.DataFrame, after: pd.DataFrame
) -> tuple[
    pd.DataFrame | pd.io.formats.style.Styler,
    pd.DataFrame | pd.io.formats.style.Styler,
]:
    """Return ``before``/``after`` for preview: numeric cols rounded, diffs highlighted."""
    br = round_numeric_preview(before, 2)
    ar = round_numeric_preview(after, 2)
    mask = diff_cell_mask(br, ar)
    if mask is None:
        return _apply_preview_display(br, None), _apply_preview_display(ar, None)

    def _highlight_row(row: pd.Series) -> list[str]:
        idx = row.name
        # Truthiness, not ``is True``: mask values are ``numpy.bool_`` after astype(bool).
        return [_PREVIEW_DIFF_BG if mask.at[idx, c] else "" for c in row.index]

    b_h = br.style.apply(_highlight_row, axis=1)
    a_h = ar.style.apply(_highlight_row, axis=1)
    return (
        _apply_preview_display(br, b_h),
        _apply_preview_display(ar, a_h),
    )


def _mismatch_views_from_compare(
    compare: Compare,
    value_cols: list[str],
    k: int,
    row_id: str | None,
    order_map: dict | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``compare.intersect_rows`` into before/after views for ``value_cols``.

    Takes up to ``k`` mismatching rows (most differing columns first), then—if
    fewer than ``k`` mismatching rows exist but at least one mismatch does—pads
    with fully matching intersection rows in stable ``row_id`` / index order
    so the preview table reaches up to ``k`` rows when possible. When there are
    no mismatches, returns empty frames (callers may show an all-clear message).
    """
    if k <= 0 or not value_cols:
        return _empty_pair(value_cols)

    ir = compare.intersect_rows
    match_cols = [f"{c}_match" for c in value_cols if f"{c}_match" in ir.columns]
    if not match_cols:
        return _empty_pair(value_cols)

    rename_before = {f"{c}_df1": c for c in value_cols}
    rename_after = {f"{c}_df2": c for c in value_cols}
    df1_cols = [f"{c}_df1" for c in value_cols]
    df2_cols = [f"{c}_df2" for c in value_cols]

    mism = ~ir[match_cols].all(axis=1)
    sub = ir.loc[mism].copy()
    if sub.empty:
        return _empty_pair(value_cols)

    n_mismatch = (~sub[match_cols]).sum(axis=1).astype(int)
    sub["_n_mm"] = n_mismatch

    if order_map is not None and row_id is not None and row_id in sub.columns:
        sub["_ord"] = sub[row_id].map(order_map)
        sub["_ord"] = sub["_ord"].fillna(len(order_map))
        sub = sub.sort_values(
            by=["_n_mm", "_ord"],
            ascending=[False, True],
            kind="mergesort",
        )
    else:
        sub["_idx"] = sub.index
        sub = sub.sort_values(
            by=["_n_mm", "_idx"],
            ascending=[False, True],
            kind="mergesort",
        )

    sub_use = sub.head(k)
    before_m = sub_use.loc[:, df1_cols].rename(columns=rename_before)
    after_m = sub_use.loc[:, df2_cols].rename(columns=rename_after)
    n_m = len(before_m)
    if n_m >= k:
        return (
            before_m.reset_index(drop=True).copy(),
            after_m.reset_index(drop=True).copy(),
        )

    need = k - n_m
    taken_idx = set(sub_use.index)
    good = ir.loc[ir[match_cols].all(axis=1) & ~ir.index.isin(taken_idx)].copy()

    if order_map is not None and row_id is not None and row_id in good.columns:
        good["_ord_pad"] = good[row_id].map(order_map)
        good["_ord_pad"] = good["_ord_pad"].fillna(len(order_map))
        good = good.sort_values(by=["_ord_pad"], ascending=True, kind="mergesort")
    else:
        good = good.sort_index(kind="mergesort")

    good_use = good.head(need)
    if good_use.empty:
        return (
            before_m.reset_index(drop=True).copy(),
            after_m.reset_index(drop=True).copy(),
        )

    before_p = good_use.loc[:, df1_cols].rename(columns=rename_before)
    after_p = good_use.loc[:, df2_cols].rename(columns=rename_after)
    before = pd.concat([before_m, before_p], ignore_index=True)
    after = pd.concat([after_m, after_p], ignore_index=True)
    return before.copy(), after.copy()


def reorder_cleaned_for_export(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    row_id: str,
) -> pd.DataFrame:
    """Return cleaned data without ``row_id``, columns ordered for CSV export.

    Columns that appear in ``raw`` (except ``row_id``) are ordered like ``raw``.
    Any remaining columns from ``cleaned`` are appended in their order in
    ``cleaned``.
    """
    out = cleaned.drop(columns=[row_id], errors="ignore")
    if out.shape[1] == 0:
        return out.copy()

    raw_data_cols = [c for c in raw.columns if c != row_id]
    raw_data_set = set(raw_data_cols)
    ordered = [c for c in raw_data_cols if c in out.columns]
    for c in out.columns:
        if c not in raw_data_set:
            ordered.append(c)
    return out.loc[:, ordered].copy()


def preview_aligned_frames(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    row_id: str,
    k: int,
) -> AlignedPreview:
    """Build before/after preview tables via DataComPy ``Compare``.

    When ``row_id`` exists in both frames, uses ``Compare`` on one row per id.
    Among intersection rows that differ on at least one column, the preview
    keeps up to ``k`` rows with the **largest count of differing columns**
    (DataComPy ``*_match``); ties use first appearance of id in ``cleaned``.
    Rows only in raw or only in cleaned (by join key) stay in
    ``only_in_before`` / ``only_in_after``.

    When alignment is not possible, uses ``Compare(..., on_index=True)`` on
    positional slices of common columns (excluding ``row_id``). The same
    top-``k``-by-mismatch-count rule applies; ties use row index order.

    Parameters
    ----------
    raw
        Original upload (includes ``row_id`` if alignment is expected).
    cleaned
        Cleaned result from the agent.
    row_id
        Name of the stable row identifier column.
    k
        Maximum rows in the before/after preview. At least one column must
        differ on a row for it to count as a **mismatch** row; those are listed
        first (most differing columns first). If fewer than ``k`` mismatching
        rows exist, **matching** intersection rows are appended in stable id
        order until the preview reaches ``k`` rows (or intersection is exhausted).

    Returns
    -------
    AlignedPreview
        Views, alignment flag, and optional only-in-one-side tables.
    """
    kk = max(0, int(k))
    cmp_kw = {
        "cast_column_names_lower": False,
        "abs_tol": 0.0,
        "rel_tol": 0.0,
    }

    def _fallback() -> AlignedPreview:
        common = _preview_columns_in_upload_order(raw, cleaned, row_id)
        empty = AlignedPreview(
            before_view=_empty_frame(common),
            after_view=_empty_frame(common),
            aligned=False,
        )
        if not common or kk <= 0:
            return empty
        b0 = raw.drop(columns=[row_id], errors="ignore").reset_index(drop=True)
        a0 = cleaned.drop(columns=[row_id], errors="ignore").reset_index(drop=True)
        n = min(len(b0), len(a0))
        if n == 0:
            return empty
        compare = Compare(
            _normalize_object_nulls_for_compare(b0.iloc[:n][common].copy()),
            _normalize_object_nulls_for_compare(a0.iloc[:n][common].copy()),
            on_index=True,
            **cmp_kw,
        )
        before_view, after_view = _mismatch_views_from_compare(
            compare, common, kk, row_id=None, order_map=None
        )
        if before_view.empty:
            return empty
        return AlignedPreview(
            before_view=before_view,
            after_view=after_view,
            aligned=False,
        )

    if row_id not in raw.columns or row_id not in cleaned.columns:
        return _fallback()

    display_cols = _preview_columns_in_upload_order(raw, cleaned, row_id)

    cols_for_compare = [row_id] + display_cols if display_cols else [row_id]
    d1 = (
        raw
        .loc[:, [c for c in cols_for_compare if c in raw.columns]]
        .groupby(row_id, as_index=False, sort=False)
        .first()
    )
    d2 = (
        cleaned
        .loc[:, [c for c in cols_for_compare if c in cleaned.columns]]
        .groupby(row_id, as_index=False, sort=False)
        .first()
    )

    d1 = _normalize_object_nulls_for_compare(d1)
    d2 = _normalize_object_nulls_for_compare(d2)

    compare = Compare(d1, d2, join_columns=row_id, **cmp_kw)

    order_list = _all_distinct_in_order(cleaned.loc[:, row_id])
    order_map = {v: i for i, v in enumerate(order_list)}

    before_view, after_view = _mismatch_views_from_compare(
        compare, display_cols, kk, row_id=row_id, order_map=order_map
    )

    only_before = _ordered_drop_row_id(compare.df1_unq_rows, raw, row_id)
    only_after = _ordered_drop_row_id(compare.df2_unq_rows, cleaned, row_id)

    return AlignedPreview(
        before_view=before_view,
        after_view=after_view,
        aligned=True,
        only_in_before=only_before,
        only_in_after=only_after,
    )
