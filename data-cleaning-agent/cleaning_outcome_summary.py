"""Verified before/after facts for the cleaning Streamlit UI (no Streamlit)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from data_cleaning_agent.utils import (
    APP_SYNTHETIC_ALIGN_ROW_ID_COLUMN,
    first_column_as_series,
    sparse_missing_share,
    summarize_cleaning_row_effects,
)
from pandas.api.types import is_dtype_equal

logger = logging.getLogger(__name__)

DEFAULT_HIGH_MISSING = 0.4
DEFAULT_NULL_TOP_K = 10


def _column_heading(name: str) -> str:
    """Label for markdown; hide internal synthetic row-id column name."""
    if str(name).strip() == APP_SYNTHETIC_ALIGN_ROW_ID_COLUMN:
        return "synthetic alignment column (app-injected)"
    return str(name)


def _series_equal_ignoring_dtype(left: pd.Series, right: pd.Series) -> bool:
    """True when column values match for cleaning-summary purposes.

    Suppresses dtype-only drift (e.g. ``int64`` vs ``int32``, ``object`` vs
    ``string``) so the UI does not report a dtype change when the run did not
    materially alter values.
    """
    if left.shape[0] != right.shape[0]:
        return False
    if left.equals(right):
        return True
    l = left.reset_index(drop=True)
    r = right.reset_index(drop=True)
    try:
        pd.testing.assert_series_equal(
            l,
            r,
            check_dtype=False,
            check_names=False,
        )
    except AssertionError:
        return False
    else:
        return True


def build_cleaning_outcome_facts(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    *,
    row_id_col: str,
    high_missing_threshold: float = DEFAULT_HIGH_MISSING,
    null_top_k: int = DEFAULT_NULL_TOP_K,
) -> dict[str, Any]:
    """Build JSON-serializable facts comparing ``df_before`` to ``df_after``.

    Parameters
    ----------
    df_before, df_after
        Input and output of the same cleaner run.
    row_id_col
        Synthetic alignment column (e.g. ``preview_helpers.AGENT_ROW_ID``).
    high_missing_threshold
        Same default as pipeline step 3 in ``data_cleaning.md``.
    null_top_k
        Max number of shared columns to list in ``null_deltas`` by absolute
        change in raw ``isna()`` count.
    """
    if not 0.0 < high_missing_threshold < 1.0:
        raise ValueError("high_missing_threshold must be strictly between 0 and 1")
    if null_top_k < 1:
        raise ValueError("null_top_k must be at least 1")

    bcols = set(df_before.columns)
    acols = set(df_after.columns)
    dropped = sorted(bcols - acols)
    added = sorted(acols - bcols)
    shared = sorted(bcols & acols)

    dtype_changed: list[dict[str, str]] = []
    for name in shared:
        if name == row_id_col:
            continue
        b_ser = first_column_as_series(df_before, name)
        a_ser = first_column_as_series(df_after, name)
        if is_dtype_equal(b_ser.dtype, a_ser.dtype):
            continue
        if _series_equal_ignoring_dtype(b_ser, a_ser):
            continue
        dtype_changed.append({
            "name": name,
            "before_dtype": str(b_ser.dtype),
            "after_dtype": str(a_ser.dtype),
        })

    null_deltas: list[dict[str, Any]] = []
    for name in shared:
        if name == row_id_col:
            continue
        bmiss = int(first_column_as_series(df_before, name).isna().sum())
        amiss = int(first_column_as_series(df_after, name).isna().sum())
        delta = amiss - bmiss
        if delta != 0:
            null_deltas.append({
                "column": name,
                "missing_before": bmiss,
                "missing_after": amiss,
                "delta": delta,
            })
    null_deltas.sort(key=lambda r: abs(r["delta"]), reverse=True)
    null_deltas = null_deltas[:null_top_k]

    drop_reasons: list[dict[str, str]] = []
    for col in dropped:
        col_s = first_column_as_series(df_before, col)
        if sparse_missing_share(col_s) > high_missing_threshold:
            drop_reasons.append({"column": col, "tag": "step_3_high_missing"})
        else:
            drop_reasons.append({"column": col, "tag": "dropped"})

    aligned = row_id_col in df_before.columns and row_id_col in df_after.columns
    rows: dict[str, Any] = {
        "n_before": int(len(df_before)),
        "n_after": int(len(df_after)),
        "aligned": aligned,
        "removed_total": None,
        "added_rows_only_in_after": None,
    }
    if aligned:
        try:
            stats = summarize_cleaning_row_effects(
                df_before, df_after, row_id_col=row_id_col
            )
            rows["removed_total"] = int(stats.get("removed_total", 0))
            in_ids = set(first_column_as_series(df_before, row_id_col).tolist())
            out_ids = set(first_column_as_series(df_after, row_id_col).tolist())
            rows["added_rows_only_in_after"] = int(len(out_ids - in_ids))
        except (TypeError, ValueError, KeyError) as e:
            logger.warning("Row effect summary skipped: %s", e)
            rows["aligned"] = False
            rows["removed_total"] = None
            rows["added_rows_only_in_after"] = None

    return {
        "rows": rows,
        "columns": {
            "dropped": dropped,
            "added": added,
            "dtype_changed": dtype_changed,
        },
        "null_deltas": null_deltas,
        "drop_reasons": drop_reasons,
    }


def outcome_facts_show_any_change(facts: dict[str, Any]) -> bool:
    """True when before/after differ in shape, columns, dtypes, nulls, or enforced drops."""
    rows = facts.get("rows") or {}
    if int(rows.get("n_before", 0)) != int(rows.get("n_after", 0)):
        return True
    cols = facts.get("columns") or {}
    if cols.get("dropped") or cols.get("added") or cols.get("dtype_changed"):
        return True
    if facts.get("null_deltas"):
        return True
    if facts.get("drop_reasons"):
        return True
    return False


def format_outcome_summary_markdown(facts: dict[str, Any]) -> str:
    """Return markdown for Streamlit."""
    lines: list[str] = []
    r = facts["rows"]
    lines.append("**Rows**")
    lines.append(f"- Row count: **{r['n_before']:,}** → **{r['n_after']:,}**")

    cols = facts["columns"]
    lines.append("")
    lines.append("**Columns**")
    dropped = cols.get("dropped") or []
    added = cols.get("added") or []
    lines.append(
        f"- Dropped ({len(dropped)}): "
        + (", ".join(f"`{_column_heading(c)}`" for c in dropped) if dropped else "—")
    )
    lines.append(
        f"- Added ({len(added)}): "
        + (", ".join(f"`{_column_heading(c)}`" for c in added) if added else "—")
    )

    dtc = cols.get("dtype_changed") or []
    if dtc:
        lines.append("")
        lines.append("**Dtype Changes**")
        for e in dtc:
            lines.append(
                f"- `{_column_heading(e['name'])}`: `{e['before_dtype']}` → `{e['after_dtype']}`"
            )

    nd = facts.get("null_deltas") or []
    if nd:
        lines.append("")
        lines.append("**Missing Value Count Changes (Top by |Δ|)**")
        for row in nd:
            lines.append(
                f"- `{_column_heading(row['column'])}`: {row['missing_before']} → "
                f"{row['missing_after']} (Δ {row['delta']:+d})"
            )

    dr = facts.get("drop_reasons") or []
    if dr:
        lines.append("")
        lines.append("**Dropped Columns (Tags)**")
        tag_text = {
            "step_3_high_missing": "matches pipeline step 3 "
            "(input missing share > 40%)",
            "dropped": "dropped on output (other reason than high-missing rule)",
        }
        for item in dr:
            tag = item.get("tag", "dropped")
            lines.append(
                f"- `{_column_heading(item['column'])}`: **{tag}** — {tag_text.get(tag, tag)}"
            )

    return "\n".join(lines)
