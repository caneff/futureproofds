"""Verified before/after facts for the cleaning Streamlit UI (no Streamlit)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from data_cleaning_agent.utils import sparse_missing_share, summarize_cleaning_row_effects

logger = logging.getLogger(__name__)

DEFAULT_HIGH_MISSING = 0.4
DEFAULT_NULL_TOP_K = 10


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
        bdt = str(df_before[name].dtype)
        adt = str(df_after[name].dtype)
        if bdt != adt:
            dtype_changed.append(
                {"name": name, "before_dtype": bdt, "after_dtype": adt}
            )

    null_deltas: list[dict[str, Any]] = []
    for name in shared:
        if name == row_id_col:
            continue
        bmiss = int(df_before[name].isna().sum())
        amiss = int(df_after[name].isna().sum())
        delta = amiss - bmiss
        if delta != 0:
            null_deltas.append(
                {
                    "column": name,
                    "missing_before": bmiss,
                    "missing_after": amiss,
                    "delta": delta,
                }
            )
    null_deltas.sort(key=lambda r: abs(r["delta"]), reverse=True)
    null_deltas = null_deltas[:null_top_k]

    drop_reasons: list[dict[str, str]] = []
    for col in dropped:
        col_s = df_before[col]
        if isinstance(col_s, pd.DataFrame):
            col_s = col_s.iloc[:, 0]
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
            in_ids = set(df_before[row_id_col].tolist())
            out_ids = set(df_after[row_id_col].tolist())
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


def format_outcome_summary_markdown(
    facts: dict[str, Any],
    *,
    row_id_label: str,
) -> str:
    """Return markdown for Streamlit. ``row_id_label`` is display text only."""
    lines: list[str] = []
    r = facts["rows"]
    lines.append("**Rows**")
    lines.append(
        f"- Row count: **{r['n_before']:,}** → **{r['n_after']:,}**"
    )
    if r.get("aligned") and r.get("removed_total") is not None:
        lines.append(
            f"- By `{row_id_label}`: **{int(r['removed_total']):,}** rows removed "
            "relative to upload (intersection id logic)."
        )
        add_only = r.get("added_rows_only_in_after")
        if add_only is not None and int(add_only) > 0:
            lines.append(
                f"- By `{row_id_label}`: **{int(add_only):,}** row ids only in cleaned "
                "(new or duplicated ids)."
            )
    else:
        lines.append(
            "- Row id alignment: **unavailable** or incomplete; id-based removed / "
            "added counts are omitted."
        )

    cols = facts["columns"]
    lines.append("")
    lines.append("**Columns**")
    dropped = cols.get("dropped") or []
    added = cols.get("added") or []
    lines.append(
        f"- Dropped ({len(dropped)}): "
        + (", ".join(f"`{c}`" for c in dropped) if dropped else "—")
    )
    lines.append(
        f"- Added ({len(added)}): "
        + (", ".join(f"`{c}`" for c in added) if added else "—")
    )

    dtc = cols.get("dtype_changed") or []
    lines.append("")
    lines.append("**Dtype changes**")
    if not dtc:
        lines.append("- —")
    else:
        for e in dtc:
            lines.append(
                f"- `{e['name']}`: `{e['before_dtype']}` → `{e['after_dtype']}`"
            )

    nd = facts.get("null_deltas") or []
    lines.append("")
    lines.append("**Missing value count changes (top by |Δ|)**")
    if not nd:
        lines.append("- —")
    else:
        for row in nd:
            lines.append(
                f"- `{row['column']}`: {row['missing_before']} → "
                f"{row['missing_after']} (Δ {row['delta']:+d})"
            )

    dr = facts.get("drop_reasons") or []
    lines.append("")
    lines.append("**Dropped columns (tags)**")
    if not dr:
        lines.append("- —")
    else:
        tag_text = {
            "step_3_high_missing": "matches pipeline step 3 "
            "(input missing share > 40%)",
            "dropped": "dropped on output (other reason than high-missing rule)",
        }
        for item in dr:
            tag = item.get("tag", "dropped")
            lines.append(
                f"- `{item['column']}`: **{tag}** — {tag_text.get(tag, tag)}"
            )

    return "\n".join(lines)
