"""Read-only per-column display rows from a cleaning plan dict."""

from __future__ import annotations

from typing import Any

from data_cleaning_agent.utils import (
    coerce_cleaning_plan_columns,
    merged_plan_actions_by_column,
)


def plan_columns_to_summary_rows(plan: dict[str, Any] | None) -> list[dict[str, str]]:
    """
    Build display rows: each dict has ``column`` and ``actions`` (joined string).

    Parameters
    ----------
    plan
        Cleaning plan dict with ``columns`` in any shape accepted by coercion.

    Returns
    -------
    list[dict[str, str]]
        Column order follows the merged plan (coercion + first-seen column order).
    """
    if not plan or not isinstance(plan, dict):
        return []
    raw_cols = plan.get("columns")
    rows = coerce_cleaning_plan_columns(raw_cols)
    merged = merged_plan_actions_by_column(rows)
    out: list[dict[str, str]] = []
    for name, acts in merged.items():
        text = "; ".join(acts) if acts else ""
        out.append({"column": name, "actions": text})
    return out
