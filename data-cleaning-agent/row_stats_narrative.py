"""Pure helpers for Cleaning plan row-stats narrative (Streamlit UI)."""

from __future__ import annotations


def verified_row_stats_strip_items(stats: object) -> list[tuple[str, str]] | None:
    """Build (label, value) pairs for the summary strip, or None if unavailable.

    Omits the all-null column when ``removed_all_null_input_user_cols`` is
    ``None`` (row id not aligned for that measurement).
    """
    if not isinstance(stats, dict) or stats.get("error"):
        return None
    if "n_in" not in stats or "n_out" not in stats:
        return None
    n_in = int(stats["n_in"])
    n_out = int(stats["n_out"])
    removed = int(stats["removed_total"])
    items: list[tuple[str, str]] = [
        ("Rows in", f"{n_in:,}"),
        ("Rows out", f"{n_out:,}"),
        ("Removed", f"{removed:,}"),
    ]
    rnull = stats.get("removed_all_null_input_user_cols")
    if rnull is not None:
        items.append(("All-null (removed)", f"{int(rnull):,}"))
    return items


def glossary_bullets() -> list[str]:
    """Short bullets for the 'What do these numbers mean?' expander."""
    return [
        "**Verified run:** One run of the **current** cleaner on this upload "
        "(same cleaner you would get if you hit **Apply Cleaning** now)—not inferred from plan wording.",
        "**Rows in / out / removed:** Before vs after that run; rows matched by "
        "synthetic row id when it exists in both frames.",
        "**All-null (removed):** Removed rows that were all-null on **your** "
        "columns only (synthetic row id excluded). Omitted if not computable.",
        "**Missing counts:** Measurement failed (often a code error). Fix or "
        "regenerate, then reopen the plan to refresh.",
    ]
