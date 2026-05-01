"""src/loop/sprint_review_synth.py — Sprint Review synthesis helpers.

Pure functions used by the API to build the system_review block, merge it with
human_review notes into unified Sprint Notes, and propose the next sprint.

All functions have deterministic fallbacks so the test suite never needs an
LLM. The PO LLM agent is only consulted when explicitly enabled.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_system_review(state: dict, *, generated_files_count: int = 0) -> dict:
    """Aggregate sprint_state into a system_review payload."""
    from src.loop.sprint_summary import compute_summary  # local import

    tasks   = state.get("tasks", []) or []
    backlog = state.get("backlog", []) or []
    increment = state.get("increment", {}) or {}
    summary = compute_summary(tasks, generated_files_count=int(generated_files_count or 0))

    done    = [t for t in tasks if str(t.get("status", "")).lower() in ("done", "completed", "accepted")]
    blocked = [t for t in tasks if str(t.get("status", "")).lower() in ("blocked", "rejected", "failed")]
    waiting = [t for t in tasks if str(t.get("status", "")).lower() in ("waiting_review",)]

    return {
        "summary":              summary,
        "done_titles":          [t.get("title", t.get("id", "?")) for t in done],
        "blocked_titles":       [t.get("title", t.get("id", "?")) for t in blocked],
        "waiting_review_titles":[t.get("title", t.get("id", "?")) for t in waiting],
        "increment":            increment,
        "backlog_count":        len(backlog),
        "generated_at":         _now_iso(),
    }


def synthesize_merged_notes(system_review: dict, human_review: dict | None) -> str:
    """Produce a merged Sprint Notes string from system + human review."""
    s = system_review or {}
    h = human_review  or {}
    summary = s.get("summary") or {}

    lines: list[str] = []
    lines.append("# Sprint Notes")
    lines.append("")
    lines.append("## Outcomes")
    lines.append(
        f"- {summary.get('done_tasks', 0)} done, "
        f"{summary.get('blocked_tasks', 0)} blocked, "
        f"{summary.get('waiting_review_tasks', 0)} waiting review "
        f"(confidence {summary.get('confidence', 0)}%)."
    )
    if s.get("done_titles"):
        lines.append("- Delivered: " + "; ".join(s["done_titles"][:8]))
    if s.get("blocked_titles"):
        lines.append("- Blocked: " + "; ".join(s["blocked_titles"][:8]))

    text = (h.get("text") or "").strip()
    if text:
        reviewer = h.get("reviewer_name") or h.get("reviewer") or "reviewer"
        lines.append("")
        lines.append(f"## Human Review — {reviewer}")
        lines.append(text)

    return "\n".join(lines).strip()


def propose_next_sprint(
    system_review: dict,
    backlog_items: list[dict] | None,
    *,
    last_goal: str = "",
) -> dict:
    """Propose a next-sprint plan from this sprint's outcomes + project backlog."""
    backlog_items = backlog_items or []
    blocked_titles = (system_review or {}).get("blocked_titles", []) or []

    # Carry-over: any item still 'in_sprint' in the closing sprint becomes a
    # candidate. We expose item IDs that the caller will hand back to planning.
    carry_over_item_ids: list[str] = [
        it.get("item_id") for it in backlog_items
        if str(it.get("status", "")).lower() == "in_sprint" and it.get("item_id")
    ]

    # Recommend the highest-priority untouched items from the backlog.
    priority_rank = {"high": 3, "medium": 2, "low": 1}
    fresh = [
        it for it in backlog_items
        if str(it.get("status", "")).lower() in ("new", "deferred")
        and it.get("item_id")
    ]
    fresh.sort(key=lambda it: -priority_rank.get(str(it.get("priority") or "medium").lower(), 2))
    recommended = [it["item_id"] for it in fresh[:5]]

    if blocked_titles:
        suggested_goal = f"Recover from blocked items: {'; '.join(blocked_titles[:3])}"[:280]
    elif last_goal:
        suggested_goal = f"Build on previous sprint: {last_goal}"[:280]
    elif recommended:
        suggested_goal = "Tackle highest-priority backlog items"
    else:
        suggested_goal = "Plan next increment"

    return {
        "suggested_goal":            suggested_goal,
        "carry_over_item_ids":       carry_over_item_ids,
        "recommended_new_item_ids":  recommended,
        "generated_at":              _now_iso(),
    }
