"""services/api/sprint_bridge.py — HTTP bridge for the autonomous sprint loop.

Wraps src.loop.sprint_loop.run_sprint() so it can be invoked from
/v1/chat/completions (model="cognitwin-sprint") and the debug endpoint
/v1/sprint/run.

Responsibilities
----------------
  1. Call run_sprint(goal) with the user's text as the sprint goal.
  2. Read the resulting sprint_state.json snapshot via SprintStateStore.
  3. Render a human-readable summary suitable for LibreChat's chat bubble:
       • Sprint Goal + Product Goal
       • Product Backlog (stories w/ AC count + status)
       • Roadmap packages (if any)
       • Tasks executed this run
       • C8 Acceptance Criteria results (per task)
       • PO review outcome (accept / reject + missing criteria)
       • Product Increment
       • Blocked / rejected items
       • Retro actions + latest meeting notes

No agile_workflow refactor, no behaviour change in existing model routes.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _extract_goal(user_text: str) -> str:
    """Strip leading "/sprint" or "sprint:" prefixes users may type."""
    t = (user_text or "").strip()
    for prefix in ("/sprint ", "/sprint:", "sprint:", "Sprint:"):
        if t.lower().startswith(prefix.lower()):
            return t[len(prefix):].strip()
    return t


def _fmt_backlog(backlog: list[dict]) -> list[str]:
    if not backlog:
        return ["  (empty)"]
    lines = []
    for s in backlog[:20]:
        ac = len(s.get("acceptance_criteria") or [])
        lines.append(
            f"  [{s.get('story_id', '?')}] {s.get('title', '-')[:70]}"
            f"  · {s.get('priority', '?')} · {s.get('status', '?')} · AC={ac}"
        )
    if len(backlog) > 20:
        lines.append(f"  … +{len(backlog) - 20} more")
    return lines


def _fmt_roadmap(roadmap: list[dict]) -> list[str]:
    if not roadmap:
        return ["  (no roadmap packages)"]
    return [
        f"  [{e.get('package_id', '?')}] {e.get('title', '-')[:60]}"
        f"  · sprint={e.get('target_sprint', '-') or '-'}"
        f"  · date={e.get('target_date', '-') or '-'}"
        f"  · status={e.get('status', '-')}"
        for e in roadmap[:10]
    ]


def _fmt_tasks_with_ac(tasks: list[dict], task_ids: set[str]) -> list[str]:
    lines = []
    for t in tasks:
        if t.get("id") not in task_ids:
            continue
        ac     = t.get("acceptance_criteria") or []
        status = t.get("status", "?")
        po     = t.get("po_status", "-")
        c8     = "PASS" if t.get("ac_validated") else ("SKIP" if not ac else "FAIL")
        legacy = " · LEGACY(no-AC)" if t.get("legacy_no_ac") else ""
        lines.append(
            f"  [{t.get('id', '?')}] {t.get('title', '-')[:60]}"
            f"  · status={status} · po={po} · C8={c8} · AC={len(ac)}{legacy}"
        )
    return lines or ["  (no tasks executed)"]


def _fmt_blocked(blocked: list[dict]) -> list[str]:
    if not blocked:
        return ["  (none)"]
    lines = []
    for b in blocked[:10]:
        line = (
            f"  ✗ {b.get('story_id', '?')} / {b.get('task_id', '-')}"
            f"  — {str(b.get('reason', ''))[:80]}"
        )
        missing = b.get("missing_criteria") or []
        if missing:
            line += f"\n      missing: {'; '.join(missing[:3])}"
        lines.append(line)
    return lines


def _fmt_meeting_notes(notes: list[dict]) -> list[str]:
    if not notes:
        return []
    # Most recent 3, newest last in state — show last 3 reversed
    recent = notes[-3:]
    lines = []
    for n in recent:
        lines.append(
            f"  [{n.get('note_id', '?')}] {n.get('event_type', '-')}"
            f"  · {n.get('date', '-')}"
        )
        for d in (n.get("decisions") or [])[:2]:
            lines.append(f"      · {str(d)[:80]}")
    return lines


def render_sprint_report(result: Any, store: Any, executed_task_ids: set[str]) -> str:
    """Render the SprintResult + state snapshot into a readable text block."""
    state = store.load()
    lines: list[str] = []

    lines.append("=== COGNITWIN SPRINT RUN ===")
    lines.append(f"Sprint ID   : {result.sprint_id}")
    lines.append(f"Goal        : {result.goal}")
    lines.append(f"Product Goal: {state.get('product_goal') or '(not set)'}")
    lines.append(f"Sprint Goal : {state.get('sprint', {}).get('goal', '-')}")
    lines.append(
        f"Summary     : {len(result.completed_stories)} completed · "
        f"{len(result.blocked_stories)} blocked · "
        f"confidence={result.avg_confidence:.0%} · steps={result.total_steps}"
    )

    lines.append("\n-- Product Backlog --")
    lines += _fmt_backlog(state.get("backlog", []))

    roadmap = state.get("roadmap", [])
    if roadmap:
        lines.append("\n-- Roadmap --")
        lines += _fmt_roadmap(roadmap)

    lines.append("\n-- Tasks executed this run (C8 / PO status) --")
    lines += _fmt_tasks_with_ac(state.get("tasks", []), executed_task_ids)

    increment = state.get("increment", [])
    lines.append("\n-- Product Increment (PO-accepted tasks) --")
    if increment:
        lines.append("  " + ", ".join(increment))
    else:
        lines.append("  (empty)")

    lines.append("\n-- Blocked / Rejected --")
    lines += _fmt_blocked(result.blocked_stories)

    retro = state.get("retro_actions", [])
    if retro:
        lines.append("\n-- Retrospective actions carried forward --")
        for a in retro:
            lines.append(f"  · {a}")

    notes = state.get("meeting_notes", [])
    if notes:
        lines.append("\n-- Recent meeting notes --")
        lines += _fmt_meeting_notes(notes)

    lines.append("\n============================")
    return "\n".join(lines)


def run_sprint_for_ui(user_text: str) -> dict:
    """Execute run_sprint and return a LibreChat-compatible dict.

    Returns
    -------
    {
        "answer":        <readable summary>,
        "workflow_meta": {
            "sprint_id":         str,
            "completed_stories": int,
            "blocked_stories":   int,
            "avg_confidence":    float,
            "total_steps":       int,
            "increment":         list[str],
        }
    }
    """
    # Lazy import — run_sprint imports developer_runner which pulls in ollama.
    # Keep the import here so test modules that patch sprint_loop can do so
    # without ollama on the path.
    from src.loop.sprint_loop import run_sprint  # noqa: PLC0415
    from src.pipeline.scrum_team.sprint_state_store import SprintStateStore  # noqa: PLC0415

    goal = _extract_goal(user_text) or "Autonomous sprint run (no explicit goal)"
    logger.info("sprint_bridge: invoking run_sprint goal=%r", goal[:80])

    # Snapshot task IDs before run so we can report which ones were created.
    store      = SprintStateStore()
    before_ids = {t.get("id") for t in store.load().get("tasks", [])}

    result = run_sprint(goal)

    after_ids         = {t.get("id") for t in store.load().get("tasks", [])}
    executed_task_ids = after_ids - before_ids

    answer = render_sprint_report(result, store, executed_task_ids)

    return {
        "answer": answer,
        "workflow_meta": {
            "sprint_id":         result.sprint_id,
            "completed_stories": len(result.completed_stories),
            "blocked_stories":   len(result.blocked_stories),
            "avg_confidence":    result.avg_confidence,
            "total_steps":       result.total_steps,
            "increment":         store.get_increment(),
        },
    }
