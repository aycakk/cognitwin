"""services/api/sprint_bridge.py — HTTP bridge for the autonomous sprint loop.

Wraps src.loop.sprint_loop.run_sprint() so it can be invoked from
/v1/chat/completions (model="cognitwin-sprint") and the debug endpoint
/v1/sprint/run.

Responsibilities
----------------
  1. Strip report-output instructions from the user message before passing the
     goal to run_sprint() — phrases like "Also produce: Roadmap, Past week..."
     are report directives, not product scope.
  2. Call run_sprint(cleaned_goal).
  3. After the sprint: derive/persist product goal, generate sprint-event
     meeting notes, auto-build roadmap if none exists.
  4. Render a human-readable summary:
       • Product Goal
       • Product Backlog
       • Product Roadmap
       • Tasks executed this run (C8 / PO / artifact)
       • Product Increment — strictly filtered to done+accepted+ac_validated
       • Reopened (human feedback)
       • Blocked / Rejected
       • Past Week Accomplishments
       • Next Week Plan / Carry-over
       • Sprint Meeting Notes

No agile_workflow refactor, no behaviour change in existing model routes.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Goal cleaning — strip report-output instructions before PO sees the goal
# ─────────────────────────────────────────────────────────────────────────────

# Matches "Also produce:", "Additionally produce:", "Ayrıca üret:" etc.
_INSTRUCTION_SPLIT_RE = re.compile(
    r"(?:also\s+produce|additionally\s+produce|ayrıca\s+(?:üret|oluştur)|"
    r"generate\s+also|also\s+(?:generate|create))\s*[:\-]",
    re.I,
)

# Matches standalone instruction lines that should NOT become backlog stories.
_INSTRUCTION_LINE_RE = re.compile(
    r"^\s*[-•*]?\s*(?:"
    r"product\s+(?:goal|backlog|roadmap|increment)"
    r"|past\s+week(?:\s+accomplishments?)?"
    r"|next\s+week(?:\s+plan)?"
    r"|meeting\s+notes?"
    r"|sprint\s+summ"
    r")\b",
    re.I,
)


def _clean_goal_for_po(raw_goal: str) -> str:
    """Strip report-output instruction lines from a user goal string.

    Lines like "Also produce: Product Goal, Roadmap, Past week accomplishments"
    are output-format instructions, not product scope. The PO agent should only
    receive the actual product goal so it does not create stories like
    "Past Week Tasks" or "Product Roadmap".
    """
    # Split on the "Also produce:" marker and discard everything after it.
    m = _INSTRUCTION_SPLIT_RE.search(raw_goal)
    if m:
        raw_goal = raw_goal[: m.start()].strip()

    # Also strip any remaining standalone instruction-keyword lines.
    cleaned = "\n".join(
        line for line in raw_goal.splitlines()
        if not _INSTRUCTION_LINE_RE.match(line)
    ).strip()
    return cleaned or raw_goal.strip()


# ─────────────────────────────────────────────────────────────────────────────
#  Increment eligibility — strict gate on task-object state
# ─────────────────────────────────────────────────────────────────────────────

def _task_is_increment_eligible(task: dict) -> bool:
    """Return True only when the task satisfies every Done condition.

    Rules (all must hold):
      • status == "done"          — developer completed the work
      • po_status in accepted set — PO (automated or human) accepted the output
      • ac_validated == True OR no acceptance_criteria defined

    Blocked, C8-FAIL, PO-unaccepted, and reopened tasks are always excluded.
    This function is the single source of truth for Product Increment membership.
    """
    status = task.get("status", "")
    po     = task.get("po_status", "")
    ac     = task.get("acceptance_criteria") or []
    return (
        status == "done"
        and po in ("accepted", "human_accepted")
        and (bool(task.get("ac_validated")) or not ac)
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Formatters
# ─────────────────────────────────────────────────────────────────────────────

def _extract_goal(user_text: str) -> str:
    """Strip leading '/sprint' or 'sprint:' prefixes users may type."""
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
        return ["  (no roadmap packages yet)"]
    lines = []
    for e in roadmap[:10]:
        # RoadmapPlanner uses "release_package"; manual entries may use "title"
        pkg_name = e.get("release_package") or e.get("title", "-")
        lines.append(
            f"  [{e.get('package_id', '?')}] {pkg_name[:60]}"
            f"  · sprint={e.get('target_sprint', '-') or '-'}"
            f"  · date={e.get('target_date', '-') or '-'}"
            f"  · status={e.get('status', '-')}"
        )
        sc = e.get("success_criteria") or []
        if sc:
            lines.append(f"    Success: {'; '.join(str(c) for c in sc[:2])}")
    return lines


_HUMAN_FEEDBACK_ICONS = {
    "accept":         "👤✓",
    "reject":         "👤✗",
    "change_request": "👤↺",
}

_STATUS_ICONS = {
    "done":             "✓",
    "reopened":         "↺",
    "blocked":          "✗",
    "in_progress":      "▶",
    "todo":             "○",
}


def _fmt_tasks_with_ac(tasks: list[dict], task_ids: set[str]) -> list[str]:
    lines = []
    for t in tasks:
        if t.get("id") not in task_ids:
            continue
        ac       = t.get("acceptance_criteria") or []
        status   = t.get("status", "?")
        po       = t.get("po_status", "-")
        c8       = "PASS" if t.get("ac_validated") else ("SKIP" if not ac else "FAIL")
        legacy   = " · LEGACY(no-AC)" if t.get("legacy_no_ac") else ""
        artifact = t.get("artifact_type", "unknown")
        icon     = _STATUS_ICONS.get(status, "?")

        hf_list = t.get("human_feedback") or []
        hf_tag  = ""
        if hf_list:
            latest   = hf_list[-1]
            hf_icon  = _HUMAN_FEEDBACK_ICONS.get(latest.get("action", ""), "👤?")
            hf_reason = (latest.get("reason") or "")[:50]
            hf_tag   = f" · {hf_icon} {hf_reason}" if hf_reason else f" · {hf_icon}"

        lines.append(
            f"  [{icon}] [{t.get('id', '?')}] {t.get('title', '-')[:55]}"
            f"  · status={status} · po={po} · C8={c8}"
            f" · AC={len(ac)} · artifact={artifact}{legacy}{hf_tag}"
        )
    return lines or ["  (no tasks executed)"]


def _fmt_blocked(blocked: list[dict]) -> list[str]:
    if not blocked:
        return ["  (none)"]
    lines = []
    for b in blocked[:10]:
        reason = str(b.get("reason", ""))
        if reason.startswith("PO rejected"):
            label = "⛔ PO-rejected"
        elif "Step limit" in reason or "Max reroutes" in reason:
            label = "⏱ Step-limit"
        elif "Acceptance criteria not validated" in reason:
            label = "🔒 AC-not-validated"
        else:
            label = "✗ Blocked"

        line = (
            f"  {label}: {b.get('story_id', '?')} / {b.get('task_id', '-')}"
            f"  — {reason[:80]}"
        )
        missing = b.get("missing_criteria") or []
        if missing:
            line += f"\n      missing AC: {'; '.join(missing[:3])}"
        lines.append(line)
    return lines


def _fmt_needs_attention(run_tasks: list[dict], blocked_stories: list[dict] | None = None) -> list[str]:
    """List run tasks that are blocked, C8=FAIL, or PO-rejected.

    Falls back to story-level _fmt_blocked when run_tasks is empty (e.g.
    executed_task_ids was empty and the caller passes result.blocked_stories).
    """
    def _c8_fail(t: dict) -> bool:
        ac = t.get("acceptance_criteria") or []
        return bool(ac) and not t.get("ac_validated")

    attention: list[tuple[str, dict]] = []
    for t in run_tasks:
        status = t.get("status", "")
        po     = t.get("po_status", "")
        if status == "blocked":
            attention.append(("✗ Blocked", t))
        elif po == "rejected":
            attention.append(("⛔ PO-rejected", t))
        elif _c8_fail(t):
            attention.append(("⟳ Needs retry (C8=FAIL)", t))

    if not attention:
        if blocked_stories:
            return _fmt_blocked(blocked_stories)
        return ["  (none)"]

    lines = []
    for label, t in attention[:10]:
        c8_tag = "FAIL" if _c8_fail(t) else "PASS"
        lines.append(
            f"  {label}: [{t.get('id', '?')}] {t.get('title', '-')[:55]}"
            f"  · status={t.get('status', '-')} · po={t.get('po_status', '-')}"
            f" · C8={c8_tag}"
        )
        missing = t.get("missing_criteria") or []
        if missing:
            lines[-1] += f"\n      missing AC: {'; '.join(missing[:3])}"
    return lines


def _fmt_meeting_notes(notes: list[dict]) -> list[str]:
    """Format the most recent 4 meeting notes with decisions."""
    if not notes:
        return ["  (none generated this run)"]
    lines = []
    for n in notes[-4:]:
        lines.append(
            f"  [{n.get('note_id', '?')}] {n.get('event_type', '-')}"
            f"  · {n.get('date', '-')}"
        )
        for d in (n.get("decisions") or [])[:3]:
            lines.append(f"      · {str(d)[:100]}")
        for ai in (n.get("action_items") or [])[:2]:
            owner  = ai.get("owner", "?")
            action = (ai.get("action") or "")[:80]
            lines.append(f"      → [{owner}] {action}")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
#  Report renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_sprint_report(result: Any, store: Any, executed_task_ids: set[str]) -> str:
    """Render SprintResult + state snapshot into a readable text block.

    Increment filtering guarantee
    -----------------------------
    The Product Increment section is computed ONLY from current task-object
    state — never from state["increment"] blindly. A task ID can be stale in
    state["increment"] (e.g. blocked after the last run, or human-rejected).
    _task_is_increment_eligible() re-validates every candidate.
    """
    state = store.load()
    lines: list[str] = []

    # Pre-compute task map and per-run task counts for Summary and Needs Attention.
    all_tasks_map = {t.get("id"): t for t in state.get("tasks", [])}
    run_tasks = [all_tasks_map[tid] for tid in executed_task_ids if tid in all_tasks_map]
    done_count        = sum(1 for t in run_tasks if t.get("status") == "done")
    blocked_count     = sum(1 for t in run_tasks if t.get("status") == "blocked")
    in_progress_count = sum(1 for t in run_tasks if t.get("status") == "in_progress")

    # ── Header ────────────────────────────────────────────────────────
    lines.append("=== COGNITWIN SPRINT RUN ===")
    lines.append(f"Sprint ID   : {result.sprint_id}")
    lines.append(f"Goal        : {result.goal}")
    lines.append(f"Product Goal: {state.get('product_goal') or '(not set)'}")
    lines.append(f"Sprint Goal : {state.get('sprint', {}).get('goal', '-')}")
    lines.append(
        f"Summary     : {done_count} completed tasks · "
        f"{in_progress_count} in progress · "
        f"{blocked_count} blocked · "
        f"confidence={result.avg_confidence:.0%} · steps={result.total_steps}"
    )

    # ── Product Backlog ───────────────────────────────────────────────
    lines.append("\n-- Product Backlog --")
    lines += _fmt_backlog(state.get("backlog", []))

    # ── Roadmap ───────────────────────────────────────────────────────
    roadmap = state.get("roadmap", [])
    lines.append("\n-- Product Roadmap --")
    lines += _fmt_roadmap(roadmap)

    # ── Tasks executed this run ───────────────────────────────────────
    lines.append("\n-- Tasks executed this run (C8 / PO / artifact) --")
    lines += _fmt_tasks_with_ac(state.get("tasks", []), executed_task_ids)

    # ── Product Increment — strictly re-derived from task objects ─────
    # Do NOT use state["increment"] directly: it may contain stale IDs
    # for tasks that were subsequently blocked, human-rejected, or reopened.
    safe_increment = [
        tid for tid in executed_task_ids
        if _task_is_increment_eligible(all_tasks_map.get(tid, {}))
    ]

    artifact_types = {
        all_tasks_map[tid].get("artifact_type", "unknown")
        for tid in safe_increment
        if tid in all_tasks_map
    }
    only_text_plan = artifact_types <= {"text_plan", "unknown"} and bool(safe_increment)

    lines.append("\n-- Product Increment (done + PO-accepted + C8 PASS, this run) --")
    if safe_increment:
        lines.append("  " + ", ".join(safe_increment))
        if only_text_plan:
            lines.append(
                "  ⚠ artifact_type=text_plan — Developer produced implementation plans only."
                " No real files were written or committed."
            )
        else:
            for tid in safe_increment:
                t  = all_tasks_map.get(tid, {})
                at = t.get("artifact_type", "unknown")
                if at != "text_plan":
                    cf = t.get("changed_files", [])
                    lines.append(f"  {tid}: artifact_type={at}  changed_files={cf}")
    else:
        if executed_task_ids:
            lines.append("  (empty — no task passed all Done criteria this run)")
            lines.append("  (see 'Tasks executed this run' above for per-task status)")
        else:
            lines.append("  (empty — no tasks executed)")

    # ── Reopened (human feedback) ─────────────────────────────────────
    reopened = [
        t for t in state.get("tasks", [])
        if t.get("id") in executed_task_ids and t.get("status") == "reopened"
    ]
    if reopened:
        lines.append("\n-- Reopened (human feedback — needs re-run) --")
        for t in reopened:
            hf_list  = t.get("human_feedback") or []
            latest   = hf_list[-1] if hf_list else {}
            action   = latest.get("action", "-")
            reason   = (latest.get("reason") or "")[:80]
            hf_icon  = _HUMAN_FEEDBACK_ICONS.get(action, "👤?")
            lines.append(
                f"  {hf_icon} [{t.get('id', '?')}] {t.get('title', '-')[:55]}"
                f"  — {action}: {reason}"
            )

    # ── Blocked / Rejected / Needs Attention ─────────────────────────
    lines.append("\n-- Blocked / Rejected / Needs Attention --")
    lines += _fmt_needs_attention(run_tasks, result.blocked_stories)

    # ── Past Week Accomplishments ─────────────────────────────────────
    lines.append("\n-- Past Week Accomplishments --")
    try:
        from src.pipeline.weekly_reporter import WeeklyReporter  # noqa: PLC0415
        past = WeeklyReporter(store).past_week_report()
        for line in past.splitlines():
            if not line.startswith("==="):
                lines.append(line)
    except Exception as exc:
        logger.warning("sprint_bridge: WeeklyReporter.past_week_report failed: %s", exc)
        lines.append("  (unavailable)")

    # ── Next Week Plan / Carry-over ───────────────────────────────────
    lines.append("\n-- Next Week Plan / Carry-over --")
    try:
        from src.pipeline.weekly_reporter import WeeklyReporter  # noqa: PLC0415
        nxt = WeeklyReporter(store).next_week_plan()
        for line in nxt.splitlines():
            if not line.startswith("==="):
                lines.append(line)
    except Exception as exc:
        logger.warning("sprint_bridge: WeeklyReporter.next_week_plan failed: %s", exc)
        lines.append("  (unavailable)")

    # Carry-over: backlog stories not successfully completed this run.
    # These are stories whose tasks are blocked or whose tasks are not
    # in the safe increment (i.e. they need to run again next sprint).
    executed_source_ids = {
        all_tasks_map.get(tid, {}).get("source_story_id")
        for tid in safe_increment
    }
    carryover_stories = [
        s for s in state.get("backlog", [])
        if s.get("status") not in ("accepted", "rejected")
        and s.get("story_id") not in executed_source_ids
        and s.get("story_id") is not None
    ]
    if carryover_stories:
        lines.append(f"\n  Carry-over stories ({len(carryover_stories)} not yet Done):")
        for s in carryover_stories[:8]:
            lines.append(
                f"    [{s.get('story_id', '?')}] {s.get('title', '-')[:60]}"
                f"  · {s.get('priority', '?')} · {s.get('status', '?')}"
            )

    # ── Retrospective actions ─────────────────────────────────────────
    retro = state.get("retro_actions", [])
    if retro:
        lines.append("\n-- Retrospective Actions (carry-forward) --")
        for a in retro:
            lines.append(f"  · {a}")

    # ── Sprint Meeting Notes ──────────────────────────────────────────
    notes = state.get("meeting_notes", [])
    lines.append("\n-- Sprint Meeting Notes --")
    lines += _fmt_meeting_notes(notes)

    lines.append("\n============================")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  UI entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_sprint_for_ui(user_text: str, isolated: bool = True) -> dict:
    """Execute run_sprint and return a LibreChat-compatible dict.

    Parameters
    ----------
    user_text : str
        Raw user message (may include "Also produce:" instructions).
    isolated : bool, default True
        When True (default for chat-completions), the sprint state is fully
        reset before the run so that stale backlog/tasks/roadmap/goal from a
        previous unrelated sprint do not appear in the new report.
        Set to False only for continuation runs (e.g. /v1/sprint/run with
        reset_state=false) where the caller explicitly wants to reuse state.

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
            "increment":         list[str],   # only eligible tasks
        }
    }
    """
    # Lazy imports — run_sprint pulls in ollama; keep importable without it.
    from src.loop.sprint_loop import run_sprint  # noqa: PLC0415
    from src.pipeline.scrum_team.sprint_state_store import SprintStateStore  # noqa: PLC0415

    raw_goal = _extract_goal(user_text) or "Autonomous sprint run (no explicit goal)"

    # Strip report-output instructions before passing goal to the PO/sprint loop.
    # "Also produce: Roadmap, Past week..." must not become backlog stories.
    goal = _clean_goal_for_po(raw_goal)
    if goal != raw_goal:
        logger.info(
            "sprint_bridge: stripped report instructions from goal. "
            "original=%r cleaned=%r", raw_goal[:80], goal[:80]
        )

    store = SprintStateStore()

    if isolated:
        store.reset_for_isolated_sprint()
        logger.info("sprint_bridge: state reset for isolated sprint run goal=%r", goal[:60])

    logger.info("sprint_bridge: invoking run_sprint goal=%r", goal[:80])

    before_ids = {t.get("id") for t in store.load().get("tasks", [])}

    result = run_sprint(goal)

    after_ids         = {t.get("id") for t in store.load().get("tasks", [])}
    executed_task_ids = after_ids - before_ids

    # ── Post-sprint: derive product goal if still empty ───────────────
    # sprint_loop already tries to set it, but guard here as well.
    if not store.get_product_goal() and goal:
        store.set_product_goal(goal[:300])
        logger.info("sprint_bridge: persisted product_goal from cleaned goal")

    # ── Post-sprint: auto-build roadmap if none exists ────────────────
    try:
        if not store.get_roadmap():
            from src.pipeline.roadmap_planner import RoadmapPlanner  # noqa: PLC0415
            RoadmapPlanner(store).build_from_backlog()
            logger.info("sprint_bridge: roadmap generated from backlog")
    except Exception as exc:
        logger.warning("sprint_bridge: roadmap generation failed: %s", exc)

    # ── Post-sprint: generate Scrum event meeting notes ───────────────
    try:
        from src.pipeline.meeting_notes import MeetingNotesManager  # noqa: PLC0415
        notes_mgr = MeetingNotesManager(store)
        notes_mgr.save(notes_mgr.generate_sprint_planning_notes())
        notes_mgr.save(notes_mgr.generate_sprint_review_notes())
        logger.info("sprint_bridge: sprint planning + review notes generated")
    except Exception as exc:
        logger.warning("sprint_bridge: meeting notes generation failed: %s", exc)

    answer = render_sprint_report(result, store, executed_task_ids)

    # workflow_meta.increment must reflect only eligible tasks this run.
    final_state     = store.load()
    final_tasks_map = {t.get("id"): t for t in final_state.get("tasks", [])}
    run_tasks_meta  = [final_tasks_map[tid] for tid in executed_task_ids if tid in final_tasks_map]
    run_increment   = [
        tid for tid in executed_task_ids
        if _task_is_increment_eligible(final_tasks_map.get(tid, {}))
    ]
    done_task_count        = sum(1 for t in run_tasks_meta if t.get("status") == "done")
    blocked_task_count     = sum(1 for t in run_tasks_meta if t.get("status") == "blocked")
    in_progress_task_count = sum(1 for t in run_tasks_meta if t.get("status") == "in_progress")

    return {
        "answer": answer,
        "workflow_meta": {
            "sprint_id":           result.sprint_id,
            "completed_stories":   len(result.completed_stories),
            "blocked_stories":     len(result.blocked_stories),
            "completed_tasks":     done_task_count,
            "blocked_tasks":       blocked_task_count,
            "in_progress_tasks":   in_progress_task_count,
            "avg_confidence":      result.avg_confidence,
            "total_steps":         result.total_steps,
            "increment":           run_increment,
        },
    }
