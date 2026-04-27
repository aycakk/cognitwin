"""pipeline/weekly_reporter.py — Weekly status report generator.

Produces two structured reports from sprint state:
  1. Past-week accomplishments: completed/accepted tasks, blocked tasks,
     and review outcomes in the 7-day window.
  2. Next-week plan: ready backlog items (with AC), carry-over in-progress
     tasks, active blockers, and upcoming roadmap packages.

No LLM calls — all reports are derived deterministically from sprint state.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from src.pipeline.scrum_team.sprint_state_store import SprintStateStore

logger = logging.getLogger(__name__)


class WeeklyReporter:
    """Generates past-week and next-week reports from sprint state."""

    def __init__(self, state_store: SprintStateStore | None = None) -> None:
        self._store = state_store or SprintStateStore()

    # ─────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────

    def past_week_report(self, as_of: date | None = None) -> str:
        """Generate a past-week accomplishments report.

        Covers the 7-day window ending on *as_of* (defaults to today).
        Includes: completed tasks, accepted stories, active blockers.
        """
        cutoff = as_of or date.today()
        since  = cutoff - timedelta(days=7)

        with self._store.state_lock():
            state = self._store.load()

        tasks   = state.get("tasks", [])
        backlog = state.get("backlog", [])
        goal    = state.get("sprint", {}).get("goal", "—")

        done_tasks = [
            t for t in tasks
            if t.get("status") == "done"
            and _in_window(t.get("completed_at"), since, cutoff)
        ]
        accepted_stories = [
            s for s in backlog
            if s.get("status") == "accepted"
            and _in_window(s.get("updated_at"), since, cutoff)
        ]
        blocked_tasks = [t for t in tasks if t.get("status") == "blocked"]

        lines = [
            f"=== PAST WEEK REPORT ({since} → {cutoff}) ===",
            f"Sprint Goal : {goal}",
            "",
        ]

        lines.append(f"✓ Completed Tasks ({len(done_tasks)}):")
        if done_tasks:
            for t in done_tasks:
                lines.append(
                    f"  [{t['id']}] {t.get('title', '-')[:70]}"
                    f"  ({t.get('assignee', 'unassigned')})"
                )
        else:
            lines.append("  None completed this week.")

        lines.append(f"\n✓ Accepted Stories ({len(accepted_stories)}):")
        if accepted_stories:
            for s in accepted_stories:
                lines.append(f"  [{s['story_id']}] {s.get('title_en') or s.get('title', '-')[:70]}")
        else:
            lines.append("  None accepted this week.")

        lines.append(f"\n✗ Active Blockers ({len(blocked_tasks)}):")
        if blocked_tasks:
            for t in blocked_tasks:
                lines.append(
                    f"  [{t['id']}] {t.get('title', '-')[:60]}"
                    f"  — {t.get('blocker', 'no reason')[:60]}"
                )
        else:
            lines.append("  No active blockers.")

        lines.append("\n=== END PAST WEEK ===")
        return "\n".join(lines)

    def next_week_plan(self, as_of: date | None = None) -> str:
        """Generate a next-week plan report.

        Covers: ready/draft backlog items with AC, carry-over in-progress tasks,
        stories needing refinement, upcoming roadmap packages, and active blockers.
        """
        today         = as_of or date.today()
        next_week_end = today + timedelta(days=7)

        with self._store.state_lock():
            state = self._store.load()

        tasks   = state.get("tasks", [])
        backlog = state.get("backlog", [])
        roadmap = state.get("roadmap", [])
        goal    = state.get("sprint", {}).get("goal", "—")

        in_progress = [t for t in tasks if t.get("status") == "in_progress"]
        ready_stories = [
            s for s in backlog
            if s.get("status") in ("draft", "ready")
            and s.get("acceptance_criteria")
        ]
        needs_refinement = [
            s for s in backlog
            if s.get("status") == "needs_refinement"
            or (s.get("status") in ("draft",) and not s.get("acceptance_criteria"))
        ]
        upcoming_packages = [
            r for r in roadmap
            if r.get("status") == "planned"
            and r.get("target_date", "9999") <= str(next_week_end)
        ]
        blocked = [t for t in tasks if t.get("status") == "blocked"]

        lines = [
            f"=== NEXT WEEK PLAN ({today} → {next_week_end}) ===",
            f"Sprint Goal : {goal}",
            "",
        ]

        lines.append(f"→ Carry-over In-Progress ({len(in_progress)}):")
        if in_progress:
            for t in in_progress:
                lines.append(
                    f"  [{t['id']}] {t.get('title', '-')[:70]}"
                    f"  ({t.get('assignee', 'unassigned')})"
                )
        else:
            lines.append("  None.")

        lines.append(f"\n→ Ready for Sprint ({len(ready_stories)} stories with AC):")
        if ready_stories:
            for s in ready_stories[:5]:
                ac_count = len(s.get("acceptance_criteria", []))
                title = s.get("title_en") or s.get("title", "-")
                lines.append(
                    f"  [{s['story_id']}] {title[:60]}"
                    f"  | priority={s.get('priority', '?')}"
                    f"  | SP={s.get('story_points', '?')}"
                    f"  | AC={ac_count}"
                )
        else:
            lines.append("  No ready stories with acceptance criteria.")

        if needs_refinement:
            lines.append(
                f"\n⚠ Needs Refinement ({len(needs_refinement)} stories missing AC):"
            )
            for s in needs_refinement[:5]:
                title = s.get("title_en") or s.get("title", "-")
                lines.append(f"  [{s['story_id']}] {title[:60]}")

        if upcoming_packages:
            lines.append(f"\n→ Upcoming Roadmap Packages ({len(upcoming_packages)}):")
            for pkg in upcoming_packages:
                lines.append(
                    f"  [{pkg.get('package_id', '?')}] {pkg.get('release_package', '-')}"
                    f"  | sprint={pkg.get('target_sprint', 'TBD')}"
                    f"  | date={pkg.get('target_date', 'TBD')}"
                )

        if blocked:
            lines.append(f"\n✗ Carry-over Blockers ({len(blocked)}):")
            for t in blocked:
                lines.append(
                    f"  [{t['id']}] {t.get('title', '-')[:60]}"
                    f"  — {t.get('blocker', '')[:60]}"
                )

        lines.append("\n=== END NEXT WEEK ===")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────────────────────────────────────

def _in_window(ts_str: str | None, since: date, until: date) -> bool:
    """Return True if ISO timestamp falls within the [since, until] date range."""
    if not ts_str:
        return False
    try:
        dt = datetime.fromisoformat(ts_str).date()
        return since <= dt <= until
    except (ValueError, TypeError):
        return False
