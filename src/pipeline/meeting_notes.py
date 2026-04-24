"""pipeline/meeting_notes.py — Structured Scrum event meeting notes.

MeetingNotesManager:
  - Generates structured note templates from current sprint state for each
    of the five Scrum events.
  - Persists notes in sprint_state.json["meeting_notes"] via SprintStateStore.
  - Exposes retrospective action_items for cross-sprint injection via
    SprintStateStore.get_retro_actions().

No LLM calls — notes are derived deterministically from sprint state.
Retrospective action items are first-class state so the next sprint planning
pass can load and act on them.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from src.pipeline.scrum_team.sprint_state_store import SprintStateStore

logger = logging.getLogger(__name__)

SCRUM_EVENTS = frozenset({
    "sprint_planning",
    "daily_scrum",
    "sprint_review",
    "sprint_retrospective",
    "backlog_refinement",
})


class MeetingNotesManager:
    """
    Creates and persists structured Scrum event meeting notes.

    Meeting note schema
    -------------------
    {
      "note_id":       "MN-001",
      "event_type":    "sprint_planning",
      "date":          "2026-04-24",
      "participants":  ["ProductOwnerAgent", "ScrumMasterAgent"],
      "sprint_goal":   "...",
      "decisions":     ["Decision text", ...],
      "blockers":      ["Blocker description", ...],
      "action_items":  [{"owner": "...", "action": "..."}, ...],
      "created_at":    "2026-04-24T12:00:00",
    }
    """

    def __init__(self, state_store: SprintStateStore | None = None) -> None:
        self._store = state_store or SprintStateStore()

    # ─────────────────────────────────────────────────────────────────
    #  Note generators — one per Scrum event
    # ─────────────────────────────────────────────────────────────────

    def generate_sprint_planning_notes(self) -> dict[str, Any]:
        """Build Sprint Planning notes from current state (no LLM)."""
        with self._store.state_lock():
            state = self._store.load()

        sprint       = state.get("sprint", {})
        backlog      = state.get("backlog", [])
        product_goal = state.get("product_goal", "Not defined")

        in_sprint   = [s for s in backlog if s.get("status") == "in_sprint"]
        without_ac  = [s for s in in_sprint if not s.get("acceptance_criteria")]

        decisions = [
            f"Sprint goal set to: {sprint.get('goal', 'TBD')}",
            f"{len(in_sprint)} stories selected for sprint backlog",
        ]
        if without_ac:
            decisions.append(
                f"WARNING: {len(without_ac)} selected stories lack acceptance criteria"
                " — status is needs_refinement until AC are defined"
            )

        retro_actions = self._store.get_retro_actions()
        if retro_actions:
            decisions.append(
                f"Carrying {len(retro_actions)} retrospective action(s) from last sprint"
            )

        return self._build_note(
            event_type="sprint_planning",
            participants=["ProductOwnerAgent", "ScrumMasterAgent"],
            decisions=decisions,
            blockers=[],
            action_items=[
                {"owner": "ScrumMasterAgent", "action": "Assign all sprint tasks to developers"},
                {"owner": "ProductOwnerAgent", "action": "Confirm acceptance criteria for all stories"},
            ],
            extra={"sprint_goal": sprint.get("goal", ""), "product_goal": product_goal},
        )

    def generate_daily_scrum_notes(self) -> dict[str, Any]:
        """Build Daily Scrum notes from current task state (no LLM)."""
        with self._store.state_lock():
            state = self._store.load()

        tasks   = state.get("tasks", [])
        sprint  = state.get("sprint", {})
        blocked = [t for t in tasks if t.get("status") == "blocked"]
        in_prog = [t for t in tasks if t.get("status") == "in_progress"]
        done    = [t for t in tasks if t.get("status") == "done"]

        decisions = [
            f"In progress: {len(in_prog)} tasks",
            f"Done: {len(done)} tasks",
            f"Blocked: {len(blocked)} tasks",
        ]
        blockers_list = [
            f"[{t['id']}] {t.get('title', '-')[:60]} — {t.get('blocker', 'no reason')[:60]}"
            for t in blocked
        ]
        action_items = [
            {"owner": "ScrumMasterAgent", "action": f"Resolve blocker on {t['id']}"}
            for t in blocked[:3]
        ]

        return self._build_note(
            event_type="daily_scrum",
            participants=["ScrumMasterAgent", "DeveloperAgent"],
            decisions=decisions,
            blockers=blockers_list,
            action_items=action_items,
            extra={"sprint_goal": sprint.get("goal", "")},
        )

    def generate_sprint_review_notes(self) -> dict[str, Any]:
        """Build Sprint Review notes from done/accepted state (no LLM)."""
        with self._store.state_lock():
            state = self._store.load()

        tasks   = state.get("tasks", [])
        backlog = state.get("backlog", [])
        sprint  = state.get("sprint", {})

        done_tasks     = [t for t in tasks if t.get("status") == "done"]
        accepted       = [s for s in backlog if s.get("status") == "accepted"]
        pending_review = [t for t in tasks if t.get("po_status") == "ready_for_review"]
        velocity       = sprint.get("velocity", len(done_tasks))

        decisions = [
            f"Sprint velocity: {velocity} story points",
            f"{len(done_tasks)} tasks completed, {len(accepted)} stories accepted by PO",
        ]
        if pending_review:
            decisions.append(
                f"{len(pending_review)} stories still pending PO review"
            )

        action_items = [
            {
                "owner": "ProductOwnerAgent",
                "action": f"Review and accept/reject story linked to {t['id']}",
            }
            for t in pending_review[:3]
        ]

        return self._build_note(
            event_type="sprint_review",
            participants=["ProductOwnerAgent", "ScrumMasterAgent", "DeveloperAgent"],
            decisions=decisions,
            blockers=[],
            action_items=action_items,
            extra={"velocity": velocity, "increment_size": len(accepted)},
        )

    def generate_retrospective_notes(
        self,
        keep: list[str] | None = None,
        improve: list[str] | None = None,
        actions: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build Sprint Retrospective notes and persist action items.

        Parameters
        ----------
        keep:    Things that went well.
        improve: Areas to improve next sprint.
        actions: Concrete action items for next sprint (persisted to state for
                 cross-sprint injection into sprint planning).
        """
        with self._store.state_lock():
            state = self._store.load()

        tasks  = state.get("tasks", [])
        sprint = state.get("sprint", {})

        done_count    = sum(1 for t in tasks if t.get("status") == "done")
        blocked_count = sum(1 for t in tasks if t.get("status") == "blocked")

        keep    = keep    or [f"{done_count} tasks completed successfully this sprint"]
        improve = improve or (
            [f"Resolve {blocked_count} recurring blockers before next sprint"]
            if blocked_count else ["Continue current pace"]
        )
        actions = actions or []

        decisions = [
            "Keep: "    + "; ".join(keep),
            "Improve: " + "; ".join(improve),
        ]

        note = self._build_note(
            event_type="sprint_retrospective",
            participants=["ScrumMasterAgent", "DeveloperAgent"],
            decisions=decisions,
            blockers=[],
            action_items=[{"owner": "ScrumMasterAgent", "action": a} for a in actions],
            extra={"keep": keep, "improve": improve},
        )

        # Persist retro actions so next sprint planning can pick them up
        if actions:
            self._store.add_retro_actions(actions)
            logger.info("meeting_notes: persisted %d retro actions", len(actions))

        return note

    def generate_backlog_refinement_notes(self) -> dict[str, Any]:
        """Build Backlog Refinement notes (no LLM)."""
        with self._store.state_lock():
            state = self._store.load()

        backlog = state.get("backlog", [])

        draft_stories       = [s for s in backlog if s.get("status") == "draft"]
        needs_refinement    = [s for s in backlog if s.get("status") == "needs_refinement"]
        stories_without_ac  = [s for s in backlog if not s.get("acceptance_criteria")]

        decisions = [
            f"Total backlog: {len(backlog)} stories",
            f"Draft: {len(draft_stories)}, Needs refinement: {len(needs_refinement)}",
        ]
        if stories_without_ac:
            decisions.append(
                f"{len(stories_without_ac)} stories still lack acceptance criteria"
            )

        action_items = [
            {
                "owner": "ProductOwnerAgent",
                "action": f"Define acceptance criteria for {s['story_id']} — {s.get('title', '-')[:50]}",
            }
            for s in stories_without_ac[:5]
        ]

        return self._build_note(
            event_type="backlog_refinement",
            participants=["ProductOwnerAgent", "ScrumMasterAgent"],
            decisions=decisions,
            blockers=[],
            action_items=action_items,
        )

    # ─────────────────────────────────────────────────────────────────
    #  Persistence helpers
    # ─────────────────────────────────────────────────────────────────

    def save(self, note: dict[str, Any]) -> str:
        """Persist a note dict to sprint state. Returns the note_id."""
        return self._store.add_meeting_note(note)

    def get_all(self, event_type: str | None = None) -> list[dict[str, Any]]:
        """Return all persisted notes, optionally filtered by event_type."""
        return self._store.get_meeting_notes(event_type=event_type)

    def get_retro_actions(self) -> list[str]:
        """Return action items from the most recent retrospective."""
        return self._store.get_retro_actions()

    # ─────────────────────────────────────────────────────────────────
    #  Internal factory
    # ─────────────────────────────────────────────────────────────────

    def _build_note(
        self,
        event_type: str,
        participants: list[str],
        decisions: list[str],
        blockers: list[str],
        action_items: list[dict],
        extra: dict | None = None,
    ) -> dict[str, Any]:
        if event_type not in SCRUM_EVENTS:
            raise ValueError(
                f"Unknown event_type {event_type!r}. Valid: {sorted(SCRUM_EVENTS)}"
            )
        note: dict[str, Any] = {
            "event_type":   event_type,
            "date":         str(datetime.now().date()),
            "participants": participants,
            "decisions":    decisions,
            "blockers":     blockers,
            "action_items": action_items,
            "created_at":   datetime.now().isoformat(timespec="seconds"),
        }
        if extra:
            note.update(extra)
        return note
