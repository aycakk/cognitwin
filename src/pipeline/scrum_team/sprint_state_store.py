"""pipeline/scrum_team/sprint_state_store.py — single owner of sprint state.

Extracted from src/agents/scrum_master_agent.py so that both Scrum Master
and Developer runners can share the same locked state without one pipeline
instantiating the other's agent.

Architectural invariant: this is the ONLY module that reads or writes
data/sprint_state.json.  All consumers go through this interface.

  ScrumMasterAgent  →  load + mutate + save  (read/write)
  DeveloperRunner   →  read_context_block    (read-only)
"""

from __future__ import annotations

import copy
import json
import logging
import re
import threading
import warnings
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Generator

try:
    from filelock import FileLock as _FileLock

    _FILELOCK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FILELOCK_AVAILABLE = False
    warnings.warn(
        "filelock is not installed. sprint_state.json writes are NOT safe under "
        "concurrent access. Install it with: pip install filelock",
        RuntimeWarning,
        stacklevel=1,
    )

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Default state
# ─────────────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_STATE_PATH = _PROJECT_ROOT / "data" / "sprint_state.json"

DEFAULT_SPRINT_STATE: dict[str, Any] = {
    "sprint": {
        "id": "sprint-1",
        "goal": "Sprint hedefi henüz tanımlanmamış.",
        "start": str(date.today()),
        "end": "",
        "velocity": 0,
    },
    "product_goal": "",
    "tasks": [],
    "backlog": [],
    "roadmap": [],
    "meeting_notes": [],
    "increment": [],
    "retro_actions": [],
    "team": [
        # Role IDs are scenario-specific. "developer-default" is a placeholder
        # and must NEVER appear in SM or Developer output. The SM LLM assigns
        # meaningful roles (backend-developer, frontend-developer, fullstack-developer)
        # based on the PO stories. This team entry is only used for capacity tracking.
        {"id": "backend-developer",    "role": "Backend Developer",    "capacity": 8},
        {"id": "frontend-developer",   "role": "Frontend Developer",   "capacity": 8},
        {"id": "fullstack-developer",  "role": "Fullstack Developer",  "capacity": 8},
    ],
}


class SprintStateStore:
    """
    Thread-safe + process-safe sprint state persistence.

    Single architectural owner of sprint_state.json.
    Injected into both ScrumMasterAgent and DeveloperRunner.
    """

    def __init__(self, state_path: Path | None = None) -> None:
        self._path = state_path or _DEFAULT_STATE_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._thread_lock: threading.RLock = threading.RLock()
        _lock_path = self._path.with_suffix(".json.lock")
        self._file_lock = _FileLock(str(_lock_path)) if _FILELOCK_AVAILABLE else None

    @contextmanager
    def state_lock(self) -> Generator[None, None, None]:
        """Acquire both thread lock and file lock for read-modify-write."""
        with self._thread_lock:
            if self._file_lock is not None:
                with self._file_lock:
                    yield
            else:
                yield

    # ─────────────────────────────────────────────────────────────────────
    #  Core I/O (used by ScrumMasterAgent for read/write)
    # ─────────────────────────────────────────────────────────────────────

    def load(self) -> dict[str, Any]:
        """Load sprint state from disk. Creates default if missing."""
        if not self._path.exists():
            self.save(copy.deepcopy(DEFAULT_SPRINT_STATE))
            return copy.deepcopy(DEFAULT_SPRINT_STATE)
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return copy.deepcopy(DEFAULT_SPRINT_STATE)

    def save(self, state: dict[str, Any]) -> None:
        """Write sprint state to disk."""
        self._path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ─────────────────────────────────────────────────────────────────────
    #  Read-only accessors (used by DeveloperRunner and external callers)
    # ─────────────────────────────────────────────────────────────────────

    def get_sprint_goal(self) -> str:
        with self.state_lock():
            state = self.load()
        return state.get("sprint", {}).get("goal", "Sprint hedefi tanımlanmamış.")

    def get_assignments(self) -> list[dict]:
        """Return non-done assigned tasks."""
        with self.state_lock():
            state = self.load()
        return [
            t
            for t in state.get("tasks", [])
            if t.get("assignee") and t.get("status") != "done"
        ]

    def get_blocked_tasks(self) -> list[dict]:
        with self.state_lock():
            state = self.load()
        return [t for t in state.get("tasks", []) if t.get("status") == "blocked"]

    def read_context_block(self) -> str:
        """
        Build a formatted sprint context block for LLM injection.

        Used by DeveloperRunner to inject team state into the LLM prompt
        without importing or instantiating ScrumMasterAgent.
        """
        try:
            sprint_goal = self.get_sprint_goal()
            assignments = self.get_assignments()
            blocked = self.get_blocked_tasks()
        except Exception:
            return ""

        lines = ["=== SPRINT CONTEXT ==="]
        lines.append(f"Sprint Goal: {sprint_goal}")

        lines.append("Active Assignments:")
        if assignments:
            for t in assignments:
                lines.append(
                    f"  [{t.get('id', '?')}] {t.get('title', '-')}"
                    f" \u2192 {t.get('assignee', 'unassigned')}"
                    f" ({t.get('status', '?')})"
                )
        else:
            lines.append("  (none)")

        lines.append("Blocked Tasks:")
        if blocked:
            for t in blocked:
                blocker_note = t.get("blocker") or "no description"
                lines.append(
                    f"  [{t.get('id', '?')}] {t.get('title', '-')}"
                    f" | Blocker: {blocker_note}"
                )
        else:
            lines.append("  (none)")

        lines.append("=== END SPRINT CONTEXT ===")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────
    #  Backlog management (used by ProductOwnerAgent)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _next_story_id(state: dict[str, Any]) -> str:
        """Generate next S-NNN story ID from existing backlog items."""
        backlog = state.get("backlog", [])
        nums = [
            int(s["story_id"].split("-")[1])
            for s in backlog
            if re.match(r"S-\d+", s.get("story_id", ""))
        ] or [0]
        return f"S-{(max(nums) + 1):03d}"

    def add_story(
        self,
        title: str,
        description: str = "",
        priority: str = "medium",
        acceptance_criteria: list[str] | None = None,
        source_request: str = "",
        *,
        epic: str = "",
        title_en: str = "",
        title_tr: str = "",
        user_story_en: str = "",
        user_story_tr: str = "",
        story_points: int = 0,
        target_sprint: str = "",
        target_date: str = "",
        deployment_package: str = "",
    ) -> str:
        """Append a new story to the backlog. Returns the story_id.

        Canonical English fields (title_en, user_story_en, …) are optional but
        recommended for reports and papers. Turkish display fields (title_tr,
        user_story_tr) are likewise optional.

        A story is marked 'needs_refinement' when acceptance_criteria is empty;
        it becomes 'draft' once criteria are provided.
        """
        with self.state_lock():
            state = self.load()
            story_id = self._next_story_id(state)
            ac = acceptance_criteria or []
            # Enforce AC requirement: no AC → needs_refinement, not draft
            initial_status = "draft" if ac else "needs_refinement"
            story: dict[str, Any] = {
                "story_id":            story_id,
                "title":               title[:120],
                "description":         description[:500],
                "priority":            priority if priority in ("high", "medium", "low") else "medium",
                "acceptance_criteria": ac,
                "source_request":      source_request[:200],
                "status":              initial_status,
                "created_at":          datetime.now().isoformat(timespec="seconds"),
                "updated_at":          None,
                # ── Canonical English fields ──────────────────────────────────
                "epic":                epic[:80] if epic else "",
                "title_en":            (title_en or title)[:120],
                "title_tr":            title_tr[:120] if title_tr else "",
                "user_story_en":       user_story_en[:500] if user_story_en else description[:500],
                "user_story_tr":       user_story_tr[:500] if user_story_tr else "",
                "story_points":        max(0, int(story_points)) if story_points else 0,
                "target_sprint":       target_sprint[:30] if target_sprint else "",
                "target_date":         target_date[:10] if target_date else "",
                "deployment_package":  deployment_package[:80] if deployment_package else "",
            }
            backlog = state.get("backlog", [])
            backlog.append(story)
            state["backlog"] = backlog
            self.save(state)
        return story_id

    def get_story(self, story_id: str) -> dict | None:
        """Return a single backlog story by ID, or None."""
        with self.state_lock():
            state = self.load()
        for s in state.get("backlog", []):
            if s.get("story_id") == story_id:
                return s
        return None

    def get_backlog(self) -> list[dict]:
        """Return all backlog items."""
        with self.state_lock():
            state = self.load()
        return state.get("backlog", [])

    def update_story(self, sid: str, **fields: Any) -> bool:
        """Update mutable fields on a backlog story. Returns True if found.

        Parameters
        ----------
        sid : str
            The story ID to update (e.g. "S-001").
        **fields
            Allowed keys: title, description, priority, acceptance_criteria, status.
            Other keys are silently ignored.
        """
        allowed = {
            "title", "description", "priority", "acceptance_criteria", "status",
            "epic", "title_en", "title_tr", "user_story_en", "user_story_tr",
            "story_points", "target_sprint", "target_date", "deployment_package",
        }
        with self.state_lock():
            state = self.load()
            for s in state.get("backlog", []):
                if s.get("story_id") == sid:
                    for k, v in fields.items():
                        if k in allowed:
                            s[k] = v
                    # Auto-promote needs_refinement → draft when AC are now defined
                    if (
                        "acceptance_criteria" in fields
                        and fields["acceptance_criteria"]
                        and s.get("status") == "needs_refinement"
                    ):
                        s["status"] = "draft"
                    s["updated_at"] = datetime.now().isoformat(timespec="seconds")
                    self.save(state)
                    return True
        return False

    def accept_story(self, sid: str) -> bool:
        """Mark a story as accepted by the Product Owner.

        Also updates po_status on any linked sprint tasks from
        'ready_for_review' → 'accepted', closing the review loop.
        """
        with self.state_lock():
            state = self.load()
            found = False
            for s in state.get("backlog", []):
                if s.get("story_id") == sid:
                    s["status"] = "accepted"
                    s["updated_at"] = datetime.now().isoformat(timespec="seconds")
                    found = True
                    break
            if not found:
                return False
            for t in state.get("tasks", []):
                if t.get("source_story_id") == sid and t.get("po_status") == "ready_for_review":
                    t["po_status"] = "accepted"
            self.save(state)
        return True

    def reject_story(self, sid: str, reason: str = "") -> bool:
        """Mark a story as rejected by the Product Owner.

        Also updates po_status on any linked sprint tasks from
        'ready_for_review' → 'rejected', closing the review loop.
        """
        with self.state_lock():
            state = self.load()
            found = False
            for s in state.get("backlog", []):
                if s.get("story_id") == sid:
                    s["status"] = "rejected"
                    s["rejection_reason"] = reason[:200]
                    s["updated_at"] = datetime.now().isoformat(timespec="seconds")
                    found = True
                    break
            if not found:
                return False
            for t in state.get("tasks", []):
                if t.get("source_story_id") == sid and t.get("po_status") == "ready_for_review":
                    t["po_status"] = "rejected"
            self.save(state)
        return True

    def promote_story_to_sprint_task(self, story_id: str) -> str | None:
        """Move a backlog story into the sprint tasks array.

        This is the explicit PO → SM handoff point.
        Sets story status to 'in_sprint' and creates a corresponding task.
        Returns the new task ID, or None if the story was not found.
        Only ScrumMasterAgent should call this method.
        """
        with self.state_lock():
            state = self.load()
            story = None
            for s in state.get("backlog", []):
                if s.get("story_id") == story_id:
                    story = s
                    break
            if story is None:
                return None

            story["status"] = "in_sprint"
            story["updated_at"] = datetime.now().isoformat(timespec="seconds")

            # Generate next T-NNN task ID
            tasks = state.get("tasks", [])
            nums = [
                int(t["id"].split("-")[1])
                for t in tasks
                if re.match(r"T-\d+", t.get("id", ""))
            ] or [0]
            new_task_id = f"T-{(max(nums) + 1):03d}"

            task_ac = story.get("acceptance_criteria", [])
            new_task: dict[str, Any] = {
                "id":                  new_task_id,
                "title":               story.get("title", ""),
                "description":         story.get("description", ""),
                "type":                "story",
                "status":              "todo",
                "assignee":            None,
                "priority":            story.get("priority", "medium"),
                "story_points":        0,
                "blocker":             None,
                "acceptance_criteria": task_ac,
                "source_story_id":     story_id,
                "po_status":           "pending_review",
                "created_at":          datetime.now().isoformat(timespec="seconds"),
                # Legacy marker: tasks without AC bypass C8 and complete_task
                # enforcement. Surfaced here so operators and reports can audit
                # which completed tasks skipped the AC pipeline.
                "legacy_no_ac":        not bool(task_ac),
            }
            tasks.append(new_task)
            state["tasks"] = tasks
            self.save(state)
        return new_task_id

    def read_backlog_context_block(self) -> str:
        """Build a formatted backlog context block for LLM injection."""
        with self.state_lock():
            state = self.load()
        backlog = state.get("backlog", [])

        lines = ["=== BACKLOG CONTEXT ==="]

        active = [s for s in backlog if s.get("status") not in ("accepted", "rejected")]
        lines.append("Active Stories:")
        if active:
            for s in active:
                criteria_count = len(s.get("acceptance_criteria", []))
                lines.append(
                    f"  [{s.get('story_id', '?')}] {s.get('title', '-')}"
                    f" | priority={s.get('priority', '?')}"
                    f" | status={s.get('status', '?')}"
                    f" | criteria={criteria_count}"
                )
        else:
            lines.append("  (none)")

        lines.append("=== END BACKLOG CONTEXT ===")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────
    #  Backward-compat alias
    # ─────────────────────────────────────────────────────────────────────

    #: Deprecated name kept so existing callers and tests do not break.
    #: Prefer promote_story_to_sprint_task for new code.
    promote_to_sprint = promote_story_to_sprint_task

    # ─────────────────────────────────────────────────────────────────────
    #  Developer task lifecycle  (used by developer_runner)
    # ─────────────────────────────────────────────────────────────────────

    def get_tasks_for_assignee(self, assignee_id: str) -> list[dict]:
        """Return all non-done tasks assigned to *assignee_id*.

        Used by the Developer to show its own active work queue.
        """
        with self.state_lock():
            state = self.load()
        return [
            t for t in state.get("tasks", [])
            if t.get("assignee") == assignee_id and t.get("status") != "done"
        ]

    def start_task(self, task_id: str) -> bool:
        """Transition a task to 'in_progress' and record started_at.

        The SM may have already flipped status to in_progress during assignment,
        but started_at is only set when the Developer explicitly starts work.
        Always writes started_at so the Developer's command is recorded even
        when the status was already in_progress.
        Returns False when the task_id is not found.
        """
        with self.state_lock():
            state = self.load()
            for t in state.get("tasks", []):
                if t["id"] == task_id:
                    t["status"] = "in_progress"
                    t["started_at"] = datetime.now().isoformat(timespec="seconds")
                    self.save(state)
                    return True
        return False

    def mark_ac_validated(self, task_id: str) -> bool:
        """Record that the C8 acceptance-criteria gate passed for this task.

        Must be called by the sprint loop (or any caller) before complete_task()
        when the task carries non-empty acceptance_criteria. Without this flag,
        complete_task() will refuse to mark the task done.

        Returns False when the task_id is not found.
        """
        with self.state_lock():
            state = self.load()
            for t in state.get("tasks", []):
                if t["id"] == task_id:
                    t["ac_validated"] = True
                    self.save(state)
                    return True
        return False

    def complete_task(self, task_id: str, result_summary: str = "") -> bool:
        """Mark a task as done and record the developer's result summary.

        Enforcement rule: a task that has non-empty acceptance_criteria can only
        be marked done once ac_validated == True (set by mark_ac_validated()).
        Tasks with empty acceptance_criteria are unaffected (legacy behaviour).

        When the task carries a source_story_id (i.e. it was promoted from
        the PO backlog), the po_status field is set to 'ready_for_review'
        so the PO can inspect and accept or reject the linked story.

        Returns False when the task_id is not found OR when AC have not been
        validated (logs a warning in the latter case).
        """
        with self.state_lock():
            state = self.load()
            for t in state.get("tasks", []):
                if t["id"] == task_id:
                    # AC enforcement: block completion when criteria are defined
                    # but have not been validated by the C8 gate.
                    ac = t.get("acceptance_criteria", [])
                    if ac and not t.get("ac_validated"):
                        logger.warning(
                            "complete_task blocked: task %s has %d acceptance criteria "
                            "that have not been validated (ac_validated is not set). "
                            "Call mark_ac_validated() after C8 gate passes.",
                            task_id, len(ac),
                        )
                        return False
                    t["status"] = "done"
                    t["result_summary"] = result_summary[:500]
                    t["completed_at"] = datetime.now().isoformat(timespec="seconds")
                    if t.get("source_story_id"):
                        t["po_status"] = "ready_for_review"
                    self.save(state)
                    return True
        return False

    def block_task(self, task_id: str, reason: str) -> bool:
        """Mark a task as blocked and record the reason.

        Returns False when the task_id is not found.
        """
        with self.state_lock():
            state = self.load()
            for t in state.get("tasks", []):
                if t["id"] == task_id:
                    t["status"] = "blocked"
                    t["blocker"] = reason[:200]
                    t["updated_at"] = datetime.now().isoformat(timespec="seconds")
                    self.save(state)
                    return True
        return False

    def get_tasks_ready_for_review(self) -> list[dict]:
        """Return tasks the Developer completed that are awaiting PO acceptance.

        These are tasks with po_status == 'ready_for_review', meaning the
        Developer called complete_task() and the story originated in the
        PO backlog (has a source_story_id).
        """
        with self.state_lock():
            state = self.load()
        return [
            t for t in state.get("tasks", [])
            if t.get("po_status") == "ready_for_review"
        ]

    # ─────────────────────────────────────────────────────────────────────
    #  Sprint loop helpers  (used by sprint_loop.py for autonomous runs)
    # ─────────────────────────────────────────────────────────────────────

    def assign_task(self, task_id: str, assignee_id: str) -> bool:
        """Assign a task to an agent/developer and record the timestamp.

        Returns False when the task_id is not found.
        Only the sprint loop should call this — interactive assignment is
        handled by ScrumMasterAgent._handle_assign().
        """
        with self.state_lock():
            state = self.load()
            for t in state.get("tasks", []):
                if t["id"] == task_id:
                    t["assignee"]    = assignee_id
                    t["assigned_at"] = datetime.now().isoformat(timespec="seconds")
                    self.save(state)
                    return True
        return False

    def reset_for_workflow(self) -> None:
        """Clear stale tasks and backlog before a fresh agile workflow run.

        Called by scrum_master_runner at the start of each workflow-mode sprint
        planning step so that T-NNN tasks and S-NNN backlog items from previous
        unrelated sessions do not contaminate the new project's sprint plan.

        Sprint metadata (id, start, end, velocity) and team capacity are
        preserved — only tasks and backlog are cleared.
        """
        with self.state_lock():
            state = self.load()
            state["tasks"]   = []
            state["backlog"] = []
            if "sprint" in state:
                state["sprint"]["goal"] = "Sprint hedefi henüz tanımlanmamış."
            self.save(state)
        logger.info("sprint-state: cleared for new workflow run")

    def set_sprint_goal(self, goal: str) -> None:
        """Update the sprint goal text.

        Only used by the autonomous sprint loop to stamp the goal into state
        so downstream agents (Developer, ScrumMaster) can read it via
        read_context_block().
        """
        with self.state_lock():
            state = self.load()
            state.setdefault("sprint", {})["goal"] = goal[:200]
            self.save(state)

    # ─────────────────────────────────────────────────────────────────────
    #  Product Goal  (distinct from Sprint Goal)
    # ─────────────────────────────────────────────────────────────────────

    def set_product_goal(self, goal: str) -> None:
        """Set the long-lived Product Goal (distinct from the Sprint Goal)."""
        with self.state_lock():
            state = self.load()
            state["product_goal"] = goal[:500]
            self.save(state)

    def get_product_goal(self) -> str:
        with self.state_lock():
            state = self.load()
        return state.get("product_goal", "")

    # ─────────────────────────────────────────────────────────────────────
    #  Product Increment
    # ─────────────────────────────────────────────────────────────────────

    def add_to_increment(self, task_id: str) -> bool:
        """Record a completed+accepted task as part of the Product Increment.

        Returns False when the task_id is not found in tasks[].
        """
        with self.state_lock():
            state = self.load()
            task_ids = [t["id"] for t in state.get("tasks", [])]
            if task_id not in task_ids:
                return False
            increment: list[str] = state.setdefault("increment", [])
            if task_id not in increment:
                increment.append(task_id)
            self.save(state)
        return True

    def get_increment(self) -> list[str]:
        """Return list of task IDs that form the current Product Increment."""
        with self.state_lock():
            state = self.load()
        return state.get("increment", [])

    # ─────────────────────────────────────────────────────────────────────
    #  Roadmap
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _next_roadmap_id(state: dict[str, Any]) -> str:
        roadmap = state.get("roadmap", [])
        nums = [
            int(e["package_id"].split("-")[1])
            for e in roadmap
            if re.match(r"PKG-\d+", e.get("package_id", ""))
        ] or [0]
        return f"PKG-{(max(nums) + 1):03d}"

    def add_roadmap_entry(self, entry: dict[str, Any]) -> str:
        """Persist a roadmap entry. Assigns a PKG-NNN id if missing.

        Returns the package_id.
        """
        with self.state_lock():
            state = self.load()
            roadmap: list[dict] = state.setdefault("roadmap", [])
            if not entry.get("package_id"):
                entry = {**entry, "package_id": self._next_roadmap_id(state)}
            roadmap.append(entry)
            self.save(state)
        return entry["package_id"]

    def get_roadmap(self) -> list[dict[str, Any]]:
        with self.state_lock():
            state = self.load()
        return state.get("roadmap", [])

    # ─────────────────────────────────────────────────────────────────────
    #  Meeting Notes
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _next_note_id(state: dict[str, Any]) -> str:
        notes = state.get("meeting_notes", [])
        nums = [
            int(n["note_id"].split("-")[1])
            for n in notes
            if re.match(r"MN-\d+", n.get("note_id", ""))
        ] or [0]
        return f"MN-{(max(nums) + 1):03d}"

    def add_meeting_note(self, note: dict[str, Any]) -> str:
        """Persist a structured meeting note. Returns the note_id."""
        with self.state_lock():
            state = self.load()
            notes: list[dict] = state.setdefault("meeting_notes", [])
            note_id = self._next_note_id(state)
            note = {**note, "note_id": note_id}
            notes.append(note)
            self.save(state)
        return note_id

    def get_meeting_notes(
        self, event_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Return all meeting notes, optionally filtered by event_type."""
        with self.state_lock():
            state = self.load()
        notes = state.get("meeting_notes", [])
        if event_type:
            notes = [n for n in notes if n.get("event_type") == event_type]
        return notes

    # ─────────────────────────────────────────────────────────────────────
    #  Retrospective Action Items  (cross-sprint context)
    # ─────────────────────────────────────────────────────────────────────

    def add_retro_actions(self, actions: list[str]) -> None:
        """Append retrospective action items (replacing previous retro actions).

        Replaces rather than appends so that stale actions from two sprints ago
        don't accumulate indefinitely.
        """
        with self.state_lock():
            state = self.load()
            state["retro_actions"] = [a[:200] for a in actions[:10]]
            self.save(state)

    def get_retro_actions(self) -> list[str]:
        """Return action items from the most recent retrospective."""
        with self.state_lock():
            state = self.load()
        return state.get("retro_actions", [])

    # ─────────────────────────────────────────────────────────────────────
    #  Legacy / audit helpers
    # ─────────────────────────────────────────────────────────────────────

    def get_legacy_tasks(self) -> list[dict[str, Any]]:
        """Return tasks that bypass AC enforcement.

        A task is legacy when it has no acceptance_criteria (either
        explicitly flagged via legacy_no_ac=True, or implicitly by
        having an empty acceptance_criteria list). These tasks can
        be marked done without calling mark_ac_validated() — the
        flag makes the bypass visible for auditing and reports.
        """
        with self.state_lock():
            state = self.load()
        return [
            t for t in state.get("tasks", [])
            if t.get("legacy_no_ac") is True
            or not t.get("acceptance_criteria")
        ]
