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
    "tasks": [],
    "backlog": [],
    "team": [
        {"id": "developer-default", "role": "Developer", "capacity": 8},
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
    ) -> str:
        """Append a new story to the backlog. Returns the story_id."""
        with self.state_lock():
            state = self.load()
            story_id = self._next_story_id(state)
            story: dict[str, Any] = {
                "story_id":            story_id,
                "title":               title[:120],
                "description":         description[:500],
                "priority":            priority if priority in ("high", "medium", "low") else "medium",
                "acceptance_criteria": acceptance_criteria or [],
                "source_request":      source_request[:200],
                "status":              "draft",
                "created_at":          datetime.now().isoformat(timespec="seconds"),
                "updated_at":          None,
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
        allowed = {"title", "description", "priority", "acceptance_criteria", "status"}
        with self.state_lock():
            state = self.load()
            for s in state.get("backlog", []):
                if s.get("story_id") == sid:
                    for k, v in fields.items():
                        if k in allowed:
                            s[k] = v
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
                "acceptance_criteria": story.get("acceptance_criteria", []),
                "source_story_id":     story_id,
                "po_status":           "pending_review",
                "created_at":          datetime.now().isoformat(timespec="seconds"),
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

    def complete_task(self, task_id: str, result_summary: str = "") -> bool:
        """Mark a task as done and record the developer's result summary.

        When the task carries a source_story_id (i.e. it was promoted from
        the PO backlog), the po_status field is set to 'ready_for_review'
        so the PO can inspect and accept or reject the linked story.

        Returns False when the task_id is not found.
        """
        with self.state_lock():
            state = self.load()
            for t in state.get("tasks", []):
                if t["id"] == task_id:
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
