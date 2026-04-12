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

import json
import logging
import threading
import warnings
from contextlib import contextmanager
from datetime import date
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
            self.save(DEFAULT_SPRINT_STATE)
            return dict(DEFAULT_SPRINT_STATE)
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return dict(DEFAULT_SPRINT_STATE)

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
