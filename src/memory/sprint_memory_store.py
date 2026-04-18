"""memory/sprint_memory_store.py — Append-only cross-sprint history store.

Persists sprint snapshots to data/sprint_history.json independently of
sprint_state.json.  Each completed autonomous sprint appends one snapshot.

Architectural invariant:
  - NEVER reads or writes sprint_state.json.
  - SprintStateStore is the sole owner of sprint_state.json.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_HISTORY_PATH = _PROJECT_ROOT / "data" / "sprint_history.json"


class SprintMemoryStore:
    """
    Append-only cross-sprint history.

    Each call to append_sprint() adds one snapshot dict to the history file.
    Reads always return the full list, oldest first.

    Thread-safe via threading.Lock (single-process use assumed for MVP).
    No file-level lock — this store is written only by the sprint loop,
    which runs sequentially.
    """

    def __init__(self, history_path: Path | None = None) -> None:
        self._path = history_path or _DEFAULT_HISTORY_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ─────────────────────────────────────────────────────────────────────────
    #  Write
    # ─────────────────────────────────────────────────────────────────────────

    def append_sprint(self, snapshot: dict[str, Any]) -> None:
        """Append a sprint summary snapshot to the history file.

        Required fields (enforced by sprint_loop.py):
          sprint_id, goal, completed_stories, blocked_stories,
          avg_confidence, step_count
        """
        if "timestamp" not in snapshot:
            snapshot = {**snapshot, "timestamp": datetime.now().isoformat(timespec="seconds")}
        with self._lock:
            history = self._load_raw()
            history.append(snapshot)
            self._path.write_text(
                json.dumps(history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        logger.debug("sprint_memory: appended sprint=%s", snapshot.get("sprint_id"))

    # ─────────────────────────────────────────────────────────────────────────
    #  Read
    # ─────────────────────────────────────────────────────────────────────────

    def load_history(self) -> list[dict[str, Any]]:
        """Return all sprint snapshots, oldest first."""
        with self._lock:
            return self._load_raw()

    def get_last_sprint(self) -> dict[str, Any] | None:
        """Return the most recent sprint snapshot, or None if history is empty."""
        history = self.load_history()
        return history[-1] if history else None

    def get_history_context(self) -> str:
        """Build a compact context string for LLM prompt injection.

        Returns empty string when no history exists (first sprint).
        """
        history = self.load_history()
        if not history:
            return ""
        last = history[-1]
        completed = len(last.get("completed_stories", []))
        blocked = len(last.get("blocked_stories", []))
        conf = last.get("avg_confidence", 0.0)
        return (
            f"Previous sprint: {last.get('sprint_id', 'unknown')} | "
            f"Goal: {str(last.get('goal', '-'))[:60]} | "
            f"Completed: {completed} stories | "
            f"Blocked: {blocked} | "
            f"Avg confidence: {conf:.2f}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal
    # ─────────────────────────────────────────────────────────────────────────

    def _load_raw(self) -> list[dict[str, Any]]:
        """Load history from disk without acquiring the lock."""
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("sprint_memory: failed to load history: %s", exc)
            return []
