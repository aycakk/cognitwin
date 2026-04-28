"""services/api/sprint_run_registry.py — in-memory registry of cockpit sprint runs.

Tracks each sprint started from the Autonomous Scrum Cockpit UI:
  status, events, phase, started_at.

Thread-safe. One entry per sprint_id. SprintStateStore is still the single
owner of task/backlog state; this registry only tracks cockpit-specific
live data (event stream, run status).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class SprintRunEntry:
    sprint_id: str
    goal: str
    status: str          # "started" | "running" | "complete" | "blocked" | "error"
    phase: str           # "planning" | "executing" | "complete" | "blocked"
    events: list         # list[dict] — event stream
    started_at: str      # ISO timestamp

    def as_dict(self) -> dict:
        return {
            "sprint_id":  self.sprint_id,
            "goal":       self.goal,
            "status":     self.status,
            "phase":      self.phase,
            "started_at": self.started_at,
        }


_registry: dict[str, SprintRunEntry] = {}
_lock: threading.Lock = threading.Lock()


def register(sprint_id: str, goal: str) -> SprintRunEntry:
    entry = SprintRunEntry(
        sprint_id  = sprint_id,
        goal       = goal,
        status     = "started",
        phase      = "planning",
        events     = [],
        started_at = datetime.now(timezone.utc).isoformat(),
    )
    with _lock:
        _registry[sprint_id] = entry
    return entry


def get(sprint_id: str) -> Optional[SprintRunEntry]:
    with _lock:
        return _registry.get(sprint_id)


def get_latest() -> Optional[SprintRunEntry]:
    """Return the most recently registered sprint run (for single-sprint UX)."""
    with _lock:
        if not _registry:
            return None
        return list(_registry.values())[-1]


def append_event(
    sprint_id: str,
    agent: str,
    event_type: str,
    message: str,
    task_id: Optional[str] = None,
) -> None:
    """Append a cockpit event to the sprint's event list.

    agent     : "system" | "po" | "sm" | "dev" | "gate"
    event_type: "thought" | "action" | "ceremony" | "status" | "artifact" | "gate"
    """
    ts = datetime.now(timezone.utc).isoformat()
    event = {
        "ts":      ts,
        "agent":   agent,
        "type":    event_type,
        "message": message,
    }
    if task_id:
        event["task_id"] = task_id

    with _lock:
        entry = _registry.get(sprint_id)
        if entry is not None:
            entry.events.append(event)

            # Derive phase from event signals
            if event_type == "ceremony" and "planning" in message.lower():
                entry.phase = "planning"
            elif agent == "dev" and event_type in ("thought", "artifact", "action"):
                if entry.phase not in ("complete", "blocked"):
                    entry.phase = "executing"
            elif agent == "system" and event_type == "status":
                if "complete" in message.lower():
                    entry.phase = "complete"
                elif "blocked" in message.lower():
                    entry.phase = "blocked"


def set_status(sprint_id: str, status: str, phase: Optional[str] = None) -> None:
    with _lock:
        entry = _registry.get(sprint_id)
        if entry is not None:
            entry.status = status
            if phase:
                entry.phase = phase


def get_events(sprint_id: str) -> list:
    with _lock:
        entry = _registry.get(sprint_id)
        return list(entry.events) if entry else []
