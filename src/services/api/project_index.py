"""services/api/project_index.py — Persistent project metadata index.

Mirrors loop/sprint_index.py: thread-safe read-modify-write JSON file at
data/projects/project_index.json. Never raises on I/O — failures are logged
and swallowed so a disk problem cannot kill an API request.

A project owns zero or more sprints. The aggregate counters
(sprint_count, active_sprint_count, etc.) are a derived cache; the
authoritative source is sprint_index.read_by_project(project_id).
Call recompute_counts() after any sprint mutation to keep them fresh.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_INDEX_PATH = Path("data/projects/project_index.json")
_lock = threading.Lock()

_TERMINAL_PHASES_DONE     = {"complete", "reviewed_complete"}
_TERMINAL_PHASES_PARTIAL  = {"reviewed_partial"}
_TERMINAL_PHASES_BLOCKED  = {"blocked", "cancelled"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_project_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"proj-{ts}-{secrets.token_hex(3)}"


def _load() -> list[dict]:
    try:
        if _INDEX_PATH.exists():
            with open(_INDEX_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("project_index: load failed: %s", exc)
    return []


def _save(entries: list[dict]) -> None:
    try:
        _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _INDEX_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, default=str)
        tmp.replace(_INDEX_PATH)
    except Exception as exc:
        logger.error("project_index: save failed: %s", exc)


def _empty_counts() -> dict:
    return {
        "sprint_count":           0,
        "active_sprint_count":    0,
        "completed_sprint_count": 0,
        "partial_sprint_count":   0,
        "blocked_sprint_count":   0,
        "latest_sprint_id":       "",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CRUD
# ─────────────────────────────────────────────────────────────────────────────

def create(
    *,
    title: str,
    project_goal: str,
    owner_user_id: str,
    owner_user_name: str = "",
    owner_user_email: str = "",
    workspace: str = "",
    description: str = "",
) -> dict:
    """Create a new project record. Returns the stored record."""
    entry = {
        "project_id":       _new_project_id(),
        "owner_user_id":    owner_user_id or "",
        "owner_user_name":  owner_user_name or "",
        "owner_user_email": owner_user_email or "",
        "workspace":        workspace or "",
        "title":            title or "Untitled Project",
        "project_goal":     project_goal or "",
        "description":      description or "",
        "status":           "active",
        "created_at":       _now(),
        "updated_at":       _now(),
        **_empty_counts(),
    }
    with _lock:
        entries = _load()
        entries.insert(0, entry)
        _save(entries)
    return dict(entry)


def get(project_id: str) -> dict | None:
    with _lock:
        for e in _load():
            if e.get("project_id") == project_id:
                return dict(e)
    return None


def read_all(*, archived: bool = False) -> list[dict]:
    """Return all projects. By default hides archived."""
    with _lock:
        entries = _load()
    if not archived:
        entries = [e for e in entries if e.get("status") != "archived"]
    return entries


def update(project_id: str, updates: dict) -> dict | None:
    """Merge updates into a project. Returns the updated record, or None."""
    with _lock:
        entries = _load()
        for e in entries:
            if e.get("project_id") == project_id:
                e.update(updates)
                e["updated_at"] = _now()
                _save(entries)
                return dict(e)
    return None


def archive(project_id: str) -> bool:
    with _lock:
        entries = _load()
        for e in entries:
            if e.get("project_id") == project_id:
                e["status"]     = "archived"
                e["updated_at"] = _now()
                _save(entries)
                return True
    return False


def reopen(project_id: str) -> bool:
    with _lock:
        entries = _load()
        for e in entries:
            if e.get("project_id") == project_id:
                e["status"]     = "active"
                e["updated_at"] = _now()
                _save(entries)
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregate counters (derived from sprint_index)
# ─────────────────────────────────────────────────────────────────────────────

def _bucket(phase: str, status: str | None = None) -> str:
    """Bucket a sprint into one of: active | completed | partial | blocked.

    Phase wins over status. Anything not in a terminal phase is "active"
    regardless of run-status (planning, running, waiting_review, etc.).
    """
    p = (phase or "").strip().lower()
    s = (status or "").strip().lower()
    if p in _TERMINAL_PHASES_PARTIAL:
        return "partial"
    if p in _TERMINAL_PHASES_BLOCKED or s == "error":
        return "blocked"
    if p in _TERMINAL_PHASES_DONE:
        return "completed"
    return "active"


def recompute_counts(project_id: str) -> dict | None:
    """Recompute aggregate sprint counters for a project from sprint_index.

    Returns the updated project record, or None if not found.
    """
    from src.loop import sprint_index  # local import — avoid cycle at module load
    sprints = sprint_index.read_by_project(project_id)

    counts = _empty_counts()
    counts["sprint_count"] = len(sprints)
    latest_sprint_id = ""
    latest_ts        = ""
    for s in sprints:
        bucket = _bucket(s.get("phase"), s.get("status"))
        if bucket == "completed":
            counts["completed_sprint_count"] += 1
        elif bucket == "partial":
            counts["partial_sprint_count"] += 1
        elif bucket == "blocked":
            counts["blocked_sprint_count"] += 1
        else:
            counts["active_sprint_count"] += 1
        ts = s.get("updated_at") or s.get("created_at") or ""
        if ts > latest_ts:
            latest_ts        = ts
            latest_sprint_id = s.get("sprint_id") or ""
    counts["latest_sprint_id"] = latest_sprint_id

    return update(project_id, counts)
