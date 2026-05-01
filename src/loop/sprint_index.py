"""loop/sprint_index.py — Persistent sprint metadata index.

Writes sprint records to data/sprints/sprint_index.json so sprint history
survives process restarts.  Thread-safe.  Never raises — all I/O errors are
logged and swallowed so a disk problem never kills a sprint.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_INDEX_PATH = Path("data/sprints/sprint_index.json")
_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load() -> list[dict]:
    try:
        if _INDEX_PATH.exists():
            with open(_INDEX_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("sprint_index: load failed: %s", exc)
    return []


def _save(entries: list[dict]) -> None:
    try:
        _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _INDEX_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, default=str)
        tmp.replace(_INDEX_PATH)
    except Exception as exc:
        logger.error("sprint_index: save failed: %s", exc)


def write_entry(entry: dict) -> None:
    """Insert or update a sprint index record (upsert by sprint_id)."""
    with _lock:
        entries = _load()
        sid = entry.get("sprint_id")
        for existing in entries:
            if existing.get("sprint_id") == sid:
                existing.update(entry)
                _save(entries)
                return
        entries.insert(0, entry)  # newest first
        _save(entries)


def update_entry(sprint_id: str, updates: dict) -> None:
    """Merge delta updates into an existing record."""
    with _lock:
        entries = _load()
        for e in entries:
            if e.get("sprint_id") == sprint_id:
                e.update(updates)
                e["updated_at"] = _now()
                _save(entries)
                return
        logger.warning("sprint_index: update_entry — sprint %s not found", sprint_id)


def read_all() -> list[dict]:
    """Return all sprint index records, newest first."""
    with _lock:
        return list(_load())


def get_entry(sprint_id: str) -> dict | None:
    """Return a single sprint record by sprint_id, or None if not found."""
    with _lock:
        for e in _load():
            if e.get("sprint_id") == sprint_id:
                return dict(e)
    return None


def read_by_project(project_id: str) -> list[dict]:
    """Return all sprint records belonging to a given project_id, newest first.

    A sprint with no project_id field (legacy) never matches a non-empty query.
    Pass an empty string to retrieve legacy sprints.
    """
    with _lock:
        entries = _load()
    if project_id:
        return [e for e in entries if e.get("project_id") == project_id]
    return [e for e in entries if not e.get("project_id")]


def archive_sprint(sprint_id: str) -> bool:
    """Mark a sprint as archived.  Returns True if the sprint was found."""
    with _lock:
        entries = _load()
        for e in entries:
            if e.get("sprint_id") == sprint_id:
                e["archived"] = True
                e["updated_at"] = _now()
                _save(entries)
                return True
    return False
