"""services/api/product_backlog_store.py — per-project Product Backlog persistence.

One JSON file per project at data/projects/product_backlogs/{project_id}.json.
Mirrors the locking + swallow-on-IO-failure pattern of project_index.py.

Item lifecycle:
    new        — fresh, never committed
    selected   — chosen for next planning (transient)
    in_sprint  — committed to a running sprint
    done       — accepted in a sprint
    deferred   — carried over from an unfinished sprint

Sources: po_llm | user | carry_over | sprint_notes
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Module-level path so tests can patch it to tmp_path.
_BACKLOG_DIR = Path("data/projects/product_backlogs")
_lock = threading.Lock()

VALID_STATUSES = {"new", "selected", "in_sprint", "done", "deferred"}
VALID_PRIORITIES = {"high", "medium", "low"}
VALID_SOURCES = {"po_llm", "user", "carry_over", "sprint_notes"}

# Statuses that block deletion (item is in flight or finished).
_LOCKED_STATUSES = {"in_sprint", "done"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path_for(project_id: str) -> Path:
    return _BACKLOG_DIR / f"{project_id}.json"


def _empty_backlog(project_id: str) -> dict:
    return {
        "project_id": project_id,
        "updated_at": _now(),
        "items":      [],
    }


def _load(project_id: str) -> dict:
    path = _path_for(project_id)
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("items"), list):
                return data
    except Exception as exc:
        logger.warning("product_backlog_store: load failed for %s: %s", project_id, exc)
    return _empty_backlog(project_id)


def _save(project_id: str, data: dict) -> None:
    path = _path_for(project_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data["updated_at"] = _now()
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        tmp.replace(path)
    except Exception as exc:
        logger.error("product_backlog_store: save failed for %s: %s", project_id, exc)


def _next_item_id(items: list[dict]) -> str:
    """Generate PB-NNN ID, monotonic per backlog file."""
    max_n = 0
    for it in items:
        iid = (it.get("item_id") or "")
        if iid.startswith("PB-"):
            try:
                n = int(iid.split("-", 1)[1])
                if n > max_n:
                    max_n = n
            except (ValueError, IndexError):
                continue
    return f"PB-{max_n + 1:03d}"


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def exists(project_id: str) -> bool:
    return _path_for(project_id).exists()


def read(project_id: str) -> dict:
    """Return the full backlog dict (always returns a dict, never None)."""
    with _lock:
        return _load(project_id)


def write_items(project_id: str, items: list[dict], *, source: str = "po_llm") -> dict:
    """Replace the entire item list. Used by seed.

    Each input item is normalised; missing IDs are auto-assigned.
    """
    src = source if source in VALID_SOURCES else "po_llm"
    normalised: list[dict] = []
    with _lock:
        data = _load(project_id)
        existing_ids = {(it.get("item_id") or "") for it in data.get("items", [])}
        for raw in items:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title", "")).strip()[:120]
            if not title:
                continue
            iid = raw.get("item_id") or _next_item_id(normalised + data.get("items", []))
            # Avoid colliding with already-loaded items if caller didn't pass IDs.
            while iid in existing_ids:
                existing_ids.add(iid)
                iid = _next_item_id(normalised + data.get("items", []) + [{"item_id": iid}])
            existing_ids.add(iid)
            ac_raw = raw.get("acceptance_criteria") or []
            ac = [str(c)[:200] for c in ac_raw if isinstance(c, str) and c.strip()][:8]
            priority = str(raw.get("priority", "medium")).strip().lower()
            if priority not in VALID_PRIORITIES:
                priority = "medium"
            try:
                points = max(1, min(13, int(raw.get("story_points", 2))))
            except (TypeError, ValueError):
                points = 2
            status = str(raw.get("status", "new")).strip().lower()
            if status not in VALID_STATUSES:
                status = "new"
            normalised.append({
                "item_id":             iid,
                "title":               title,
                "description":         str(raw.get("description", ""))[:400],
                "acceptance_criteria": ac,
                "priority":            priority,
                "story_points":        points,
                "epic":                str(raw.get("epic", ""))[:80],
                "status":              status,
                "source":              str(raw.get("source", src)),
                "committed_in_sprint": raw.get("committed_in_sprint") or "",
                "created_at":          raw.get("created_at") or _now(),
                "updated_at":          _now(),
            })
        data["items"] = normalised
        _save(project_id, data)
        return data


def add_item(project_id: str, item: dict, *, source: str = "user") -> dict:
    """Append a single item. Returns the stored item (with generated ID)."""
    src = source if source in VALID_SOURCES else "user"
    with _lock:
        data = _load(project_id)
        title = str(item.get("title", "")).strip()[:120]
        if not title:
            raise ValueError("title must not be empty")
        ac_raw = item.get("acceptance_criteria") or []
        ac = [str(c)[:200] for c in ac_raw if isinstance(c, str) and c.strip()][:8]
        priority = str(item.get("priority", "medium")).strip().lower()
        if priority not in VALID_PRIORITIES:
            priority = "medium"
        try:
            points = max(1, min(13, int(item.get("story_points", 2))))
        except (TypeError, ValueError):
            points = 2
        status = str(item.get("status", "new")).strip().lower()
        if status not in VALID_STATUSES:
            status = "new"
        entry = {
            "item_id":             _next_item_id(data["items"]),
            "title":               title,
            "description":         str(item.get("description", ""))[:400],
            "acceptance_criteria": ac,
            "priority":            priority,
            "story_points":        points,
            "epic":                str(item.get("epic", ""))[:80],
            "status":              status,
            "source":              src,
            "committed_in_sprint": "",
            "created_at":          _now(),
            "updated_at":          _now(),
        }
        data["items"].append(entry)
        _save(project_id, data)
        return dict(entry)


def update_item(project_id: str, item_id: str, updates: dict) -> dict | None:
    """Patch a single item's mutable fields. Returns updated item or None."""
    allowed = {
        "title", "description", "acceptance_criteria", "priority",
        "story_points", "epic", "status", "committed_in_sprint",
    }
    with _lock:
        data = _load(project_id)
        for it in data["items"]:
            if it.get("item_id") == item_id:
                for k, v in updates.items():
                    if k not in allowed:
                        continue
                    if k == "priority":
                        v = str(v).strip().lower()
                        if v not in VALID_PRIORITIES:
                            continue
                    elif k == "status":
                        v = str(v).strip().lower()
                        if v not in VALID_STATUSES:
                            continue
                    elif k == "story_points":
                        try:
                            v = max(1, min(13, int(v)))
                        except (TypeError, ValueError):
                            continue
                    elif k == "acceptance_criteria":
                        if not isinstance(v, list):
                            continue
                        v = [str(c)[:200] for c in v if isinstance(c, str) and c.strip()][:8]
                    elif k in ("title", "description", "epic", "committed_in_sprint"):
                        v = str(v)[:400]
                    it[k] = v
                it["updated_at"] = _now()
                _save(project_id, data)
                return dict(it)
    return None


def delete_item(project_id: str, item_id: str) -> str:
    """Delete an item. Returns 'deleted' | 'not_found' | 'locked'.

    Items in `in_sprint` or `done` cannot be deleted (would corrupt sprint history).
    """
    with _lock:
        data = _load(project_id)
        for i, it in enumerate(data["items"]):
            if it.get("item_id") == item_id:
                if it.get("status") in _LOCKED_STATUSES:
                    return "locked"
                data["items"].pop(i)
                _save(project_id, data)
                return "deleted"
    return "not_found"


def get_item(project_id: str, item_id: str) -> dict | None:
    with _lock:
        for it in _load(project_id)["items"]:
            if it.get("item_id") == item_id:
                return dict(it)
    return None


def items_by_ids(project_id: str, item_ids: list[str]) -> list[dict]:
    """Return items matching the given IDs, preserving the request order."""
    if not item_ids:
        return []
    wanted = list(item_ids)
    with _lock:
        all_items = {it.get("item_id"): dict(it) for it in _load(project_id)["items"]}
    return [all_items[i] for i in wanted if i in all_items]


def mark_in_sprint(project_id: str, item_ids: list[str], sprint_id: str) -> int:
    """Bulk-flip status to 'in_sprint' for the listed items. Returns count updated."""
    if not item_ids:
        return 0
    wanted = set(item_ids)
    count = 0
    with _lock:
        data = _load(project_id)
        for it in data["items"]:
            if it.get("item_id") in wanted:
                it["status"] = "in_sprint"
                it["committed_in_sprint"] = sprint_id
                it["updated_at"] = _now()
                count += 1
        if count:
            _save(project_id, data)
    return count
