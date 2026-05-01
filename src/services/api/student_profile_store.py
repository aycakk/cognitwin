"""services/api/student_profile_store.py — single-profile Lumi student dashboard store.

Persists the Lumi dashboard state to one JSON file so assignment status, mood,
and other small mutations survive process restarts. Single profile — no
role/account logic — because the Lumi product spec is explicit about that.

Storage path: $COGNITWIN_DATA_DIR/student/profile.json (default: data/student/profile.json).
The COGNITWIN_DATA_DIR override exists so a Docker/runtime deployment can mount
a writable volume at an absolute path.

Locking + tmp-file swap mirrors product_backlog_store.py.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Paths (module-level so tests can monkeypatch)
# ─────────────────────────────────────────────────────────────────────────────

_DATA_DIR = Path(os.environ.get("COGNITWIN_DATA_DIR", "data"))
_PROFILE_PATH = _DATA_DIR / "student" / "profile.json"
_lock = threading.Lock()


VALID_STATUSES = ("Pending", "Planned", "In Progress", "Completed", "Submitted")
VALID_MOODS = ("good", "ok", "meh", "tired", "stressed")


# ─────────────────────────────────────────────────────────────────────────────
#  Seed
# ─────────────────────────────────────────────────────────────────────────────

def _seed_profile() -> dict:
    """Seed data — all English. Dates expressed as days_offset from today so
    the dashboard never goes stale. Schedule items are time-of-day only."""
    return {
        "schema_version": 1,
        "updated_at": _now_iso(),
        "student_name": "Deniz",
        "mood": None,

        # Time-of-day only; no calendar date stored.
        "today_schedule": [
            {"id": "ts-1", "start": "09:00", "end": "10:15", "title": "Mathematics class",
             "detail": "Derivatives Q&A",          "kind": "lesson"},
            {"id": "ts-2", "start": "10:30", "end": "12:00", "title": "Physics lab",
             "detail": "Friction experiment",      "kind": "lesson"},
            {"id": "ts-3", "start": "13:00", "end": "14:00", "title": "English class",
             "detail": "Essay techniques",         "kind": "lesson"},
            {"id": "ts-4", "start": "15:30", "end": "16:30", "title": "History presentation prep",
             "detail": "Republic Era",             "kind": "assignment"},
            {"id": "ts-5", "start": "17:00", "end": "17:25", "title": "Mathematics review",
             "detail": "25 min derivatives",       "kind": "study"},
            {"id": "ts-6", "start": "20:00", "end": "21:00", "title": "Swim practice",
             "detail": "",                         "kind": "personal"},
        ],

        "courses": [
            {"id": "c-math",    "name": "Mathematics",       "teacher": "Mr. Aylin",
             "average": 72, "target": 85,
             "insight": "Derivatives look weak — 25-min review suggested."},
            {"id": "c-phys",    "name": "Physics",           "teacher": "Mr. Murat",
             "average": 68, "target": 80,
             "insight": "Friction concepts need a second pass."},
            {"id": "c-eng",     "name": "English",           "teacher": "Ms. Clarke",
             "average": 88, "target": 90,
             "insight": "Strong reading scores — push essay structure."},
            {"id": "c-hist",    "name": "History",           "teacher": "Ms. Selin",
             "average": 81, "target": 85,
             "insight": "Republic Era presentation prep is the priority."},
            {"id": "c-cs",      "name": "Computer Science",  "teacher": "Mr. Emre",
             "average": 92, "target": 95,
             "insight": "On track — keep current pace."},
        ],

        # days_offset relative to today; resolved to absolute date at read time.
        "assignments": [
            {"id": "a-1", "title": "Republic Era presentation", "course_id": "c-hist",
             "days_offset": 1,  "priority": "high",   "status": "Pending"},
            {"id": "a-2", "title": "Derivative problem set 4",  "course_id": "c-math",
             "days_offset": 3,  "priority": "high",   "status": "Pending"},
            {"id": "a-3", "title": "Friction lab report",       "course_id": "c-phys",
             "days_offset": 5,  "priority": "medium", "status": "Pending"},
            {"id": "a-4", "title": "Climate change essay",      "course_id": "c-eng",
             "days_offset": 7,  "priority": "high",   "status": "Pending"},
            {"id": "a-5", "title": "JS project todo app",       "course_id": "c-cs",
             "days_offset": 11, "priority": "low",    "status": "Submitted"},
        ],

        "exams": [
            {"id": "e-1", "title": "Mathematics second written", "course_id": "c-math",
             "days_offset": 3,  "topics": ["Derivatives", "Limits"]},
            {"id": "e-2", "title": "Physics quiz",               "course_id": "c-phys",
             "days_offset": 9,  "topics": ["Newton's laws", "Friction"]},
            {"id": "e-3", "title": "English midterm",            "course_id": "c-eng",
             "days_offset": 16, "topics": ["Reading", "Essay writing"]},
        ],

        "grades": [
            {"course_id": "c-math",  "title": "Quiz 1",      "score": 75},
            {"course_id": "c-math",  "title": "First written","score": 68},
            {"course_id": "c-phys",  "title": "Lab report 1", "score": 72},
            {"course_id": "c-phys",  "title": "First written","score": 65},
            {"course_id": "c-eng",   "title": "Essay 1",      "score": 85},
            {"course_id": "c-eng",   "title": "Vocab quiz",   "score": 92},
            {"course_id": "c-hist",  "title": "Presentation", "score": 84},
            {"course_id": "c-hist",  "title": "First written","score": 78},
            {"course_id": "c-cs",    "title": "Project 1",    "score": 95},
        ],

        # Optional persisted weekly stat — not auto-computed.
        "weekly_study_hours": 12,
        "weekly_study_delta": 3,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Internals
# ─────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir() -> None:
    parent = _PROFILE_PATH.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"student_profile_store: cannot create data dir {parent} ({exc}). "
            f"Set COGNITWIN_DATA_DIR to a writable path."
        ) from exc
    if not os.access(parent, os.W_OK):
        raise RuntimeError(
            f"student_profile_store: data dir {parent} is not writable. "
            f"Set COGNITWIN_DATA_DIR to a writable path."
        )


def _read_disk() -> dict | None:
    if not _PROFILE_PATH.exists():
        return None
    try:
        with open(_PROFILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logger.warning("student_profile_store: load failed (%s); reseeding.", exc)
    return None


def _write_disk(data: dict) -> None:
    _ensure_dir()
    data["updated_at"] = _now_iso()
    tmp = _PROFILE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    tmp.replace(_PROFILE_PATH)


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_profile() -> dict:
    """Return the persisted profile, seeding it on first call."""
    with _lock:
        data = _read_disk()
        if data is None:
            data = _seed_profile()
            try:
                _write_disk(data)
            except RuntimeError as exc:
                logger.error("%s", exc)
                # Still return in-memory seed so the API can respond.
        return data


def save_profile(profile: dict) -> dict:
    with _lock:
        _write_disk(profile)
        return profile


def update_assignment_status(assignment_id: str, status: str) -> dict | None:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {status!r}")
    with _lock:
        data = _read_disk() or _seed_profile()
        for a in data.get("assignments", []):
            if a.get("id") == assignment_id:
                a["status"] = status
                _write_disk(data)
                return dict(a)
    return None


def set_mood(mood: str) -> dict:
    if mood not in VALID_MOODS:
        raise ValueError(f"invalid mood: {mood!r}")
    with _lock:
        data = _read_disk() or _seed_profile()
        data["mood"] = mood
        _write_disk(data)
        return {"mood": mood}


# ─────────────────────────────────────────────────────────────────────────────
#  Date materialisation — called by routes before serializing to the client.
# ─────────────────────────────────────────────────────────────────────────────

def materialize_dates(profile: dict, today: date | None = None) -> dict:
    """Return a copy of profile with absolute due_date / days_left fields
    computed from days_offset. The on-disk seed is never mutated."""
    today = today or date.today()
    out = json.loads(json.dumps(profile, default=str))   # deep copy

    for collection in ("assignments", "exams"):
        for item in out.get(collection, []):
            try:
                offset = int(item.get("days_offset", 0))
            except (TypeError, ValueError):
                offset = 0
            due = today + timedelta(days=offset)
            item["due_date"] = due.isoformat()
            item["days_left"] = offset

    # Deadlines list = upcoming assignments + exams sorted by days_left.
    deadlines: list[dict] = []
    for a in out.get("assignments", []):
        if a.get("status") in ("Completed", "Submitted"):
            continue
        deadlines.append({
            "kind": "assignment",
            "title": a["title"],
            "course_id": a.get("course_id"),
            "days_left": a["days_left"],
            "due_date": a["due_date"],
        })
    for e in out.get("exams", []):
        deadlines.append({
            "kind": "exam",
            "title": e["title"],
            "course_id": e.get("course_id"),
            "days_left": e["days_left"],
            "due_date": e["due_date"],
        })
    deadlines.sort(key=lambda d: d["days_left"])
    out["deadlines"] = deadlines

    # Today's date stamp for the header.
    out["today"] = today.isoformat()

    # Pending count for the summary card.
    out["pending_count"] = sum(
        1 for a in out.get("assignments", [])
        if a.get("status") not in ("Completed", "Submitted")
    ) + len(out.get("exams", []))

    return out
