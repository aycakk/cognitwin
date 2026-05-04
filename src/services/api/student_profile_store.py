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

CURRENT_SCHEMA_VERSION = 2

# Sliding window for procrastination signals.
_PROCRASTINATION_WINDOW_DAYS = 7

# Adaptive block-length floor — Lumi never proposes blocks shorter than this.
_MIN_BLOCK_MINUTES = 15

# Recent plan history cap.
_PLAN_HISTORY_MAX = 10


def _default_personalization() -> dict:
    return {
        "preferred_study_block_minutes": 30,
        "preferred_study_time": None,
        "weak_topics": [],
        "resource_preferences": {
            "video": 0, "textbook": 0, "practice": 0, "reference": 0,
        },
        "postpone_timestamps": [],
        "shortening_timestamps": [],
        "accepted_suggestions_count": 0,
        "postponed_suggestions_count": 0,
        "shortened_plan_count": 0,
        "completed_plan_count": 0,
        "last_feedback": None,
        "focus_points": 0,
        "streak_days": 0,
        "last_active_date": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Seed
# ─────────────────────────────────────────────────────────────────────────────

def _seed_profile() -> dict:
    """Seed data — all English. Dates expressed as days_offset from today so
    the dashboard never goes stale. Schedule items are time-of-day only."""
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "updated_at": _now_iso(),
        "student_name": "Deniz",
        "mood": None,
        "personalization": _default_personalization(),
        "recent_plan_history": [],
        "last_plan": None,

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
            return _migrate(data)
    except Exception as exc:
        logger.warning("student_profile_store: load failed (%s); reseeding.", exc)
    return None


def _migrate(data: dict) -> dict:
    """Forward-compatible read-side migration. Fills missing v2 fields with
    defaults but does not rewrite the file — that happens on the next save."""
    if not isinstance(data.get("personalization"), dict):
        data["personalization"] = _default_personalization()
    else:
        defaults = _default_personalization()
        for k, v in defaults.items():
            data["personalization"].setdefault(k, v)
        # Nested resource_preferences.
        rp_defaults = defaults["resource_preferences"]
        rp = data["personalization"].setdefault("resource_preferences", {})
        for k, v in rp_defaults.items():
            rp.setdefault(k, v)
    data.setdefault("recent_plan_history", [])
    data.setdefault("last_plan", None)
    data["schema_version"] = CURRENT_SCHEMA_VERSION
    return data


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
#  Personalization mutators — Phase 1 of the personal-agent layer.
# ─────────────────────────────────────────────────────────────────────────────

def _today() -> date:
    return date.today()


def _bump_streak(personalization: dict) -> None:
    """Update streak_days based on last_active_date relative to today."""
    today_obj = _today()
    today_iso = today_obj.isoformat()
    last = personalization.get("last_active_date")
    if last == today_iso:
        return
    if last is None:
        personalization["streak_days"] = 1
    else:
        try:
            prev = date.fromisoformat(last)
        except (TypeError, ValueError):
            prev = None
        if prev is not None and (today_obj - prev).days == 1:
            personalization["streak_days"] = int(personalization.get("streak_days", 0)) + 1
        else:
            personalization["streak_days"] = 1
    personalization["last_active_date"] = today_iso


def _trim_window(timestamps: list, today: date) -> list:
    """Keep only ISO-date strings within the last N days."""
    cutoff = today - timedelta(days=_PROCRASTINATION_WINDOW_DAYS)
    out: list[str] = []
    for ts in timestamps or []:
        try:
            d = date.fromisoformat(ts[:10])
        except (TypeError, ValueError):
            continue
        if d >= cutoff:
            out.append(ts)
    return out


def apply_chat_signal(action: str | None, last_plan: dict | None = None) -> dict:
    """Apply the side-effects of a chat action (chip click) to personalization.

    Returns the updated profile so callers can keep their in-memory copy fresh.
    Also records `last_plan` (the plan returned to the student) so the
    dashboard can render a "Why this plan?" card.
    """
    with _lock:
        data = _read_disk() or _seed_profile()
        p = data.setdefault("personalization", _default_personalization())
        today_iso = _today().isoformat()

        _bump_streak(p)

        if action == "shorter":
            p["shortened_plan_count"] = int(p.get("shortened_plan_count", 0)) + 1
            ts_list = p.setdefault("shortening_timestamps", [])
            ts_list.append(today_iso)
            p["shortening_timestamps"] = _trim_window(ts_list, _today())
            current = int(p.get("preferred_study_block_minutes", 30))
            p["preferred_study_block_minutes"] = max(_MIN_BLOCK_MINUTES, current - 5)
        elif action == "postpone":
            p["postponed_suggestions_count"] = int(p.get("postponed_suggestions_count", 0)) + 1
            ts_list = p.setdefault("postpone_timestamps", [])
            ts_list.append(today_iso)
            p["postpone_timestamps"] = _trim_window(ts_list, _today())
        elif action == "resources":
            rp = p.setdefault("resource_preferences", {})
            rp["reference"] = int(rp.get("reference", 0)) + 1
        elif action in ("plan", "calendar"):
            p["accepted_suggestions_count"] = int(p.get("accepted_suggestions_count", 0)) + 1

        if last_plan is not None:
            data["last_plan"] = last_plan
            history = data.setdefault("recent_plan_history", [])
            history.insert(0, last_plan)
            del history[_PLAN_HISTORY_MAX:]

        _write_disk(data)
        return data


def complete_plan(plan_id: str | None = None) -> dict:
    """Mark the current plan complete: bumps focus points, completion count,
    and streak. Returns the updated profile."""
    with _lock:
        data = _read_disk() or _seed_profile()
        p = data.setdefault("personalization", _default_personalization())

        _bump_streak(p)
        p["completed_plan_count"] = int(p.get("completed_plan_count", 0)) + 1
        p["focus_points"] = int(p.get("focus_points", 0)) + 5
        if int(p.get("streak_days", 0)) >= 5:
            p["focus_points"] += 10  # weekly bonus

        last_plan = data.get("last_plan")
        if last_plan is not None and isinstance(last_plan, dict):
            last_plan["outcome"] = "completed"
            last_plan["completed_at"] = _now_iso()
            if plan_id and last_plan.get("id") and plan_id != last_plan["id"]:
                # Caller passed a stale id; still mark current as completed.
                pass

        _write_disk(data)
        return data


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

    # Personalization derived signals — sliding window counts + weak topics.
    p = out.get("personalization") or _default_personalization()
    p["postpones_last_7d"] = len(_trim_window(p.get("postpone_timestamps", []), today))
    p["shortenings_last_7d"] = len(_trim_window(p.get("shortening_timestamps", []), today))

    # Weak topics: lowest 2 courses by average. (v1: course names, not subtopics.)
    courses_sorted = sorted(
        out.get("courses", []),
        key=lambda c: c.get("average", 100),
    )
    p["weak_topics"] = [c["name"] for c in courses_sorted[:2]] if courses_sorted else []

    out["personalization"] = p
    return out
