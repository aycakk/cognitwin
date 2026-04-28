"""pipeline/hr/audit_logger.py — Immutable audit trail for HR agent actions.

Each HR action is appended to a JSONL file (one JSON object per line).
JSONL is append-only, human-readable, and trivially parseable.

Storage: data/hr_profiles/audit_<recruiter_id>.jsonl
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DATA_DIR = Path("data/hr_profiles")
_lock = threading.Lock()


def _audit_path(recruiter_id: str) -> Path:
    return _DATA_DIR / f"audit_{recruiter_id}.jsonl"


def _ensure_dir() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def log_action(
    recruiter_id: str,
    action: str,
    session_id: str = "",
    details: dict[str, Any] | None = None,
    token_cost: int = 0,
    token_remaining: int = 0,
    result_summary: str = "",
) -> None:
    """Append one audit record.  Never raises — failures are logged and swallowed."""
    _ensure_dir()
    entry = {
        "ts":              datetime.now(timezone.utc).isoformat(),
        "recruiter_id":    recruiter_id,
        "session_id":      session_id,
        "action":          action,
        "token_cost":      token_cost,
        "token_remaining": token_remaining,
        "result_summary":  result_summary[:200],
        "details":         details or {},
    }
    try:
        with _lock:
            with open(_audit_path(recruiter_id), "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.error("audit write failed: %s", exc)


def read_audit_tail(recruiter_id: str, n: int = 20) -> list[dict]:
    """Return the last n audit entries for a recruiter (for transparency reports)."""
    path = _audit_path(recruiter_id)
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        tail = lines[-n:] if len(lines) >= n else lines
        return [json.loads(l) for l in tail]
    except Exception as exc:
        logger.warning("audit read failed: %s", exc)
        return []
