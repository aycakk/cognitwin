"""pipeline/redo_audit.py — persistent REDO audit log writer.

Appends completed REDO session records to data/audit/redo_audit.jsonl.
Each line is a self-contained JSON object (JSONL format).

Design
------
* Append-only: old records are never modified.
* Thread-safe: a module-level threading.Lock serialises writes within a
  single process.  Across processes (multi-worker uvicorn) writes are
  individually atomic because the OS guarantees that O_APPEND writes of
  ≤ PIPE_BUF bytes are atomic on POSIX; Windows achieves the same via
  WriteFile semantics.  A single JSONL record stays well under 4 KB.
* Silent failure: if the audit file cannot be written (disk full, perms)
  a WARNING is logged but the request continues — audit failure must
  never break the user-facing pipeline.

Schema (one JSON object per line)
----------------------------------
{
  "session_id":  str | null,
  "agent_role":  str,
  "query_hash":  str,          # first 16 chars of sha256(masked_query)
  "redo_count":  int,          # number of REDO cycles triggered
  "limit_hit":   bool,         # True when MAX_REDO was exhausted
  "redo_log":    list[dict],   # full cycle records from redo.py
  "timestamp":   str           # ISO-8601 UTC
}
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_AUDIT_DIR  = Path(__file__).resolve().parents[2] / "data" / "audit"
_AUDIT_FILE = _AUDIT_DIR / "redo_audit.jsonl"
_write_lock = threading.Lock()


def _ensure_dir() -> None:
    try:
        _AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("[REDO AUDIT] Cannot create audit dir %s: %s", _AUDIT_DIR, exc)


def append_session(
    *,
    redo_log: list[dict],
    agent_role: str,
    masked_query: str,
    limit_hit: bool,
    session_id: Optional[str] = None,
) -> None:
    """
    Append a REDO session record to the audit log.

    Should be called once per request, after run_redo_loop returns.
    No-ops silently if redo_log is empty (nothing to audit).
    """
    if not redo_log:
        return  # no REDO cycles — nothing to persist

    _ensure_dir()

    record = {
        "session_id": session_id,
        "agent_role": agent_role,
        "query_hash": hashlib.sha256(masked_query.encode()).hexdigest()[:16],
        "redo_count": len(redo_log),
        "limit_hit":  limit_hit,
        "redo_log":   redo_log,
        "timestamp":  datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    line = json.dumps(record, ensure_ascii=False)

    try:
        with _write_lock:
            with _AUDIT_FILE.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except OSError as exc:
        logger.warning(
            "[REDO AUDIT] Failed to write audit record for role=%s: %s",
            agent_role, exc,
        )
