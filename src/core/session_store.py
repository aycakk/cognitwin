"""core/session_store.py — Per-agent session registry.

Purpose
───────
Track each agent's work as a separate inspectable session.  When the
Agile workflow runs (PO → SM → Developer → Composer), each step gets its
own session ID and its output is stored here.  The user can later query
any agent session to see exactly what that agent produced.

Session ID scheme
─────────────────
  Parent (user conversation):  <conversation_id>
  PO agent session:            <conversation_id>/po
  SM agent session:            <conversation_id>/sm
  Developer agent session:     <conversation_id>/dev
  Composer session:            <conversation_id>   (same as parent)

Storage
───────
  In-memory:  thread-safe dict for fast reads during a request
  On disk:    data/sessions/<sanitized_id>.json  (one file per session)
              Written on every record_output() call.
              Survives restarts — the user can always inspect past work.

Thread safety
─────────────
  A single module-level threading.Lock serialises all writes.
  Reads (get_session, list_children) acquire the same lock.

Public API
──────────
  SESSION_STORE                         — module singleton
  SESSION_STORE.create_session(...)     → session_id
  SESSION_STORE.record_output(...)      → None
  SESSION_STORE.get_session(id)         → dict | None
  SESSION_STORE.list_children(parent)   → list[dict]
  SESSION_STORE.recent(n)               → list[dict]
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Storage directory
# ─────────────────────────────────────────────────────────────────────────────

_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
_SESSIONS_DIR  = _PROJECT_ROOT / "data" / "sessions"
_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentSessionRecord:
    """A single agent's session — input, output, status, timestamps."""

    session_id:        str
    parent_session_id: str | None  = None

    # Which agent produced this output.
    agent_role:        str         = ""       # AgentRole.value string

    # Status: "pending" | "completed" | "failed"
    status:            str         = "pending"

    # Masked query sent to this agent.
    query:             str         = ""

    # Draft output produced by this agent.
    output:            str         = ""

    # ISO-format timestamps.
    created_at:        str         = field(default_factory=lambda: _now())
    completed_at:      str         = ""

    # Child session IDs spawned by this session.
    children:          list[str]   = field(default_factory=list)

    # Arbitrary extra info (workflow step, warnings, conflict count, …).
    metadata:          dict[str, Any] = field(default_factory=dict)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize(session_id: str) -> str:
    """Convert a session_id to a safe filename (replace / : with _)."""
    return re.sub(r"[/\\:*?\"<>|]", "_", session_id)


def _session_path(session_id: str) -> Path:
    return _SESSIONS_DIR / f"{_sanitize(session_id)}.json"


# ─────────────────────────────────────────────────────────────────────────────
#  Session store
# ─────────────────────────────────────────────────────────────────────────────

class AgentSessionStore:
    """Thread-safe registry for per-agent session records.

    Keeps sessions in memory for fast access and writes each record to
    data/sessions/<id>.json for persistence and offline inspection.
    """

    def __init__(self) -> None:
        self._lock: threading.Lock      = threading.Lock()
        self._sessions: dict[str, AgentSessionRecord] = {}
        self._load_from_disk()

    # ── Public API ────────────────────────────────────────────────────────────

    def create_session(
        self,
        session_id:        str,
        agent_role:        str,
        query:             str,
        parent_session_id: str | None = None,
        metadata:          dict | None = None,
    ) -> str:
        """Register a new agent session and return its session_id.

        If a session with this ID already exists it is overwritten so that
        retry calls produce clean records.
        """
        record = AgentSessionRecord(
            session_id=session_id,
            parent_session_id=parent_session_id,
            agent_role=agent_role,
            status="pending",
            query=query,
            metadata=metadata or {},
        )
        with self._lock:
            self._sessions[session_id] = record
            # Register as child of parent.
            if parent_session_id and parent_session_id in self._sessions:
                parent = self._sessions[parent_session_id]
                if session_id not in parent.children:
                    parent.children.append(session_id)
                self._write(parent)
            self._write(record)

        logger.debug(
            "session-store: created  id=%r  role=%s  parent=%r",
            session_id, agent_role, parent_session_id,
        )
        return session_id

    def record_output(
        self,
        session_id: str,
        output:     str,
        status:     str = "completed",
        metadata:   dict | None = None,
    ) -> None:
        """Store the agent's output and mark the session completed/failed."""
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                logger.warning("session-store: record_output called for unknown session %r", session_id)
                return
            record.output       = output
            record.status       = status
            record.completed_at = _now()
            if metadata:
                record.metadata.update(metadata)
            self._write(record)

        logger.debug(
            "session-store: recorded  id=%r  status=%s  len=%d",
            session_id, status, len(output),
        )

    def get_session(self, session_id: str) -> dict | None:
        """Return session record as a dict, or None if not found."""
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                # Try loading from disk (supports inspecting old sessions after restart).
                record = self._read_from_disk(session_id)
                if record:
                    self._sessions[session_id] = record
            return asdict(record) if record else None

    def list_children(self, parent_session_id: str) -> list[dict]:
        """Return all child session records for a given parent."""
        with self._lock:
            parent = self._sessions.get(parent_session_id)
            if parent is None:
                return []
            return [
                asdict(self._sessions[child_id])
                for child_id in parent.children
                if child_id in self._sessions
            ]

    def recent(self, n: int = 20) -> list[dict]:
        """Return the n most recently created sessions (newest first)."""
        with self._lock:
            records = sorted(
                self._sessions.values(),
                key=lambda r: r.created_at,
                reverse=True,
            )
            return [asdict(r) for r in records[:n]]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _write(self, record: AgentSessionRecord) -> None:
        """Write a single record to disk (caller holds self._lock)."""
        path = _session_path(record.session_id)
        try:
            path.write_text(
                json.dumps(asdict(record), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error("session-store: disk write failed for %r: %s", record.session_id, exc)

    def _read_from_disk(self, session_id: str) -> AgentSessionRecord | None:
        """Load a single session from disk (caller holds self._lock)."""
        path = _session_path(session_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return AgentSessionRecord(**data)
        except Exception as exc:
            logger.error("session-store: disk read failed for %r: %s", session_id, exc)
            return None

    def _load_from_disk(self) -> None:
        """Pre-load all session files from disk on startup."""
        loaded = 0
        for path in _SESSIONS_DIR.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                record = AgentSessionRecord(**data)
                self._sessions[record.session_id] = record
                loaded += 1
            except Exception as exc:
                logger.warning("session-store: skipped corrupt file %s: %s", path.name, exc)
        if loaded:
            logger.info("session-store: loaded %d session(s) from disk", loaded)


# ─────────────────────────────────────────────────────────────────────────────
#  Module singleton — import and use directly
# ─────────────────────────────────────────────────────────────────────────────

SESSION_STORE: AgentSessionStore = AgentSessionStore()
