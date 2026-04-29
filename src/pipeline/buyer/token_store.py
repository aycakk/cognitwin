"""buyer/token_store.py — persistent token state for Fashion Buyer Agent.

Scope
-----
This store is intentionally isolated for Buyer-only budgeting.
No other agent reads or writes these files.

Storage files (auto-created on first use):
  data/buyer/budget_state.json
  data/buyer/token_ledger.jsonl
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_BUYER_TOTAL = 100
GLOBAL_SESSION_KEY = "__global__"


class BuyerTokenStore:
    """Thread-safe Buyer token persistence (single-process MVP)."""

    def __init__(self, data_dir: Path | None = None, default_total: int = DEFAULT_BUYER_TOTAL) -> None:
        env_dir = os.environ.get("COGNITWIN_BUYER_DATA_DIR", "").strip()
        if data_dir is not None:
            resolved_dir = data_dir
        elif env_dir:
            resolved_dir = Path(env_dir)
        else:
            project_root = Path(__file__).resolve().parents[3]
            resolved_dir = project_root / "data" / "buyer"

        self._data_dir = resolved_dir
        self._state_path = self._data_dir / "budget_state.json"
        self._ledger_path = self._data_dir / "token_ledger.jsonl"
        self._default_total = int(default_total)
        self._lock = threading.RLock()
        self._ensure_files()

    @staticmethod
    def normalize_session_key(session_id: str | None) -> str:
        value = (session_id or "").strip()
        return value if value else GLOBAL_SESSION_KEY

    def consume(
        self,
        *,
        session_id: str | None,
        required_tokens: int,
        action_costs: dict[str, int],
    ) -> dict[str, Any]:
        """Apply Buyer request cost atomically when budget is sufficient."""
        with self._lock:
            state = self._load_state()
            key = self.normalize_session_key(session_id)
            sessions = state.setdefault("sessions", {})
            session_state = sessions.get(key) or {
                "total": self._default_total,
                "used": 0,
                "remaining": self._default_total,
            }

            total = int(session_state.get("total", self._default_total))
            used_before = int(session_state.get("used", 0))
            remaining_before = int(session_state.get("remaining", max(total - used_before, 0)))

            if remaining_before < required_tokens:
                blocked_record = {
                    "event_id": str(uuid.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "blocked",
                    "session_key": key,
                    "budget_total": total,
                    "required_tokens": required_tokens,
                    "remaining_before": remaining_before,
                    "action_costs": action_costs,
                }
                self._append_ledger(blocked_record)
                return {
                    "allowed": False,
                    "session_key": key,
                    "budget_total": total,
                    "used_this_request": required_tokens,
                    "total_used": used_before,
                    "remaining": remaining_before,
                    "action_costs": action_costs,
                }

            used_after = used_before + required_tokens
            remaining_after = max(total - used_after, 0)
            updated_session_state = {
                "total": total,
                "used": used_after,
                "remaining": remaining_after,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            sessions[key] = updated_session_state
            state["sessions"] = sessions
            self._save_state(state)

            applied_record = {
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "applied",
                "session_key": key,
                "budget_total": total,
                "required_tokens": required_tokens,
                "used_before": used_before,
                "used_after": used_after,
                "remaining_after": remaining_after,
                "action_costs": action_costs,
            }
            self._append_ledger(applied_record)

            return {
                "allowed": True,
                "session_key": key,
                "budget_total": total,
                "used_this_request": required_tokens,
                "total_used": used_after,
                "remaining": remaining_after,
                "action_costs": action_costs,
            }

    def _ensure_files(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        if not self._state_path.exists():
            initial_state = {
                "default_total": self._default_total,
                "sessions": {},
            }
            self._save_state(initial_state)
        if not self._ledger_path.exists():
            self._ledger_path.write_text("", encoding="utf-8")

    def _load_state(self) -> dict[str, Any]:
        if not self._state_path.exists():
            self._ensure_files()
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("default_total", self._default_total)
                data.setdefault("sessions", {})
                return data
        except (json.JSONDecodeError, OSError):
            pass
        fallback = {"default_total": self._default_total, "sessions": {}}
        self._save_state(fallback)
        return fallback

    def _save_state(self, state: dict[str, Any]) -> None:
        tmp_path = self._state_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self._state_path)

    def _append_ledger(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

