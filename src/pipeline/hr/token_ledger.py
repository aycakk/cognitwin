"""pipeline/hr/token_ledger.py — Token/usage budget management for the HR agent.

Each recruiter has a budget (default 1 000 tokens).
Heavy operations cost more; light operations cost less.
The ledger persists to disk so budget survives restarts.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from src.pipeline.hr.hr_schemas import TokenLedger, TOKEN_ACTION_COSTS

logger = logging.getLogger(__name__)

_DATA_DIR = Path("data/hr_profiles")
_lock = threading.Lock()


def _ledger_path(recruiter_id: str) -> Path:
    return _DATA_DIR / f"{recruiter_id}_ledger.json"


def _ensure_dir() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_ledger(recruiter_id: str, default_budget: int = 1000) -> TokenLedger:
    """Load ledger from disk or create a fresh one."""
    path = _ledger_path(recruiter_id)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return TokenLedger(**{
                k: v for k, v in raw.items()
                if k in TokenLedger.__dataclass_fields__
            })
        except Exception as exc:
            logger.warning("ledger load failed for %s: %s — using fresh ledger", recruiter_id, exc)
    ledger = TokenLedger(recruiter_id=recruiter_id, total_budget=default_budget)
    _save_ledger(ledger)
    return ledger


def _save_ledger(ledger: TokenLedger) -> None:
    _ensure_dir()
    path = _ledger_path(ledger.recruiter_id)
    with _lock:
        with open(path, "w", encoding="utf-8") as f:
            data = {
                "recruiter_id": ledger.recruiter_id,
                "total_budget": ledger.total_budget,
                "used_budget":  ledger.used_budget,
                "transactions": ledger.transactions,
            }
            json.dump(data, f, ensure_ascii=False, indent=2)


def check_and_deduct(
    ledger: TokenLedger,
    action: str,
    note: str = "",
) -> tuple[bool, str]:
    """Check budget and deduct cost if sufficient.

    Returns (ok: bool, message: str).
    ok=False means the action was blocked due to insufficient budget.
    """
    cost = TOKEN_ACTION_COSTS.get(action, 20)
    if not ledger.can_afford(cost):
        msg = (
            f"Yetersiz bütçe. Bu işlem ({action}) {cost} token gerektirir; "
            f"kalan bütçeniz: {ledger.remaining}. "
            "Lütfen yöneticinizle iletişime geçin veya daha küçük bir işlem seçin."
        )
        return False, msg
    ledger.deduct(cost, action, note)
    _save_ledger(ledger)
    return True, ""


def budget_status_block(ledger: TokenLedger) -> str:
    """Short status string suitable for inclusion in agent output."""
    pct = int(100 * ledger.remaining / ledger.total_budget) if ledger.total_budget else 0
    return (
        f"[Bütçe: {ledger.remaining}/{ledger.total_budget} token (%{pct})]"
    )


def action_cost_table() -> str:
    """Human-readable cost reference."""
    lines = ["İşlem maliyetleri:"]
    labels = {
        "cv_summary":          "CV özeti",
        "cv_parse":            "CV analizi",
        "req_parse":           "İlan analizi",
        "candidate_match":     "Aday-ilan eşleştirme",
        "shortlist_5":         "5 kişilik kısa liste",
        "shortlist_10":        "10 kişilik kısa liste",
        "interview_questions": "Mülakat soruları",
        "outreach_draft":      "Outreach mesajı",
        "batch_rank_5":        "5 adayı toplu sırala",
        "batch_rank_10":       "10 adayı toplu sırala",
        "explanation":         "Açıklama/gerekçe",
        "profile_update":      "Profil güncelleme",
    }
    for key, cost in TOKEN_ACTION_COSTS.items():
        label = labels.get(key, key)
        lines.append(f"  {label}: {cost} token")
    return "\n".join(lines)
