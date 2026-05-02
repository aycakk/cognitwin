"""loop/sprint_compliance.py — Phase 6.7 advisory C3_AGILE wiring.

Single helper that runs one sprint-level Ontology4Agile compliance check
against the live SprintStateStore. Strictly advisory — the caller (run_sprint)
emits a non-blocking event and attaches the result to SprintResult, but never
blocks, reroutes, or escalates based on the outcome.

Public API:

    validate_sprint_compliance(state_store) -> dict

Returned dict shape (compact):

    {
        "gate":          "C3_AGILE",
        "advisory":      True,
        "pass":          bool,
        "evidence":      str,
        "revision_hint": str,
    }

Failure paths swallow exceptions and return a `pass=True` advisory shape so
this helper can never break sprint completion.
"""

from __future__ import annotations

import logging
from typing import Any

from src.gates.evaluator import evaluate_all_gates_rich
from src.gates.gate_result import _REVISION_HINTS
from src.pipeline.scrum_team.agile_payload_builder import build_sprint_payload

logger = logging.getLogger(__name__)

_GATE_ID = "C3_AGILE"
_DEFAULT_HINT = _REVISION_HINTS.get(_GATE_ID, "")


def _safe_load(state_store: Any) -> dict:
    if state_store is None:
        return {}
    try:
        state = state_store.load()
    except Exception as exc:
        logger.warning("validate_sprint_compliance: state_store.load failed: %s", exc)
        return {}
    return state if isinstance(state, dict) else {}


def validate_sprint_compliance(state_store: Any) -> dict:
    """Run one advisory C3_AGILE check over the sprint state snapshot.

    Returns a compact advisory dict. NEVER raises — any internal failure
    is converted to a pass=True advisory note so callers can ignore it
    safely.
    """
    state = _safe_load(state_store)
    payload = build_sprint_payload(state)

    try:
        report = evaluate_all_gates_rich(
            draft="",
            vector_context="",
            is_empty=True,
            agent_role="ScrumMasterAgent",
            redo_log=[],
            agile_payload=payload,
        )
    except Exception as exc:
        logger.warning("validate_sprint_compliance: evaluator failed: %s", exc)
        return {
            "gate":          _GATE_ID,
            "advisory":      True,
            "pass":          True,
            "evidence":      f"C3_AGILE skipped — evaluator error: {exc}",
            "revision_hint": _DEFAULT_HINT,
        }

    info = report.get("gates", {}).get(_GATE_ID)
    if not info:
        return {
            "gate":          _GATE_ID,
            "advisory":      True,
            "pass":          True,
            "evidence":      "C3_AGILE not active for ScrumMasterAgent in policy.",
            "revision_hint": _DEFAULT_HINT,
        }

    return {
        "gate":          _GATE_ID,
        "advisory":      True,
        "pass":          bool(info.get("pass", True)),
        "evidence":      str(info.get("evidence", "")),
        "revision_hint": str(info.get("revision_hint", _DEFAULT_HINT)),
    }
