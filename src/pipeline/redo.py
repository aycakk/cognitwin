"""pipeline/redo.py — REDO engine bookkeeping.

Single source of truth for opening and closing REDO cycles.
Previously defined as module-level functions in
src/services/api/pipeline.py; moved here in Step 3.1 of the
migration plan.

Pure move: no logic changes, no signature changes.
Both run_pipeline (student path) and _process_developer_message
(developer path) in pipeline.py import from here.

A REDO cycle is opened when a gate fails and the pipeline needs
to ask the LLM to revise its draft. It is closed when a
subsequent attempt passes all gates, recording which gates
passed at closure time.

The redo_log is a plain list[dict] allocated per-request in each
pipeline function; there is no shared mutable state.
"""

from __future__ import annotations

import datetime
import uuid


def _open_redo(redo_log: list[dict], trigger_gate: str, evidence: str) -> str:
    """Append a new open REDO cycle to redo_log and return its ID."""
    redo_id = str(uuid.uuid4())[:8]
    redo_log.append({
        "redo_id":         redo_id,
        "trigger_gate":    trigger_gate,
        "failed_evidence": evidence,
        "revision_action": None,
        "closure_gates":   {},
        "closed_at":       None,
    })
    return redo_id


def _close_redo(
    redo_log: list[dict],
    redo_id: str,
    action: str,
    gate_results: dict,
) -> None:
    """Mark an open REDO cycle as closed with its resolution details."""
    for rec in redo_log:
        if rec["redo_id"] == redo_id:
            rec["revision_action"] = action
            rec["closure_gates"]   = {
                k: "PASS" if v["pass"] else "FAIL"
                for k, v in gate_results.items()
            }
            rec["closed_at"] = datetime.datetime.utcnow().isoformat()
            return
