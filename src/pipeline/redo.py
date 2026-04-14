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
from typing import Callable, Optional

from src.pipeline.redo_audit import append_session


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
            rec["closed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            return


def run_redo_loop(
    draft: str,
    base_messages: list[dict],
    vector_context: str,
    is_empty: bool,
    redo_log: list[dict],
    *,
    agent_role: str,
    query: str,
    redo_rules: str,
    limit_message_template: str,
    post_process: Callable[[str], str],
    gate_fn: Callable,
    chat_fn: Callable,
    blindspot_fn: Callable,
    session_id: Optional[str] = None,
    gate_kwargs: dict | None = None,
) -> tuple[str, bool]:
    """Run the gate-verification + REDO loop shared by both pipeline paths.

    Parameters
    ----------
    draft:                  Current LLM draft to verify.
    base_messages:          Chat history used to build the REDO revision turn.
    vector_context:         Raw ChromaDB result block (passed to gate_fn).
    is_empty:               True when the vector store has no results.
    redo_log:               Per-request audit list; mutated in place.

    Keyword-only (path-specific):
    agent_role:             Passed to gate_fn for C2/C5 exemption checks.
    query:                  User's original request; passed to blindspot_fn.
    redo_rules:             Tail of the revision instruction appended after
                            the shared "Rules: Do NOT hallucinate…" prefix.
                            Preserved byte-for-byte from the original path.
    limit_message_template: Body of the limit-exceeded failure string.
                            Must contain a single {gate} placeholder which
                            is formatted with the first failing gate key.
    post_process:           Applied to the REDO response content after
                            .strip().  Student path passes identity;
                            developer path passes _sanitize_output.
    gate_fn:                evaluate_all_gates — injected to avoid circular
                            import (pipeline.py → redo.py → pipeline.py).
    chat_fn:                Ollama chat callable — injected for testability.
    blindspot_fn:           build_blindspot_block — injected to avoid circular
                            import.

    Returns
    -------
    (result, limit_hit)
      limit_hit=True:  REDO limit was reached; result is the complete
                       early-return string (blindspot block + limit message).
                       The caller must return it immediately.
      limit_hit=False: All gates passed within the allowed attempts; result
                       is the final draft string. The caller continues to
                       Stage 4 emission.
    """
    MAX_REDO       = 2
    active_redo_id: Optional[str] = None

    for attempt in range(MAX_REDO + 1):
        gate_report = gate_fn(draft, vector_context, is_empty, agent_role, redo_log, **(gate_kwargs or {}))

        if gate_report["conjunction"]:
            if active_redo_id:
                _close_redo(
                    redo_log,
                    active_redo_id,
                    "Draft passed all gates after revision.",
                    gate_report["gates"],
                )
            break

        first_fail = next(
            (k for k, v in gate_report["gates"].items() if not v["pass"]),
            "UNKNOWN",
        )
        fail_ev = gate_report["gates"].get(first_fail, {}).get("evidence", "")

        if attempt == MAX_REDO:
            _open_redo(redo_log, first_fail, fail_ev)
            append_session(
                redo_log=redo_log,
                agent_role=agent_role,
                masked_query=query,
                limit_hit=True,
                session_id=session_id,
            )
            return (
                blindspot_fn(query, f"REDO LIMIT ({first_fail} FAIL)")
                + limit_message_template.format(gate=first_fail)
            ), True

        active_redo_id = _open_redo(redo_log, first_fail, fail_ev)
        redo_instruction = (
            f"[REDO TRIGGERED — Gate {first_fail} FAILED]\n"
            f"Evidence: {fail_ev}\n"
            "Revise your previous draft to fix the failing dimension.\n"
            "Rules: Do NOT hallucinate. Do NOT unmask PII. "
            + redo_rules
        )
        redo_resp = chat_fn(
            model="llama3.2",
            messages=base_messages + [
                {"role": "assistant", "content": draft},
                {"role": "user",      "content": redo_instruction},
            ],
        )
        draft = post_process(redo_resp.message.content.strip())

    # Persist the audit trail (only if REDO was actually triggered).
    append_session(
        redo_log=redo_log,
        agent_role=agent_role,
        masked_query=query,
        limit_hit=False,
        session_id=session_id,
    )
    return draft, False
