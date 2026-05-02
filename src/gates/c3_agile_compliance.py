"""gates/c3_agile_compliance.py — Ontology4Agile-backed Scrum shape gate.

Validates Scrum-shaped payloads against Ontology4Agile v1.3.0 facts surfaced
through src/ontology/agile_contract.py. Lives alongside the existing gates
but is NOT yet active in any role's GATE_POLICY (Phase 5).

Payload shape (all keys optional — absent sections are simply not validated):

    {
        "event": {
            "name": "SprintReview",         # local Scrum event class name
            "facilitator": "ProductOwner",  # local Scrum role class name
        },
        "sprint": {
            "goal":    "<SprintGoal text>",
            "backlog": [...]   # SprintBacklog items (any non-empty iterable)
            "tasks":   [...]   # alternative to backlog
        },
        "increment": {
            "items":               [...],   # increment contents
            "dod_acknowledged":    bool,    # explicit DoD ack
            "acceptance_evidence": str|list # AC validation evidence
        },
        "impediment": {
            "owner":       "ScrumMaster",   # role name
            "description": "<text>"
        }
    }

Returns (passed, evidence). When the payload is empty / None the gate is
treated as not-applicable and PASSes — this lets the wrapper be wired into
the evaluator without affecting roles that do not provide structured data.

Degraded mode: if Ontology4Agile is unavailable and not required, ontology
lookups are skipped and the gate emits a "DEGRADED PASS" — same convention
as C3 in evaluator.py.
"""
from __future__ import annotations

from typing import Any, Optional


def _is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    if isinstance(value, str):
        return bool(value.strip())
    return True


def check_agile_compliance(payload: Optional[dict]) -> tuple[bool, str]:
    """Validate a Scrum-shaped payload against Ontology4Agile.

    Returns (True, evidence) when the payload is empty / None or every
    provided section satisfies the contract; (False, evidence) when any
    explicit section violates it.
    """
    if not payload:
        return True, "C3_AGILE not applicable — no Scrum payload provided."

    # Lazy import — keeps gate module importable even if rdflib is missing.
    from src.ontology import agile_contract

    graph = agile_contract.load_agile_graph()
    ontology_unavailable = graph is None

    failures: list[str] = []

    # --- 1. Scrum event / facilitator -------------------------------------
    event = payload.get("event") if isinstance(payload, dict) else None
    if isinstance(event, dict):
        event_name = (event.get("name") or "").strip()
        facilitator = (event.get("facilitator") or "").strip()

        if event_name and not ontology_unavailable:
            valid_events = agile_contract.valid_scrum_events()
            if valid_events and event_name not in valid_events:
                failures.append(
                    f"unknown Scrum event '{event_name}' "
                    f"(expected one of {sorted(valid_events)})"
                )
            elif facilitator:
                expected = agile_contract.valid_facilitators(event_name)
                if expected and facilitator not in expected:
                    failures.append(
                        f"event '{event_name}' facilitator should be one of "
                        f"{sorted(expected)}, got '{facilitator}'"
                    )

    # --- 2. Sprint context ------------------------------------------------
    sprint = payload.get("sprint") if isinstance(payload, dict) else None
    if isinstance(sprint, dict):
        goal = (sprint.get("goal") or "").strip()
        backlog = sprint.get("backlog")
        tasks = sprint.get("tasks")
        if not goal:
            failures.append("sprint context missing SprintGoal")
        if not (_is_non_empty(backlog) or _is_non_empty(tasks)):
            failures.append("sprint context missing SprintBacklog / tasks")

    # --- 3. Product Increment --------------------------------------------
    increment = payload.get("increment") if isinstance(payload, dict) else None
    if isinstance(increment, dict):
        dod_ack = bool(increment.get("dod_acknowledged"))
        dod_evidence = increment.get("dod_evidence")
        ac_evidence = (
            increment.get("acceptance_evidence")
            or increment.get("acceptance_criteria_validated")
        )
        has_evidence = (
            dod_ack
            or _is_non_empty(dod_evidence)
            or _is_non_empty(ac_evidence)
        )
        if not has_evidence:
            failures.append(
                "ProductIncrement missing DefinitionOfDone acknowledgement "
                "or acceptance criteria evidence"
            )

    # --- 4. Impediment ---------------------------------------------------
    impediment = payload.get("impediment") if isinstance(payload, dict) else None
    if isinstance(impediment, dict):
        owner = (impediment.get("owner") or "").strip()
        if owner and not ontology_unavailable:
            valid_roles = agile_contract.valid_scrum_roles()
            if valid_roles and owner not in valid_roles:
                failures.append(
                    f"impediment owner '{owner}' is not a recognised Scrum role "
                    f"(expected one of {sorted(valid_roles)})"
                )

    if failures:
        return False, "C3_AGILE FAIL — " + "; ".join(failures)

    if ontology_unavailable:
        return True, (
            "C3_AGILE DEGRADED PASS — Ontology4Agile unavailable; "
            "structural Scrum checks only. Set COGNITWIN_ONTOLOGY_REQUIRED=1 "
            "to treat absence as FAIL."
        )

    return True, "C3_AGILE PASS — Scrum-shape contract satisfied."
