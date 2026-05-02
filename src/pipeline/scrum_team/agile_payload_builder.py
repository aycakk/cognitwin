"""pipeline/scrum_team/agile_payload_builder.py — Phase 6.5.

Pure-data adapter that converts existing CogniTwin sprint state dicts
into the agile_payload shape consumed by src.gates.c3_agile_compliance.
No I/O, no ontology loading, no mutation of inputs.

Public API:

    build_sprint_payload(sprint_state)   -> dict
    build_event_payload(event_name, facilitator, extra=None) -> dict
    build_increment_payload(increment)   -> dict

Each function returns a fragment shaped for `check_agile_compliance`
(`event` / `sprint` / `increment` keys at the top level). The fragments
are independently usable; `build_sprint_payload` composes the sprint
and increment fragments from a single sprint_state snapshot.

NOT wired into runtime in Phase 6.5 — sprint_loop.py is unchanged.
"""

from __future__ import annotations

from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Field constants — kept here so behavior diffs are visible in code review.
# ---------------------------------------------------------------------------

_ACCEPTED_PO_STATUSES: frozenset[str] = frozenset(
    {"accepted", "agent_accepted", "human_accepted"}
)


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _coerce_iterable(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _first_non_empty(*candidates: Any) -> Any:
    """Return the first candidate that is not None / "" / empty container."""
    for c in candidates:
        if c is None:
            continue
        if isinstance(c, str) and not c.strip():
            continue
        if isinstance(c, (list, tuple, set, dict)) and len(c) == 0:
            continue
        return c
    return None


# ---------------------------------------------------------------------------
# Event fragment.
# ---------------------------------------------------------------------------


def build_event_payload(
    event_name: str,
    facilitator: str,
    extra: dict | None = None,
) -> dict:
    """Construct the {'event': {...}} fragment for C3_AGILE.

    Empty / missing event_name and facilitator are passed through; the
    gate treats them as not-applicable on its end.
    """
    event: dict[str, Any] = {
        "name":        _coerce_str(event_name),
        "facilitator": _coerce_str(facilitator),
    }
    if isinstance(extra, dict):
        for k, v in extra.items():
            if k in ("name", "facilitator"):
                continue
            event[k] = v
    return {"event": event}


# ---------------------------------------------------------------------------
# Increment fragment.
# ---------------------------------------------------------------------------


def _items_from_increment(increment: Any) -> list[Any]:
    if isinstance(increment, dict):
        items = increment.get("items")
        if items is None:
            items = increment.get("contents")
        return _coerce_iterable(items)
    return _coerce_iterable(increment)


def _evidence_from_tasks(tasks: Iterable[Any]) -> tuple[bool, list[str]]:
    """Scan task dicts for accepted+ac_validated rows.

    Returns (any_dod_acknowledged, acceptance_evidence_lines).
    """
    dod_ack = False
    evidence_lines: list[str] = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        po_status = _coerce_str(t.get("po_status"))
        ac_validated = bool(t.get("ac_validated"))
        accepted = bool(t.get("accepted")) or po_status in _ACCEPTED_PO_STATUSES
        if accepted and ac_validated:
            dod_ack = True
        if accepted or ac_validated:
            tid = _coerce_str(t.get("id")) or _coerce_str(t.get("task_id"))
            evidence_lines.append(
                f"task={tid or '?'} po_status={po_status or '?'} "
                f"ac_validated={ac_validated}"
            )
    return dod_ack, evidence_lines


def build_increment_payload(increment: Any, *, tasks: Any = None) -> dict:
    """Construct the {'increment': {...}} fragment for C3_AGILE.

    Accepts:
      - a list of increment items
      - a dict shaped {items, dod_acknowledged, ...}
      - None / empty → minimal fragment (no items, no evidence)

    Optional `tasks` is a list of task dicts (sprint_state['tasks']) used
    to infer DoD acknowledgement / acceptance evidence from task fields
    (`po_status`, `ac_validated`).
    """
    items = _items_from_increment(increment)

    dod_acknowledged = False
    dod_evidence: Any = None
    acceptance_evidence: Any = None
    acceptance_criteria_validated: list[Any] = []

    if isinstance(increment, dict):
        dod_acknowledged = bool(increment.get("dod_acknowledged"))
        dod_evidence = increment.get("dod_evidence")
        acceptance_evidence = (
            increment.get("acceptance_evidence")
            or increment.get("acceptance_criteria_validated")
        )

    if tasks is not None:
        derived_ack, derived_lines = _evidence_from_tasks(_coerce_iterable(tasks))
        dod_acknowledged = dod_acknowledged or derived_ack
        if derived_lines and not acceptance_evidence:
            acceptance_evidence = derived_lines
        for t in _coerce_iterable(tasks):
            if not isinstance(t, dict):
                continue
            if t.get("ac_validated") and t.get("acceptance_criteria"):
                acceptance_criteria_validated.extend(
                    _coerce_iterable(t.get("acceptance_criteria"))
                )

    fragment: dict[str, Any] = {"items": list(items)}
    if dod_acknowledged:
        fragment["dod_acknowledged"] = True
    if dod_evidence:
        fragment["dod_evidence"] = dod_evidence
    if acceptance_evidence:
        fragment["acceptance_evidence"] = acceptance_evidence
    if acceptance_criteria_validated:
        fragment["acceptance_criteria_validated"] = acceptance_criteria_validated

    return {"increment": fragment}


# ---------------------------------------------------------------------------
# Sprint payload composition.
# ---------------------------------------------------------------------------


def _extract_goal(sprint_state: dict) -> str:
    sprint = sprint_state.get("sprint") if isinstance(sprint_state, dict) else None
    sprint_goal = sprint.get("goal") if isinstance(sprint, dict) else None
    candidate = _first_non_empty(
        sprint_state.get("sprint_goal"),
        sprint_state.get("goal"),
        sprint_goal,
    )
    text = _coerce_str(candidate)
    # Treat the placeholder default state value as "no goal set".
    if "tanımlanmamış" in text:
        return ""
    return text


def _extract_backlog(sprint_state: dict) -> list[Any]:
    sprint = sprint_state.get("sprint") if isinstance(sprint_state, dict) else None
    nested_backlog = sprint.get("backlog") if isinstance(sprint, dict) else None
    candidate = _first_non_empty(
        sprint_state.get("sprint_backlog"),
        sprint_state.get("backlog"),
        nested_backlog,
    )
    return _coerce_iterable(candidate)


def _extract_tasks(sprint_state: dict) -> list[Any]:
    return _coerce_iterable(sprint_state.get("tasks") if isinstance(sprint_state, dict) else None)


def _extract_increment(sprint_state: dict) -> Any:
    if not isinstance(sprint_state, dict):
        return None
    return _first_non_empty(
        sprint_state.get("product_increment"),
        sprint_state.get("increment"),
    )


def build_sprint_payload(sprint_state: dict | None) -> dict:
    """Convert a sprint_state snapshot into a full agile_payload.

    The output is a fresh dict — input is never mutated. Empty / None
    input yields the minimal `{}` payload (which C3_AGILE treats as
    not-applicable).

    Composition rules:
      - sprint section always present when goal OR (backlog/tasks) found.
      - increment section always present when increment items or task
        evidence is found.
      - event/impediment sections are NOT inferred here; callers add them
        explicitly via build_event_payload() / direct dict merge.
    """
    if not isinstance(sprint_state, dict) or not sprint_state:
        return {}

    payload: dict[str, Any] = {}

    goal = _extract_goal(sprint_state)
    backlog = _extract_backlog(sprint_state)
    tasks = _extract_tasks(sprint_state)

    if goal or backlog or tasks:
        sprint_fragment: dict[str, Any] = {}
        if goal:
            sprint_fragment["goal"] = goal
        if backlog:
            sprint_fragment["backlog"] = list(backlog)
        if tasks:
            sprint_fragment["tasks"] = list(tasks)
        payload["sprint"] = sprint_fragment

    increment_raw = _extract_increment(sprint_state)
    if increment_raw or tasks:
        inc_fragment = build_increment_payload(increment_raw, tasks=tasks)["increment"]
        if inc_fragment.get("items") or any(
            k in inc_fragment
            for k in (
                "dod_acknowledged",
                "dod_evidence",
                "acceptance_evidence",
                "acceptance_criteria_validated",
            )
        ):
            payload["increment"] = inc_fragment

    return payload
