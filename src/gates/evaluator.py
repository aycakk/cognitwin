"""gates/evaluator.py — ZT4SWE gate array: wrappers + aggregator.

Single source of truth for the full C1∧C2∧C3∧C4∧C5∧C6∧C7∧A1 gate array.
Previously defined entirely in src/services/api/pipeline.py; moved here
in the Step 4 pipeline cleanup so that the gate layer is self-contained.

Each gate_cN_* wrapper:
  • calls the shared pure check function from src/gates/
  • maps the machine-readable reason code to the original English message
    (byte-for-byte preserved)
  • returns (passed: bool, evidence: str)

evaluate_all_gates aggregates all wrappers into a structured report dict.
"""

from __future__ import annotations

import logging
import re

from src.governance.policy import GATE_POLICY, DEFAULT_GATE_POLICY
from src.shared.patterns import PII_PATTERNS
from src.gates.c2_grounding import check_grounding as _check_c2
from src.gates.c3_ontology_compliance import check_ontology_compliance as _check_c3
from src.gates.c4_hallucination import check_hallucination as _check_c4
from src.gates.c5_role_permission import check_role_permission as _check_c5
from src.gates.c6_anti_sycophancy import check_anti_sycophancy as _check_c6
from src.gates.c7_blindspot import check_blindspot as _check_c7
from src.ontology.loader import _get_ontology_graph, _sparql, OntologyLoadError

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  GATE WRAPPERS  (message strings preserved byte-for-byte)
# ─────────────────────────────────────────────────────────────────────────────

def gate_c1_pii_masking(draft: str) -> tuple[bool, str]:
    """C1 — No raw PII in emitted response.

    Now sweeps the shared PII_PATTERNS superset (numeric ID, e-mail,
    phone). Previously the pipeline copy checked only ID + e-mail,
    which meant phone numbers could leak on the developer path while
    the student path caught them. Unified in Step 2 of the migration
    plan per explicit approval.
    """
    for pattern in PII_PATTERNS:
        m = pattern.search(draft)
        if m:
            matched = m.group()
            if "@" in matched:
                return False, "Unmasked email detected."
            if re.fullmatch(r"\d{9,12}", matched):
                return False, "Unmasked numeric ID detected."
            return False, "Unmasked phone number detected."
    return True, "No raw PII detected."


def gate_c2_memory_grounding(
    draft: str,
    vector_context: str,
    is_empty: bool,
) -> tuple[bool, str]:
    """C2 — Draft must be grounded in the retrieved vector context.

    Decision logic lives in src.gates.c2_grounding. This wrapper
    preserves the original English messages byte-for-byte.

    Exemptions (e.g. DeveloperAgent) are handled by GATE_POLICY in
    src/governance/policy.py — those roles simply do not have C2 in
    their gate list, so this function is never called for them.
    """
    passed, reason, overlap_count = _check_c2(
        draft,
        vector_context,
        vector_empty=is_empty,
    )
    if reason == "empty_pass":
        return True, "Vector memory empty; BlindSpot disclosure present."
    if reason == "empty_fail":
        return False, "Vector memory empty but no BlindSpot disclosure."
    if reason == "blindspot":
        return True, "Draft issued BlindSpot — acceptable when query not in results."
    if reason == "overlap_pass":
        return True, f"Grounding verified ({overlap_count} shared content terms)."
    # reason == "overlap_fail"
    return False, "Draft not grounded in vector context (overlap < 2 terms). Possible hallucination."


def gate_c3_ontology_compliance(draft: str) -> tuple[bool, str]:
    """C3 — Draft must not contradict ontology triples.

    Decision logic lives in src.gates.c3_ontology_compliance. This
    wrapper owns the two infrastructure concerns:
      1. Checking whether the rdflib graph is available.
      2. Running the SPARQL query and passing the plain result list
         to the shared function (which has no rdflib dependency).

    Degraded-mode policy
    --------------------
    When the ontology graph is unavailable AND COGNITWIN_ONTOLOGY_REQUIRED
    is not set, this gate issues a DEGRADED PASS — the word "DEGRADED"
    is in the evidence string so dashboards and log scrapers can surface it.
    The old silent "provisional PASS" message is replaced so that operators
    know the ontology is not backing this check.

    When COGNITWIN_ONTOLOGY_REQUIRED=1 and the graph failed to load,
    _get_ontology_graph() raises OntologyLoadError, which we catch here
    and convert to a hard FAIL.
    """
    try:
        graph = _get_ontology_graph()
    except OntologyLoadError as exc:
        return False, f"C3 FAIL — ontology required but unavailable: {exc}"

    if graph is None:
        return True, (
            "C3 DEGRADED PASS — ontology not loaded. "
            "Set COGNITWIN_ONTOLOGY_REQUIRED=1 to treat absence as FAIL. "
            "Check logs for missing .ttl file warnings."
        )

    pairs = _sparql("""
        PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX upper: <http://cognitwin.org/upper#>
        PREFIX coode: <http://www.co-ode.org/ontologies/ont.owl#>
        SELECT ?exam ?course WHERE {
            ?exam rdf:type upper:Exam .
            ?exam coode:activityPartOf ?course .
        }""")

    passed, reason, detail = _check_c3(draft, pairs)
    if passed:
        return True, "No ontology rule violations detected."
    exam_lbl, bad_oc, expected = detail
    return False, (
        f"Ontology violation: '{exam_lbl}' paired with '{bad_oc}', "
        f"expected '{expected}'."
    )


def gate_c4_hallucination(draft: str) -> tuple[bool, str]:
    """C4 — No weight-only or hallucinatory claim markers.

    Decision logic lives in src.gates.c4_hallucination. This wrapper
    preserves the original English messages byte-for-byte.
    """
    passed, label, matched = _check_c4(draft)
    if passed:
        return True, "No hallucination markers detected."
    return False, f"[{label}] '{matched}'"


def gate_c5_role_permission(draft: str, agent_role: str) -> tuple[bool, str]:
    """C5 — Role-permission boundary enforcement.

    Decision logic lives in src.gates.c5_role_permission. This wrapper
    preserves the original English messages byte-for-byte so existing
    callers and golden tests see identical return values.
    """
    passed, kind = _check_c5(draft, agent_role)
    if passed:
        return True, f"Role '{agent_role}' — no boundary violations."
    if kind == "bulk_grades":
        return False, f"'{agent_role}' lacks 'read_all_student_grades' permission."
    if kind == "manage_courses":
        return False, f"'{agent_role}' lacks 'manage_courses' permission."
    # Unreachable: _check_c5 returns only the codes handled above.
    return passed, f"Role '{agent_role}' — unspecified violation ({kind})."


def gate_c6_anti_sycophancy(draft: str) -> tuple[bool, str]:
    """C6 — Full ASP-NEG pattern sweep.

    Decision logic lives in src.gates.c6_anti_sycophancy. This wrapper
    preserves the original English messages byte-for-byte.
    """
    passed, violations = _check_c6(draft)
    if passed:
        return True, "All ASP-NEG classifiers: NO_MATCH."
    rendered = [f"[{label}] '{match}'" for label, match in violations]
    return False, "ASP violations: " + "; ".join(rendered)


def gate_c7_blindspot(draft: str, is_empty: bool) -> tuple[bool, str]:
    """C7 — Unanswerable queries must carry a BlindSpot disclosure.

    Decision logic lives in src.gates.c7_blindspot. This wrapper
    preserves the original English messages byte-for-byte.

    Note: student_agent._gate_c7_blindspot uses a different policy
    (both_empty = vector AND ontology empty) and is not wired to this
    shared module — see Step 2.4 audit for details.
    """
    passed, reason = _check_c7(draft, is_empty)
    if passed:
        return True, "BlindSpot completeness verified."
    return False, "Empty vector memory but BlindSpot phrase missing."


def gate_a1_redo_checksum(redo_log: list[dict]) -> tuple[bool, str]:
    """A1 — No zombie REDO cycles (more than one open cycle is anomalous).

    Renamed from gate_c8_redo_checksum: this is an orchestration-layer
    audit gate, not a content gate. It inspects the REDO log, not the
    draft text, so it does not belong in the C-series.
    """
    open_cycles = [r for r in redo_log if not r.get("closed_at")]
    if len(open_cycles) > 1:
        return False, f"Too many open REDO cycles: {len(open_cycles)}"
    return True, "REDO cycle state clean."


# ─────────────────────────────────────────────────────────────────────────────
#  GATE AGGREGATOR
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_gates(
    draft: str,
    vector_context: str,
    is_empty: bool,
    agent_role: str,
    redo_log: list[dict],
) -> dict:
    """
    Execute the gate set defined in GATE_POLICY[agent_role] and return a
    structured report.

    C1 (PII masking) is always run regardless of policy — it is a hard
    safety invariant applied to every emitted response.

    All other gates are looked up from GATE_POLICY.  Roles not in the
    policy fall back to DEFAULT_GATE_POLICY.  This replaces the previous
    hard-coded `if agent_role == "DeveloperAgent": exempt=True` logic —
    exemptions are now expressed as absent gate IDs in the policy table.

    The "A1" key is the REDO-checksum audit gate (formerly mislabeled
    "C8"). Downstream consumers that previously read report["gates"]["C8"]
    must now read report["gates"]["A1"].
    """
    role_gates = GATE_POLICY.get(agent_role, DEFAULT_GATE_POLICY)
    if agent_role not in GATE_POLICY:
        logger.warning(
            "evaluate_all_gates: unknown agent_role=%r — using DEFAULT_GATE_POLICY %s",
            agent_role,
            DEFAULT_GATE_POLICY,
        )

    # C1 is always evaluated (hard safety invariant, not role-dependent).
    gates: dict[str, tuple[bool, str]] = {
        "C1": gate_c1_pii_masking(draft),
    }

    # Role-specific gates from policy table.
    _gate_dispatch: dict[str, tuple[bool, str]] = {}

    if "C2" in role_gates:
        _gate_dispatch["C2"] = gate_c2_memory_grounding(draft, vector_context, is_empty)
    if "C3" in role_gates:
        _gate_dispatch["C3"] = gate_c3_ontology_compliance(draft)
    if "C4" in role_gates:
        _gate_dispatch["C4"] = gate_c4_hallucination(draft)
    if "C5" in role_gates:
        _gate_dispatch["C5"] = gate_c5_role_permission(draft, agent_role)
    if "C6" in role_gates:
        _gate_dispatch["C6"] = gate_c6_anti_sycophancy(draft)
    if "C7" in role_gates:
        _gate_dispatch["C7"] = gate_c7_blindspot(draft, is_empty)
    if "A1" in role_gates:
        _gate_dispatch["A1"] = gate_a1_redo_checksum(redo_log)

    gates.update(_gate_dispatch)

    structured = {k: {"pass": v[0], "evidence": v[1]} for k, v in gates.items()}
    return {
        "conjunction": all(v[0] for v in gates.values()),
        "gates": structured,
        "role": agent_role,
        "active_gates": list(gates.keys()),
    }
