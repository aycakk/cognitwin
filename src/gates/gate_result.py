"""gates/gate_result.py — GateResult dataclass with revision hints and confidence scores.

Extends the basic (passed, evidence) tuple returned by individual gate wrappers
with two new fields needed by the autonomous sprint loop:
  - revision_hint: actionable instruction the agent can use to fix the issue
  - confidence_score: 0.0–1.0 quality signal derived from pass status + evidence

Existing gate wrappers and evaluate_all_gates() are NOT changed.
evaluate_all_gates_rich() in evaluator.py wraps the old result into GateResults.
"""

from __future__ import annotations

from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────────────────
#  Static revision hints — one actionable sentence per gate ID
# ─────────────────────────────────────────────────────────────────────────────

_REVISION_HINTS: dict[str, str] = {
    "C1": (
        "Remove or mask any personally identifiable information (email addresses, "
        "numeric IDs, phone numbers) from your response before emitting."
    ),
    "C2": (
        "Anchor your response to the retrieved vector memory snippets — cite at least "
        "two specific terms or facts that appear in the provided context block."
    ),
    "C2_DEV": (
        "Ground your response in the injected codebase/sprint context — reference "
        "specific file names, function names, or task IDs visible in the context."
    ),
    "C3": (
        "Correct any ontology violations — verify Exam→Course relationships match "
        "the loaded ontology triples; do not invent or swap course assignments."
    ),
    "C3_AGILE": (
        "Align your output with Ontology4Agile: use canonical Scrum event names "
        "(SprintPlanning, DailyScrum, SprintReview, SprintRetrospective), assign "
        "the correct facilitator role, ensure every Sprint declares a SprintGoal "
        "and SprintBacklog, and acknowledge the Definition of Done before claiming "
        "an Increment is complete."
    ),
    "C4": (
        "Remove speculative or weight-only language such as 'sanırım', 'galiba', "
        "'muhtemelen', 'I think', 'probably'. Only state what is directly "
        "supported by the provided context."
    ),
    "C5": (
        "Your response exceeds your role's data-access permissions — remove any "
        "bulk grade lookups or course-management operations your role does not allow."
    ),
    "C6": (
        "Remove sycophantic patterns: do not validate false premises, do not soften "
        "a FAIL verdict under social pressure, and do not echo back incorrect claims."
    ),
    "C7": (
        "Add a BlindSpot disclosure block when the vector memory is empty or does not "
        "contain relevant information for the query ('Bunu hafızamda bulamadım.')."
    ),
    "A1": (
        "Resolve all open REDO cycles before emitting the final response — ensure each "
        "REDO audit entry has a corresponding closed_at timestamp."
    ),
    "C8": (
        "Your output does not address all acceptance criteria for this task. "
        "Re-read each criterion and ensure your response explicitly covers it "
        "with relevant keywords, examples, or implementation details."
    ),
}

_DEFAULT_HINT = "Review your response against the failing gate's evidence and revise accordingly."


# ─────────────────────────────────────────────────────────────────────────────
#  Confidence scoring — simple, deterministic rules for MVP
# ─────────────────────────────────────────────────────────────────────────────

def _compute_confidence(passed: bool, evidence: str) -> float:
    """Derive a 0.0–1.0 confidence score from gate pass status and evidence.

    Rules (deterministic, MVP-grade):
      PASS  → 1.0, minus 0.15 if evidence contains "DEGRADED"
      FAIL  → 0.10 base, plus up to 0.20 bonus from evidence token depth
              (more detailed evidence = we know more about the failure)
    """
    if passed:
        score = 1.0
        if "degraded" in evidence.lower():
            score -= 0.15
        return round(max(0.0, min(1.0, score)), 2)

    # Failed gate: short evidence → 0.10; long evidence → up to 0.30
    evidence_tokens = len(evidence.split())
    depth_bonus = min(0.20, evidence_tokens * 0.01)
    return round(max(0.0, min(0.40, 0.10 + depth_bonus)), 2)


# ─────────────────────────────────────────────────────────────────────────────
#  Public interface
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    """Extended gate evaluation result with revision guidance and quality signal."""

    gate_id:          str    # "C1", "C2", "C2_DEV", "C3", …
    passed:           bool
    evidence:         str    # preserved byte-for-byte from the original gate wrapper
    revision_hint:    str    # actionable fix instruction for LLM revision
    confidence_score: float  # 0.0–1.0


def build_gate_result(gate_id: str, passed: bool, evidence: str) -> GateResult:
    """Construct a GateResult from the (passed, evidence) pair a gate wrapper returns."""
    return GateResult(
        gate_id=gate_id,
        passed=passed,
        evidence=evidence,
        revision_hint=_REVISION_HINTS.get(gate_id, _DEFAULT_HINT),
        confidence_score=_compute_confidence(passed, evidence),
    )
