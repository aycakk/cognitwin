"""gates/c2_grounding.py — shared decision logic for C2 (pipeline path).

Single source of truth for the pipeline's memory-grounding check.

Policy (pipeline): the draft must be lexically grounded in the vector
context. "Grounded" means ≥ 2 content words (6+ chars, non-masked)
appear in both the context and the draft.

Exemptions (e.g. DeveloperAgent) are handled by GATE_POLICY in
src/governance/policy.py — roles that are exempt simply do not have
C2 in their gate list, so this function is never called for them.

Why only the pipeline copy is extracted:
  student_agent._gate_c2_grounding uses a different policy —
  combined vector+ontology corpus, different empty-source definition —
  and has no production caller. It is left untouched pending a
  separate policy decision (see Step 2.6 audit).

Return contract:
  (passed: bool, reason_code: str, overlap_count: int)

  reason_code values and meanings:
    "empty_pass"   → vector empty AND draft issued BlindSpot phrase; PASS
    "empty_fail"   → vector empty AND draft missing BlindSpot phrase; FAIL
    "blindspot"    → vector non-empty but draft issued BlindSpot; PASS
    "overlap_pass" → ≥ 2 shared content words found; PASS
                     overlap_count carries the exact count
    "overlap_fail" → < 2 shared content words; FAIL

  overlap_count is 0 for all non-overlap reason codes.
  The caller maps (passed, reason_code, overlap_count) → localized string.
"""

from __future__ import annotations

import re


def check_grounding(
    draft: str,
    vector_context: str,
    vector_empty: bool,
) -> tuple[bool, str, int]:
    """Decide whether `draft` is grounded in `vector_context`.

    Parameters
    ----------
    draft:          the LLM-produced response text to evaluate
    vector_context: raw ChromaDB result block (may contain [*_MASKED] tokens)
    vector_empty:   True when the vector store returned no documents

    Exemptions are handled by GATE_POLICY — roles not requiring C2 simply
    never have this function called for them.

    Returns (passed, reason_code, overlap_count). See module docstring
    for the full reason_code → message mapping.
    """
    if vector_empty:
        if "bulamadım" in draft.lower():
            return True, "empty_pass", 0
        return False, "empty_fail", 0

    if "bulamadım" in draft.lower():
        return True, "blindspot", 0

    context_words = {
        w.lower() for w in re.findall(r"\b\w{6,}\b", vector_context)
        if not re.match(r"\[.*_MASKED\]", w)
    }
    draft_words = {w.lower() for w in re.findall(r"\b\w{6,}\b", draft)}
    overlap = context_words & draft_words

    if len(overlap) >= 2:
        return True, "overlap_pass", len(overlap)
    return False, "overlap_fail", 0
