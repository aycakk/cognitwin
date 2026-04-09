"""gates/c7_blindspot.py — shared decision logic for C7 (pipeline path).

Single source of truth for the pipeline's BlindSpot completeness check.

Policy (pipeline): BlindSpot is required whenever the vector store
cannot ground the answer (vector_empty=True). Ontology state is not
considered here — see audit notes in Step 2.4 for rationale.

student_agent._gate_c7_blindspot uses a different policy (requires
both vector AND ontology to be empty before demanding a BlindSpot
phrase) and has no production caller. It is left untouched pending
a separate policy decision (Option B or C from the Step 2.4 audit).

Returns:
  (True,  None)              → no violation
  (False, "missing_phrase")  → vector empty but BlindSpot phrase absent
"""

from __future__ import annotations

from typing import Optional


def check_blindspot(
    draft: str,
    vector_empty: bool,
) -> tuple[bool, Optional[str]]:
    """Decide whether `draft` satisfies the BlindSpot completeness rule.

    Returns (passed, reason_kind). The caller is responsible for
    formatting the user-visible message.
    """
    if vector_empty and "bulamadım" not in draft.lower():
        return False, "missing_phrase"
    return True, None
