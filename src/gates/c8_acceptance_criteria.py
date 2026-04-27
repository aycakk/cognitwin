"""gates/c8_acceptance_criteria.py — Acceptance Criteria validation gate.

Checks whether developer output addresses each acceptance criterion
listed on the task. Uses keyword overlap as a lightweight, deterministic
check — no LLM call required.

Algorithm
---------
For each criterion:
  1. Extract "significant words": alphabetic tokens longer than 3 characters.
  2. Count how many appear verbatim in the (lowercased) draft.
  3. If the match ratio >= 0.40 the criterion is considered addressed.
  4. Criteria with no significant words are skipped (always pass).

Thresholds
----------
  MATCH_RATIO  = 0.40   (40% of criterion keywords found in output)
  MAX_REPORTED = 3      (max number of failing criteria shown in evidence)

Returns (passed: bool, evidence: str).
"""

from __future__ import annotations

import re
from typing import Sequence

MATCH_RATIO  = 0.40
MAX_REPORTED = 3


def check_acceptance_criteria(
    draft: str,
    criteria: Sequence[str],
) -> tuple[bool, str]:
    """Validate that developer output addresses each acceptance criterion.

    Parameters
    ----------
    draft:
        The developer-produced output text.
    criteria:
        Ordered list of acceptance criterion strings from the sprint task.

    Returns
    -------
    (True,  evidence_str)  when all criteria are addressed or list is empty.
    (False, evidence_str)  when one or more criteria are not addressed.
    """
    if not criteria:
        return True, "No acceptance criteria defined — gate skipped."

    draft_lower = draft.lower()
    failed: list[str] = []

    for criterion in criteria:
        # Significant words: pure alpha tokens > 3 chars (handles both
        # ASCII and extended Latin used in Turkish/EN mixed text)
        key_words = re.findall(r"[a-zA-Z\u00c0-\u024f]{4,}", criterion.lower())
        if not key_words:
            continue  # nothing to check — skip this criterion

        matched = sum(1 for w in key_words if w in draft_lower)
        ratio   = matched / len(key_words)

        if ratio < MATCH_RATIO:
            failed.append(criterion)

    total = len(criteria)
    if not failed:
        return True, f"All {total} acceptance criteria addressed in developer output."

    fail_count = len(failed)
    snippets   = "; ".join(f"[{c[:60]}]" for c in failed[:MAX_REPORTED])
    return (
        False,
        f"{fail_count}/{total} acceptance criteria not addressed: {snippets}",
    )
