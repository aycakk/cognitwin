"""gates/c4_hallucination.py — shared decision logic for C4.

Single source of truth for hallucination-marker detection.
Previously duplicated as:
  - pipeline.py::gate_c4_hallucination
  - student_agent.py::StudentAgent._gate_c4_synthesis

Both copies walked ASP_NEG_PATTERNS looking only for the two labels
ASP-NEG-02_HALLUCINATION and ASP-NEG-05_WEIGHT_ONLY. The two copies
differed only in the language of the returned message.

This module exposes a pure decision function returning the machine-
readable label and the matched substring. Each caller formats its own
localized message, so existing return strings are preserved
byte-for-byte.

Returns:
  (True,  None, None)               → no hallucination marker
  (False, label, matched_substring) → a marker was hit;
                                      label is "ASP-NEG-02_HALLUCINATION"
                                      or "ASP-NEG-05_WEIGHT_ONLY"
"""

from __future__ import annotations

from typing import Optional

from src.shared.patterns import ASP_NEG_PATTERNS

_HALLUCINATION_LABELS = frozenset({
    "ASP-NEG-02_HALLUCINATION",
    "ASP-NEG-05_WEIGHT_ONLY",
})


def check_hallucination(
    draft: str,
) -> tuple[bool, Optional[str], Optional[str]]:
    """Decide whether `draft` contains a hallucination / weight-only marker.

    Returns (passed, label, matched). The caller is responsible for
    formatting the user-visible message.
    """
    for label, pattern in ASP_NEG_PATTERNS:
        if label in _HALLUCINATION_LABELS:
            m = pattern.search(draft)
            if m:
                return False, label, m.group()
    return True, None, None
