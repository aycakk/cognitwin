"""gates/c6_anti_sycophancy.py — shared decision logic for C6.

Single source of truth for the full ASP-NEG pattern sweep.
Previously duplicated as:
  - pipeline.py::gate_c6_anti_sycophancy
  - student_agent.py::StudentAgent._gate_c6_anti_sycophancy

Both copies walked every entry in ASP_NEG_PATTERNS and collected all
matches into a single violation report. They differed only in:
  • The language of the error prefix
  • Whether each violation was rendered as "[label] 'match'"
    (pipeline) or just "'match'" (student)

This module exposes a pure decision function returning the list of
(label, matched_substring) pairs. Each caller formats its own
localized message, so existing return strings are preserved
byte-for-byte.

Returns:
  (True,  [])                         → no violations
  (False, [(label, match), ...])     → one or more ASP-NEG patterns matched
"""

from __future__ import annotations

from src.shared.patterns import ASP_NEG_PATTERNS


def check_anti_sycophancy(
    draft: str,
) -> tuple[bool, list[tuple[str, str]]]:
    """Decide whether `draft` contains any ASP-NEG pattern match.

    Returns (passed, violations) where violations is a list of
    (label, matched_substring) pairs in pattern-registry order.
    The caller is responsible for formatting the user-visible message.
    """
    violations: list[tuple[str, str]] = []
    for label, pattern in ASP_NEG_PATTERNS:
        m = pattern.search(draft)
        if m:
            violations.append((label, m.group()))
    return (not violations), violations
