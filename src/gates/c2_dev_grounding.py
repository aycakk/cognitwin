"""gates/c2_dev_grounding.py — developer-specific grounding gate.

Unlike C2 (which checks against ChromaDB vector memory for academic
queries), C2_DEV checks that the developer draft is grounded in the
injected codebase + sprint context.

"Grounded" means ≥ 3 content words (5+ chars, not masked, not stop-words)
appear in BOTH the draft and the injected context.

When no context was injected (empty string), the draft MUST contain
the BlindSpot phrase "bulamadım" — otherwise it is fabricating from
parametric memory.

Return contract:
  (passed: bool, reason_code: str, overlap_count: int)

  reason_code values:
    "no_context_pass"    → context empty AND draft has BlindSpot phrase; PASS
    "no_context_fail"    → context empty AND draft missing BlindSpot; FAIL
    "blindspot"          → context present but draft has BlindSpot; PASS
    "overlap_pass"       → ≥ 3 shared content words; PASS
    "overlap_fail"       → < 3 shared content words; FAIL
"""

from __future__ import annotations

import re

# Stop words that should not count as grounding evidence
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
    "to", "for", "of", "and", "or", "but", "not", "with", "this",
    "that", "it", "be", "as", "by", "from", "has", "have", "had",
    "will", "would", "could", "should", "can", "may", "must",
    "been", "being", "about", "into", "over", "than", "then",
    "when", "where", "which", "while", "class", "import", "return",
    "self", "none", "true", "false", "print",
    # Turkish stop words
    "ve", "bir", "bu", "da", "de", "ile", "için", "olan", "var",
    "yok", "ise", "gibi", "daha", "sonra", "olarak", "ancak",
})

_MIN_WORD_LEN = 5
_MIN_OVERLAP = 3


def _extract_content_words(text: str) -> set[str]:
    """Extract meaningful content words from text."""
    words = set()
    for w in re.findall(r"\b\w+\b", text.lower()):
        if (
            len(w) >= _MIN_WORD_LEN
            and w not in _STOP_WORDS
            and not re.match(r"\[.*_MASKED\]", w, re.I)
            and not w.isdigit()
        ):
            words.add(w)
    return words


def check_dev_grounding(
    draft: str,
    codebase_context: str,
    context_empty: bool,
) -> tuple[bool, str, int]:
    """Decide whether `draft` is grounded in developer codebase context.

    Parameters
    ----------
    draft:             the LLM-produced response text
    codebase_context:  combined codebase + sprint context block
    context_empty:     True when no codebase/sprint context was injected

    Returns (passed, reason_code, overlap_count).
    """
    if context_empty:
        if "bulamadım" in draft.lower():
            return True, "no_context_pass", 0
        return False, "no_context_fail", 0

    if "bulamadım" in draft.lower():
        return True, "blindspot", 0

    context_words = _extract_content_words(codebase_context)
    draft_words = _extract_content_words(draft)
    overlap = context_words & draft_words

    if len(overlap) >= _MIN_OVERLAP:
        return True, "overlap_pass", len(overlap)
    return False, "overlap_fail", len(overlap)
