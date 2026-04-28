"""pipeline/router.py — request routing decision.

Maps the LibreChat model name to (mode, strategy), deciding which
pipeline path process_user_message will invoke.

Pure function: no I/O, no side effects.
Previously defined as _resolve_mode in src/services/api/pipeline.py;
moved here in Step 3.3 of the migration plan.
"""

from __future__ import annotations

import logging

from src.core.exceptions import UnknownRoleError

logger = logging.getLogger(__name__)


# Keep the old name as an alias so existing callers (pipeline.py, tests) do
# not need to be updated immediately.
UnknownModelError = UnknownRoleError


# ---------------------------------------------------------------------------
# Known routing tokens.  Add new model families here — nowhere else.
# ---------------------------------------------------------------------------
_ROUTING_TABLE: list[tuple[str, str, str]] = [
    # (substring_to_match, mode, strategy)
    # "product_owner" must precede "scrum" to prevent false substring matches.
    # "hr" must precede generic fallbacks.
    ("composer",      "composer",      "rule"),
    ("product_owner", "product_owner", "rule"),
    ("developer",     "developer",     "auto"),
    ("scrum",         "scrum_master",  "rule"),
    ("student",       "student",       "llm"),
    ("hr",            "hr",            "llm"),
]

# Models that are explicitly recognised as valid student-path models
# (no routing keyword needed — these are Ollama base models).
_KNOWN_BASE_MODELS: frozenset[str] = frozenset({
    "llama3.2",
    "llama3",
    "llama2",
    "mistral",
    "gemma",
    "cognitwin",
})


class UnknownModelError(ValueError):
    """Raised when the model name cannot be mapped to any known agent."""


def resolve_mode(model: str) -> tuple[str, str]:
    """
    Map the LibreChat model name to (mode, strategy).

    mode     : 'student' | 'developer' | 'scrum_master' | 'product_owner' | 'composer' | 'hr'
    strategy : 'auto' (developer) | 'llm' (student) | 'rule' (scrum_master, product_owner, composer)

    Raises UnknownModelError if the name matches nothing in the routing
    table AND is not a known base model.  The caller decides how to surface
    this to the user; it must NOT be silently swallowed.

    Routing order matters — more-specific tokens are checked first.
    """
    model_lower = (model or "").strip().lower()

    # 1. Check explicit routing tokens
    #    Normalise hyphens to underscores so that both
    #    "cognitwin-product-owner" and "product_owner" match the same entry.
    model_normalised = model_lower.replace("-", "_")
    for token, mode, strategy in _ROUTING_TABLE:
        if token in model_normalised:
            logger.info("router: model=%r → mode=%s strategy=%s", model, mode, strategy)
            return mode, strategy

    # 2. Accept bare Ollama base model names as student path
    for base in _KNOWN_BASE_MODELS:
        if model_lower.startswith(base):
            logger.info(
                "router: model=%r matched base=%r → mode=student strategy=llm",
                model, base,
            )
            return "student", "llm"

    # 3. Empty / None → default to student but log a warning (LibreChat sends
    #    empty model strings in some configurations)
    if not model_lower:
        logger.warning("router: empty model name received → defaulting to student")
        return "student", "llm"

    # 4. Completely unknown — raise so the caller can return a visible error
    logger.error("router: unrecognised model=%r — no route found", model)
    raise UnknownModelError(
        f"Model {model!r} does not match any known agent. "
        "Use a model name containing 'composer', 'product_owner', "
        "'developer', 'scrum', 'student', or 'hr', "
        "or a recognised base model (llama3.2, mistral, …)."
    )
