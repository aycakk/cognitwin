"""pipeline/router.py — request routing decision.

Maps the LibreChat model name to (mode, strategy), deciding which
pipeline path process_user_message will invoke.

Pure function: no imports, no I/O, no side effects.
Previously defined as _resolve_mode in src/services/api/pipeline.py;
moved here in Step 3.3 of the migration plan.
"""

from __future__ import annotations


def resolve_mode(model: str) -> tuple[str, str]:
    """
    Map the LibreChat model name to (mode, strategy).

    mode     : 'student' | 'developer' | 'scrum_master'
    strategy : 'auto' (developer) | 'llm' (student) | 'rule' (scrum_master)

    Routing order matters — check more-specific prefixes first.
    """
    model_lower = (model or "").lower()
    if "developer" in model_lower:
        return "developer", "auto"
    if "scrum" in model_lower:
        return "scrum_master", "rule"
    return "student", "llm"
