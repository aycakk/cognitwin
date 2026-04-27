"""services/api/model_access.py — role-based model visibility and access control.

Single source of truth for which models are accessible by which role.
Consulted by BOTH /v1/models (discovery) and /v1/chat/completions (enforcement).

How it works
────────────
LibreChat sends the `apiKey` from librechat.yaml as:
    Authorization: Bearer <api_key>

The backend reads that header, maps it to a role, and either:
  • returns a filtered model list  (/v1/models)
  • accepts or rejects the request (/v1/chat/completions)

Roles
─────
  student  — academic / Q&A path only.  Sees cognitwin-student-llm.
  agile    — project workflow users.    Sees PO, SM, Developer, Composer.
  admin    — sees everything.

API key configuration
─────────────────────
Set these in your .env (defaults work for local development):
  COGNITWIN_STUDENT_KEY   default: cognitwin-student
  COGNITWIN_AGILE_KEY     default: cognitwin-agile
  COGNITWIN_ADMIN_KEY     default: cognitwin-admin

The legacy key "cognitwin" (existing deployments) maps to "student" for
backward compatibility.

Security guarantee
──────────────────
Even if a user manually types an agile model name in the student endpoint,
the backend reads the Bearer token, sees "student" role, and returns HTTP 403.
LibreChat cannot override this — the backend is the enforcement layer.
"""

from __future__ import annotations

import logging
import os

from fastapi import Request

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Model registry
#  Add new models here — nowhere else.
#  "group" must be one of: "student", "agile"
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_REGISTRY: list[dict] = [
    # ── Student / academic path ────────────────────────────────────────────
    {
        "id":    "cognitwin-student-llm",
        "group": "student",
        "label": "CogniTwin Student",
    },
    {
        "id":    "llama3.2",
        "group": "student",
        "label": "Llama 3.2 (Student)",
    },
    # ── Agile / project-workflow path ─────────────────────────────────────
    {
        "id":    "cognitwin-product-owner",
        "group": "agile",
        "label": "CogniTwin Product Owner",
    },
    {
        "id":    "cognitwin-scrum",
        "group": "agile",
        "label": "CogniTwin Scrum Master",
    },
    {
        "id":    "cognitwin-developer",
        "group": "agile",
        "label": "CogniTwin Developer",
    },
    {
        "id":    "cognitwin-composer",
        "group": "agile",
        "label": "CogniTwin Composer",
    },
    {
        "id":    "cognitwin-sprint",
        "group": "agile",
        "label": "CogniTwin Sprint (autonomous)",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
#  Role → model group mapping
# ─────────────────────────────────────────────────────────────────────────────

_ROLE_GROUPS: dict[str, frozenset[str]] = {
    "student": frozenset({"student"}),
    "agile":   frozenset({"agile"}),
    "admin":   frozenset({"student", "agile"}),
}

# Pre-built per-role model ID sets for O(1) access-check lookups.
# Computed once at import time — no repeated list comprehension in hot path.
_ROLE_MODEL_IDS: dict[str, frozenset[str]] = {
    role: frozenset(
        m["id"] for m in _MODEL_REGISTRY
        if m["group"] in groups
    )
    for role, groups in _ROLE_GROUPS.items()
}

# ─────────────────────────────────────────────────────────────────────────────
#  API key → role mapping
#  Built at import time from environment variables.
# ─────────────────────────────────────────────────────────────────────────────

def _build_key_role_map() -> dict[str, str]:
    """Read API keys from environment and return key→role dict.

    Called once at module import.  Changing env vars requires restart.
    """
    student_key = os.environ.get("COGNITWIN_STUDENT_KEY", "cognitwin-student")
    agile_key   = os.environ.get("COGNITWIN_AGILE_KEY",   "cognitwin-agile")
    admin_key   = os.environ.get("COGNITWIN_ADMIN_KEY",   "cognitwin-admin")

    mapping: dict[str, str] = {
        student_key: "student",
        agile_key:   "agile",
        admin_key:   "admin",
        # Legacy key — keeps existing deployments working without config change
        "cognitwin": "student",
    }
    logger.info(
        "model-access: key map initialised — %d keys registered "
        "(student=%r agile=%r admin=%r)",
        len(mapping), student_key, agile_key, admin_key,
    )
    return mapping


# Singleton — built once, shared by all requests
_KEY_ROLE_MAP: dict[str, str] = _build_key_role_map()


# ─────────────────────────────────────────────────────────────────────────────
#  Public API — called by openai_routes.py
# ─────────────────────────────────────────────────────────────────────────────

def get_role(request: Request) -> str:
    """Resolve the caller's role from the Authorization: Bearer <key> header.

    LibreChat sends the endpoint apiKey as the Bearer token on every request.
    Unknown or missing keys default to "student" (most restrictive fallback).

    This is the single extraction point — call it once per request at the
    start of the handler, then pass the returned role string downstream.
    """
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    else:
        token = auth.strip()

    role = _KEY_ROLE_MAP.get(token, "student")

    if token and token not in _KEY_ROLE_MAP:
        # Log a warning so admins can spot misconfigured LibreChat endpoints
        # without exposing the full key (security: truncate to 8 chars).
        logger.warning(
            "model-access: unrecognised API key (first 8 chars: %r) "
            "→ defaulting to role='student'",
            token[:8],
        )

    logger.debug("model-access: resolved role=%r  token_prefix=%r", role, token[:8] if token else "")
    return role


def get_models_for_role(role: str) -> list[dict]:
    """Return all model metadata entries visible to *role*.

    Used by GET /v1/models to build the filtered response payload.
    Always returns at least the student models (safe fallback for unknown roles).
    """
    groups = _ROLE_GROUPS.get(role, frozenset({"student"}))
    return [m for m in _MODEL_REGISTRY if m["group"] in groups]


def is_model_allowed(role: str, model_id: str) -> bool:
    """Return True iff *model_id* is accessible to *role*.

    Used by POST /v1/chat/completions to block unauthorized model usage.
    Agile models are invisible and inaccessible to student-role requests,
    even if the model name is guessed manually.
    """
    allowed = _ROLE_MODEL_IDS.get(role, _ROLE_MODEL_IDS["student"])
    return model_id in allowed
