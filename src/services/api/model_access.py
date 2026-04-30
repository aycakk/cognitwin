"""Role-based model visibility and access control for OpenAI-compatible routes."""

from __future__ import annotations

import os
from typing import Iterable


ROLE_STUDENT = "student"
ROLE_AGILE = "agile"
ROLE_ADMIN = "admin"

ALL_MODELS: tuple[str, ...] = (
    "llama3.2",
    "cognitwin-student-llm",
    "cognitwin-developer",
    "cognitwin-scrum",
    "cognitwin-product-owner",
    "cognitwin-composer",
    "cognitwin-buyer",
)

ROLE_MODELS: dict[str, tuple[str, ...]] = {
    ROLE_STUDENT: (
        "llama3.2",
        "cognitwin-student-llm",
    ),
    ROLE_AGILE: (
        "cognitwin-developer",
        "cognitwin-scrum",
        "cognitwin-product-owner",
        "cognitwin-composer",
        "cognitwin-buyer",
    ),
    ROLE_ADMIN: ALL_MODELS,
}


def _extract_bearer_token(authorization: str | None) -> str:
    value = (authorization or "").strip()
    if not value:
        return ""
    if not value.lower().startswith("bearer "):
        return ""
    return value[7:].strip()


def resolve_role_from_authorization(authorization: str | None) -> str:
    """Resolve caller role from Authorization header.

    Fallback is student to preserve backward compatibility.
    """
    token = _extract_bearer_token(authorization)
    if not token:
        return ROLE_STUDENT

    student_key = os.environ.get("COGNITWIN_STUDENT_KEY", "cognitwin-student").strip()
    agile_key = os.environ.get("COGNITWIN_AGILE_KEY", "cognitwin-agile").strip()
    admin_key = os.environ.get("COGNITWIN_ADMIN_KEY", "cognitwin-admin").strip()

    # Legacy key kept for compatibility.
    if token == "cognitwin":
        return ROLE_STUDENT
    if token == admin_key:
        return ROLE_ADMIN
    if token == agile_key:
        return ROLE_AGILE
    if token == student_key:
        return ROLE_STUDENT
    return ROLE_STUDENT


def models_for_role(role: str) -> tuple[str, ...]:
    return ROLE_MODELS.get(role, ROLE_MODELS[ROLE_STUDENT])


def is_model_allowed_for_role(model: str, role: str) -> bool:
    model_lower = (model or "").strip().lower()
    if not model_lower:
        return False
    return model_lower in {m.lower() for m in models_for_role(role)}


def to_model_list_payload(models: Iterable[str], now_ts: int) -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": mid,
                "object": "model",
                "created": now_ts,
                "owned_by": "cognitwin",
            }
            for mid in models
        ],
    }

