"""Model visibility and API-key based access control for LibreChat endpoints."""
from __future__ import annotations

import os

ALL_MODEL_IDS: list[str] = [
    "llama3.2",
    "cognitwin-student-llm",
    "cognitwin-developer",
    "cognitwin-scrum",
    "cognitwin-product-owner",
    "cognitwin-composer",
    "cognitwin-hr",
]

_STUDENT_MODELS = ["cognitwin-student-llm"]
_AGILE_MODELS = [
    "cognitwin-developer",
    "cognitwin-scrum",
    "cognitwin-product-owner",
    "cognitwin-composer",
]
_HR_MODELS = ["cognitwin-hr"]


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def _keys() -> dict[str, str]:
    return {
        "student": _env("COGNITWIN_STUDENT_KEY", "cognitwin-student"),
        "agile": _env("COGNITWIN_AGILE_KEY", "cognitwin-agile"),
        "hr": _env("COGNITWIN_HR_KEY", "cognitwin-hr"),
        "legacy": _env("COGNITWIN_LEGACY_KEY", "cognitwin"),
    }


def _extract_token(headers: dict[str, str]) -> str:
    auth = headers.get("authorization", "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return headers.get("x-api-key", "").strip()


def visible_models_for_headers(headers: dict[str, str]) -> list[str]:
    """Return the model list visible to the caller by API key.

    Unknown tokens get a minimal public-safe list so model visibility is not widened.
    """
    token = _extract_token(headers)
    keys = _keys()

    if token == keys["student"]:
        return list(_STUDENT_MODELS)
    if token == keys["agile"]:
        return list(_AGILE_MODELS)
    if token == keys["hr"]:
        return list(_HR_MODELS)
    if token == keys["legacy"]:
        return list(ALL_MODEL_IDS)

    # Conservative fallback.
    return list(_STUDENT_MODELS)
