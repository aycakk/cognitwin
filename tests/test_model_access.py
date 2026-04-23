"""tests/test_model_access.py — unit tests for role-based model access control.

Tests cover:
  • Role resolution from API key / Authorization header
  • Model visibility per role (GET /v1/models behaviour)
  • Model access enforcement (POST /v1/chat/completions behaviour)
  • Student cannot see or access agile models
  • Agile role sees agile models but not student models
  • Admin sees all models
  • Unknown key defaults to student (most restrictive)
  • Legacy "cognitwin" key still works

These are pure unit tests — no FastAPI app, no Ollama, no Docker required.
The Request object is mocked with a minimal headers dict.
"""

from __future__ import annotations

import os
import importlib
from typing import Any
from unittest.mock import MagicMock

import pytest


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_request(api_key: str | None) -> Any:
    """Build a minimal mock of fastapi.Request with Authorization header."""
    mock = MagicMock()
    if api_key is None:
        mock.headers = {}
    else:
        mock.headers = {"Authorization": f"Bearer {api_key}"}
    return mock


def _reload_model_access(**env_overrides: str):
    """Reload model_access with patched env vars and return fresh module."""
    # Patch os.environ for the duration of this call
    original = {k: os.environ.get(k) for k in env_overrides}
    for k, v in env_overrides.items():
        os.environ[k] = v
    try:
        import src.services.api.model_access as ma
        importlib.reload(ma)
        return ma
    finally:
        for k, orig_v in original.items():
            if orig_v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = orig_v


# Use the default module for most tests (no env override needed)
import src.services.api.model_access as model_access


# ─────────────────────────────────────────────────────────────────────────────
#  Role resolution tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGetRole:
    def test_student_key_returns_student(self):
        req = _make_request("cognitwin-student")
        assert model_access.get_role(req) == "student"

    def test_agile_key_returns_agile(self):
        req = _make_request("cognitwin-agile")
        assert model_access.get_role(req) == "agile"

    def test_admin_key_returns_admin(self):
        req = _make_request("cognitwin-admin")
        assert model_access.get_role(req) == "admin"

    def test_legacy_cognitwin_key_returns_student(self):
        """The old single apiKey must keep working as student role."""
        req = _make_request("cognitwin")
        assert model_access.get_role(req) == "student"

    def test_unknown_key_defaults_to_student(self):
        req = _make_request("totally-random-key-xyz")
        assert model_access.get_role(req) == "student"

    def test_no_key_defaults_to_student(self):
        req = _make_request(None)
        assert model_access.get_role(req) == "student"

    def test_empty_key_defaults_to_student(self):
        req = _make_request("")
        assert model_access.get_role(req) == "student"

    def test_bearer_prefix_stripped(self):
        """Authorization header without 'Bearer ' prefix still works."""
        mock = MagicMock()
        mock.headers = {"Authorization": "cognitwin-agile"}
        assert model_access.get_role(mock) == "agile"


# ─────────────────────────────────────────────────────────────────────────────
#  Model visibility tests (get_models_for_role)
# ─────────────────────────────────────────────────────────────────────────────

class TestGetModelsForRole:
    def _ids(self, role: str) -> set[str]:
        return {m["id"] for m in model_access.get_models_for_role(role)}

    def test_student_sees_student_models(self):
        ids = self._ids("student")
        assert "cognitwin-student-llm" in ids
        assert "llama3.2" in ids

    def test_student_does_not_see_agile_models(self):
        ids = self._ids("student")
        assert "cognitwin-product-owner" not in ids
        assert "cognitwin-scrum"         not in ids
        assert "cognitwin-developer"     not in ids
        assert "cognitwin-composer"      not in ids

    def test_agile_sees_all_agile_models(self):
        ids = self._ids("agile")
        assert "cognitwin-product-owner" in ids
        assert "cognitwin-scrum"         in ids
        assert "cognitwin-developer"     in ids
        assert "cognitwin-composer"      in ids

    def test_agile_does_not_see_student_models(self):
        ids = self._ids("agile")
        assert "cognitwin-student-llm" not in ids
        assert "llama3.2"              not in ids

    def test_admin_sees_all_models(self):
        ids = self._ids("admin")
        assert "cognitwin-student-llm"   in ids
        assert "llama3.2"                in ids
        assert "cognitwin-product-owner" in ids
        assert "cognitwin-scrum"         in ids
        assert "cognitwin-developer"     in ids
        assert "cognitwin-composer"      in ids

    def test_unknown_role_falls_back_to_student(self):
        ids = self._ids("unknown-role-xyz")
        assert "cognitwin-student-llm" in ids
        assert "cognitwin-product-owner" not in ids

    def test_returns_list_of_dicts(self):
        result = model_access.get_models_for_role("student")
        assert isinstance(result, list)
        for item in result:
            assert "id" in item
            assert "group" in item


# ─────────────────────────────────────────────────────────────────────────────
#  Access enforcement tests (is_model_allowed)
# ─────────────────────────────────────────────────────────────────────────────

class TestIsModelAllowed:
    # ── Student access ────────────────────────────────────────────────────
    def test_student_can_access_student_llm(self):
        assert model_access.is_model_allowed("student", "cognitwin-student-llm") is True

    def test_student_can_access_llama32(self):
        assert model_access.is_model_allowed("student", "llama3.2") is True

    def test_student_cannot_access_product_owner(self):
        assert model_access.is_model_allowed("student", "cognitwin-product-owner") is False

    def test_student_cannot_access_scrum(self):
        assert model_access.is_model_allowed("student", "cognitwin-scrum") is False

    def test_student_cannot_access_developer(self):
        assert model_access.is_model_allowed("student", "cognitwin-developer") is False

    def test_student_cannot_access_composer(self):
        assert model_access.is_model_allowed("student", "cognitwin-composer") is False

    # ── Agile access ──────────────────────────────────────────────────────
    def test_agile_can_access_product_owner(self):
        assert model_access.is_model_allowed("agile", "cognitwin-product-owner") is True

    def test_agile_can_access_scrum(self):
        assert model_access.is_model_allowed("agile", "cognitwin-scrum") is True

    def test_agile_can_access_developer(self):
        assert model_access.is_model_allowed("agile", "cognitwin-developer") is True

    def test_agile_can_access_composer(self):
        assert model_access.is_model_allowed("agile", "cognitwin-composer") is True

    def test_agile_cannot_access_student_llm(self):
        """Agile users must not be routed to the student path."""
        assert model_access.is_model_allowed("agile", "cognitwin-student-llm") is False

    def test_agile_cannot_access_llama32(self):
        assert model_access.is_model_allowed("agile", "llama3.2") is False

    # ── Admin access ──────────────────────────────────────────────────────
    def test_admin_can_access_student_llm(self):
        assert model_access.is_model_allowed("admin", "cognitwin-student-llm") is True

    def test_admin_can_access_product_owner(self):
        assert model_access.is_model_allowed("admin", "cognitwin-product-owner") is True

    def test_admin_can_access_developer(self):
        assert model_access.is_model_allowed("admin", "cognitwin-developer") is True

    # ── Guessed model names blocked ───────────────────────────────────────
    def test_student_cannot_guess_agile_model_directly(self):
        """Backend blocks access even if user manually types the model name."""
        for model in ("cognitwin-product-owner", "cognitwin-scrum",
                      "cognitwin-developer", "cognitwin-composer"):
            assert model_access.is_model_allowed("student", model) is False, \
                f"student should NOT be able to access {model!r}"

    def test_unknown_model_is_denied(self):
        assert model_access.is_model_allowed("admin", "non-existent-model") is False
        assert model_access.is_model_allowed("student", "non-existent-model") is False


# ─────────────────────────────────────────────────────────────────────────────
#  End-to-end role+model scenario tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndScenarios:
    """Simulate the full auth → visibility → access flow."""

    def test_student_login_flow(self):
        """Student uses cognitwin-student key → sees only student models → cannot call agile."""
        req  = _make_request("cognitwin-student")
        role = model_access.get_role(req)
        assert role == "student"

        visible = {m["id"] for m in model_access.get_models_for_role(role)}
        assert "cognitwin-student-llm" in visible
        assert "cognitwin-product-owner" not in visible

        # Backend blocks even if model name is guessed
        assert model_access.is_model_allowed(role, "cognitwin-product-owner") is False
        assert model_access.is_model_allowed(role, "cognitwin-student-llm")   is True

    def test_agile_user_login_flow(self):
        """Project user with agile key → sees agile models → can call all agile agents."""
        req  = _make_request("cognitwin-agile")
        role = model_access.get_role(req)
        assert role == "agile"

        visible = {m["id"] for m in model_access.get_models_for_role(role)}
        assert "cognitwin-product-owner" in visible
        assert "cognitwin-scrum"         in visible
        assert "cognitwin-developer"     in visible
        assert "cognitwin-composer"      in visible
        assert "cognitwin-student-llm"   not in visible

        for mid in ("cognitwin-product-owner", "cognitwin-scrum",
                    "cognitwin-developer", "cognitwin-composer"):
            assert model_access.is_model_allowed(role, mid) is True

    def test_legacy_key_still_works(self):
        """The old 'cognitwin' key keeps working as student role."""
        req  = _make_request("cognitwin")
        role = model_access.get_role(req)
        assert role == "student"
        assert model_access.is_model_allowed(role, "cognitwin-student-llm") is True
        assert model_access.is_model_allowed(role, "cognitwin-developer")   is False

    def test_custom_env_keys(self):
        """API keys can be changed via environment variables."""
        ma = _reload_model_access(
            COGNITWIN_STUDENT_KEY="my-student-secret",
            COGNITWIN_AGILE_KEY="my-agile-secret",
            COGNITWIN_ADMIN_KEY="my-admin-secret",
        )
        req_student = _make_request("my-student-secret")
        req_agile   = _make_request("my-agile-secret")
        req_admin   = _make_request("my-admin-secret")

        assert ma.get_role(req_student) == "student"
        assert ma.get_role(req_agile)   == "agile"
        assert ma.get_role(req_admin)   == "admin"
