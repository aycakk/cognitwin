"""
test_sprint_ui_bridge.py — UI → run_sprint bridge (cognitwin-sprint model).

Covers:
  A. Router maps cognitwin-sprint → mode="sprint"
  B. model_access registers cognitwin-sprint as agile-visible
  C. process_user_message(model="cognitwin-sprint") routes to run_sprint
  D. Existing model names still route to their old handlers (backward compat)
  E. Report rendering contains Backlog / Roadmap / Increment / C8 / PO review
  F. /v1/chat/completions with model="cognitwin-sprint" calls the bridge
  G. /v1/sprint/run direct endpoint works
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch, MagicMock

import pytest


# ─────────────────────────────────────────────────────────────────────────────
#  A. Router
# ─────────────────────────────────────────────────────────────────────────────

class TestRouter:
    def test_cognitwin_sprint_maps_to_sprint_mode(self):
        from src.pipeline.router import resolve_mode
        mode, strategy = resolve_mode("cognitwin-sprint")
        assert mode == "sprint"
        assert strategy == "auto"

    def test_existing_models_unchanged(self):
        from src.pipeline.router import resolve_mode
        assert resolve_mode("cognitwin-developer")[0]     == "developer"
        assert resolve_mode("cognitwin-product-owner")[0] == "product_owner"
        assert resolve_mode("cognitwin-scrum")[0]         == "scrum_master"
        assert resolve_mode("cognitwin-composer")[0]      == "composer"
        assert resolve_mode("llama3.2")[0]                == "student"


# ─────────────────────────────────────────────────────────────────────────────
#  B. Model access
# ─────────────────────────────────────────────────────────────────────────────

class TestModelAccess:
    def test_sprint_model_visible_to_agile(self):
        pytest.importorskip("fastapi")
        from src.services.api.model_access import is_model_allowed, get_models_for_role
        assert is_model_allowed("agile", "cognitwin-sprint") is True
        assert is_model_allowed("admin", "cognitwin-sprint") is True
        ids = {m["id"] for m in get_models_for_role("agile")}
        assert "cognitwin-sprint" in ids

    def test_sprint_model_hidden_from_student(self):
        pytest.importorskip("fastapi")
        from src.services.api.model_access import is_model_allowed
        assert is_model_allowed("student", "cognitwin-sprint") is False


# ─────────────────────────────────────────────────────────────────────────────
#  C + D. process_user_message routing
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _FakeResult:
    sprint_id: str = "sprint-test"
    goal: str = "Add login"
    completed_stories: list = field(default_factory=list)
    blocked_stories:   list = field(default_factory=list)
    total_steps: int = 3
    avg_confidence: float = 0.9
    summary: str = "ok"


class TestProcessUserMessageRouting:

    def _skip_unless_deps(self):
        pytest.importorskip("ollama")
        pytest.importorskip("fastapi")

    def test_sprint_model_invokes_run_sprint(self):
        self._skip_unless_deps()
        from src.services.api import pipeline as pipeline_mod

        fake_result = _FakeResult(goal="Add login")
        with patch("src.services.api.sprint_bridge.run_sprint", return_value=fake_result) as run_sp, \
             patch("src.services.api.sprint_bridge.SprintStateStore") as store_cls:
            fake_store = MagicMock()
            fake_store.load.return_value = {
                "backlog": [],
                "roadmap": [],
                "tasks": [],
                "increment": [],
                "retro_actions": [],
                "meeting_notes": [],
                "sprint": {"goal": "Add login"},
                "product_goal": "",
            }
            fake_store.get_increment.return_value = []
            store_cls.return_value = fake_store

            result = pipeline_mod.process_user_message(
                "Add login feature", model="cognitwin-sprint"
            )
            run_sp.assert_called_once()
            assert "sprint-test" in result["answer"]
            assert result["workflow_meta"]["sprint_id"] == "sprint-test"

    def test_developer_model_still_calls_old_handler(self):
        """Guard: existing developer routing must not regress."""
        self._skip_unless_deps()
        from src.services.api import pipeline as pipeline_mod

        fake_resp = MagicMock()
        fake_resp.draft = "dev answer"
        fake_resp.metadata = {}
        with patch.object(pipeline_mod, "_process_developer_message", return_value=fake_resp) as dev, \
             patch("src.services.api.sprint_bridge.run_sprint") as run_sp:
            pipeline_mod.process_user_message("hi", model="cognitwin-developer")
            dev.assert_called_once()
            run_sp.assert_not_called()

    def test_product_owner_model_still_calls_old_handler(self):
        self._skip_unless_deps()
        from src.services.api import pipeline as pipeline_mod

        fake_resp = MagicMock()
        fake_resp.draft = "po answer"
        fake_resp.metadata = {}
        with patch.object(pipeline_mod, "run_product_owner_pipeline", return_value=fake_resp), \
             patch("src.pipeline.agile_workflow.is_workflow_request", return_value=False), \
             patch("src.services.api.sprint_bridge.run_sprint") as run_sp:
            pipeline_mod.process_user_message("backlog listele", model="cognitwin-product-owner")
            run_sp.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
#  E. Report rendering
# ─────────────────────────────────────────────────────────────────────────────

class TestReportRendering:

    def test_report_contains_required_sections(self, tmp_path):
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
        from src.services.api.sprint_bridge import render_sprint_report

        store = SprintStateStore(state_path=tmp_path / "state.json")
        store.set_product_goal("Ship auth MVP")
        sid = store.add_story(
            title="Login", acceptance_criteria=["login works"], priority="high"
        )
        tid = store.promote_story_to_sprint_task(sid)
        store.assign_task(tid, "developer-default")
        store.start_task(tid)
        store.mark_ac_validated(tid)
        store.complete_task(tid, "Login implemented. Login works.")
        store.accept_story(sid)
        store.add_to_increment(tid)
        store.add_roadmap_entry({
            "title": "MVP auth", "target_sprint": "sprint-1", "status": "planned",
        })
        store.add_retro_actions(["Write tests first"])

        result = _FakeResult(
            goal="Add login",
            completed_stories=[{"story_id": sid, "task_id": tid, "title": "Login"}],
            blocked_stories=[],
        )

        text = render_sprint_report(result, store, executed_task_ids={tid})
        assert "Product Backlog" in text
        assert "Roadmap" in text
        assert "Sprint Goal" in text
        assert "Product Increment" in text
        assert "C8=PASS" in text
        assert sid in text
        assert tid in text
        assert "Write tests first" in text

    def test_report_shows_blocked_with_missing_criteria(self, tmp_path):
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
        from src.services.api.sprint_bridge import render_sprint_report

        store = SprintStateStore(state_path=tmp_path / "state.json")
        result = _FakeResult(
            blocked_stories=[{
                "story_id": "S-001", "task_id": "T-001",
                "title": "x", "reason": "PO rejected: bad output",
                "missing_criteria": ["rotate passwords every 90 days"],
            }],
        )
        text = render_sprint_report(result, store, executed_task_ids=set())
        assert "PO rejected" in text
        assert "rotate passwords" in text


# ─────────────────────────────────────────────────────────────────────────────
#  F. /v1/chat/completions with model=cognitwin-sprint
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch):
    """Build a TestClient; inject a legit agile key."""
    pytest.importorskip("fastapi")
    pytest.importorskip("ollama")
    monkeypatch.setenv("COGNITWIN_AGILE_KEY", "test-agile-key")
    # Force module re-import so the key map is rebuilt
    import importlib
    import src.services.api.model_access as ma_mod
    importlib.reload(ma_mod)
    import src.services.api.openai_routes as routes_mod
    importlib.reload(routes_mod)

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(routes_mod.openai_router)
    return TestClient(app)


class TestHTTPEndpoints:

    def test_chat_completions_with_sprint_model(self, client):
        fake_result = _FakeResult(goal="Add login")
        with patch("src.services.api.sprint_bridge.run_sprint", return_value=fake_result), \
             patch("src.services.api.sprint_bridge.SprintStateStore") as store_cls:
            store = MagicMock()
            store.load.return_value = {
                "backlog": [], "roadmap": [], "tasks": [],
                "increment": [], "retro_actions": [], "meeting_notes": [],
                "sprint": {"goal": "x"}, "product_goal": "",
            }
            store.get_increment.return_value = []
            store_cls.return_value = store

            resp = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer test-agile-key"},
                json={
                    "model": "cognitwin-sprint",
                    "messages": [{"role": "user", "content": "Add login"}],
                    "stream": False,
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        assert "COGNITWIN SPRINT RUN" in content
        assert "sprint-test" in content

    def test_sprint_run_direct_endpoint(self, client):
        fake_result = _FakeResult(goal="Add login")
        with patch("src.services.api.sprint_bridge.run_sprint", return_value=fake_result), \
             patch("src.services.api.sprint_bridge.SprintStateStore") as store_cls:
            store = MagicMock()
            store.load.return_value = {
                "backlog": [], "roadmap": [], "tasks": [],
                "increment": [], "retro_actions": [], "meeting_notes": [],
                "sprint": {"goal": "x"}, "product_goal": "",
            }
            store.get_increment.return_value = []
            store_cls.return_value = store

            resp = client.post(
                "/v1/sprint/run",
                headers={"Authorization": "Bearer test-agile-key"},
                json={"goal": "Add login", "reset_state": False},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert body["workflow_meta"]["sprint_id"] == "sprint-test"
        assert "COGNITWIN SPRINT RUN" in body["answer"]

    def test_sprint_run_denied_for_student_role(self, client):
        resp = client.post(
            "/v1/sprint/run",
            headers={"Authorization": "Bearer cognitwin-student"},
            json={"goal": "anything"},
        )
        assert resp.status_code == 403
