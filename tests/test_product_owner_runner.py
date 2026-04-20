"""
test_product_owner_runner.py — Unit tests for the Product Owner pipeline runner.

Covers: happy path, PII blocking, hallucination sanitization.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.core.schemas import AgentTask, AgentRole, TaskStatus


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_store(tmp_path):
    from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
    state_file = tmp_path / "sprint_state.json"
    store = SprintStateStore(state_path=state_file)
    store.save({
        "sprint": {"id": "sprint-test", "goal": "Test", "start": "2026-04-07",
                    "end": "2026-04-21", "velocity": 20},
        "tasks": [],
        "backlog": [],
        "team": [{"id": "developer-default", "role": "Developer", "capacity": 8}],
    })
    return store


@pytest.fixture
def runner(tmp_store):
    """Import runner with patched singletons pointing to tmp store."""
    from src.agents.product_owner_agent import ProductOwnerAgent
    import src.pipeline.product_owner_runner as runner_mod

    runner_mod._SPRINT_STATE = tmp_store
    runner_mod._agent = ProductOwnerAgent(state_store=tmp_store)
    return runner_mod


# ─────────────────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestProductOwnerRunner:

    def test_happy_path_create_story(self, runner):
        task = AgentTask(
            session_id="test-session",
            role=AgentRole.PRODUCT_OWNER,
            masked_input="hikaye oluştur: API refactoring",
        )
        response = runner.run_product_owner_pipeline(task)

        assert response.status == TaskStatus.COMPLETED
        assert response.agent_role == AgentRole.PRODUCT_OWNER
        assert "S-001" in response.draft
        assert "API refactoring" in response.draft

    def test_happy_path_backlog_empty(self, runner):
        task = AgentTask(
            session_id="test-session",
            role=AgentRole.PRODUCT_OWNER,
            masked_input="backlog listele",
        )
        response = runner.run_product_owner_pipeline(task)

        assert response.status == TaskStatus.COMPLETED
        assert "boş" in response.draft.lower()

    def test_pii_blocked(self, runner):
        """If the agent output somehow contains PII, it should be blocked."""
        original_handle = runner._agent.handle_query

        def fake_handle(query):
            return "Kişi: 12345678901 numaralı öğrenci"

        runner._agent.handle_query = fake_handle
        try:
            task = AgentTask(
                session_id="test-session",
                role=AgentRole.PRODUCT_OWNER,
                masked_input="backlog listele",
            )
            response = runner.run_product_owner_pipeline(task)
            assert response.status == TaskStatus.FAILED
            assert "kişisel veri" in response.draft.lower() or "PII" in response.draft
        finally:
            runner._agent.handle_query = original_handle

    def test_hallucination_sanitized(self, runner):
        """Hedging phrases should be replaced with [doğrulanmamış]."""
        original_handle = runner._agent.handle_query

        def fake_handle(query):
            return "Sanırım bu hikaye hazır."

        runner._agent.handle_query = fake_handle
        try:
            task = AgentTask(
                session_id="test-session",
                role=AgentRole.PRODUCT_OWNER,
                masked_input="backlog durumu",
            )
            response = runner.run_product_owner_pipeline(task)
            assert response.status == TaskStatus.COMPLETED
            assert "[doğrulanmamış]" in response.draft
            assert "Sanırım" not in response.draft
        finally:
            runner._agent.handle_query = original_handle

    def test_task_id_echoed(self, runner):
        task = AgentTask(
            task_id="test-task-123",
            session_id="test-session",
            role=AgentRole.PRODUCT_OWNER,
            masked_input="backlog durumu",
        )
        response = runner.run_product_owner_pipeline(task)
        assert response.task_id == "test-task-123"


# ─────────────────────────────────────────────────────────────────────────────
#  Turkish planning intent detection smoke tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRoutingDispatch:
    """Verify the runner dispatches to the correct path without calling Ollama."""

    @pytest.fixture(autouse=True)
    def _patch_store(self, tmp_store):
        import src.pipeline.product_owner_runner as runner_mod
        from src.agents.product_owner_agent import ProductOwnerAgent
        runner_mod._SPRINT_STATE = tmp_store
        runner_mod._agent = ProductOwnerAgent(state_store=tmp_store)

    def test_english_planning_prompt_calls_llm_path(self, tmp_store):
        """'create epics for X' must reach _run_llm_planning, not the rule engine."""
        import src.pipeline.product_owner_runner as runner_mod
        with patch.object(runner_mod, "_run_llm_planning", return_value="LLM planning called") as mock_llm, \
             patch.object(runner_mod._agent, "handle_query") as mock_rule:
            task = AgentTask(
                session_id="s",
                role=AgentRole.PRODUCT_OWNER,
                masked_input="create epics for the authentication module",
            )
            response = runner_mod.run_product_owner_pipeline(task)
            mock_llm.assert_called_once()
            mock_rule.assert_not_called()
            assert response.status == TaskStatus.COMPLETED
            assert "LLM planning called" in response.draft

    def test_legacy_command_never_calls_llm_path(self, tmp_store):
        """'backlog listele' must reach the rule engine, not POLLMAgent."""
        import src.pipeline.product_owner_runner as runner_mod
        with patch.object(runner_mod, "_run_llm_planning") as mock_llm:
            task = AgentTask(
                session_id="s",
                role=AgentRole.PRODUCT_OWNER,
                masked_input="backlog listele",
            )
            runner_mod.run_product_owner_pipeline(task)
            mock_llm.assert_not_called()

    def test_story_id_command_never_calls_llm_path(self, tmp_store):
        """Explicit story ID reference must stay on rule path."""
        import src.pipeline.product_owner_runner as runner_mod
        with patch.object(runner_mod, "_run_llm_planning") as mock_llm:
            task = AgentTask(
                session_id="s",
                role=AgentRole.PRODUCT_OWNER,
                masked_input="S-001 öncelik high",
            )
            runner_mod.run_product_owner_pipeline(task)
            mock_llm.assert_not_called()

    def test_turkish_planning_prompt_calls_llm_path(self, tmp_store):
        """'epic oluştur' must reach _run_llm_planning."""
        import src.pipeline.product_owner_runner as runner_mod
        with patch.object(runner_mod, "_run_llm_planning", return_value="LLM done") as mock_llm, \
             patch.object(runner_mod._agent, "handle_query") as mock_rule:
            task = AgentTask(
                session_id="s",
                role=AgentRole.PRODUCT_OWNER,
                masked_input="epic oluştur",
            )
            runner_mod.run_product_owner_pipeline(task)
            mock_llm.assert_called_once()
            mock_rule.assert_not_called()


class TestTurkishPlanningIntentDetection:
    """Verify _is_planning_request correctly routes Turkish prompts."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from src.pipeline.product_owner_runner import _is_planning_request
        self._detect = _is_planning_request

    # ── Planning prompts — must resolve True ─────────────────────────────────

    @pytest.mark.parametrize("query", [
        "epic oluştur",
        "epikler oluştur",
        "epik yaz",
        "backlog oluştur",
        "hikaye oluştur",
        "hikayeler oluştur",
        "hikaye yaz",
        "kullanıcı hikayeleri yaz",
        "kullanıcı hikayelerini yaz",
        "kabul kriterleri oluştur",
        "kabul kriterlerini yaz",
        "kriter oluştur",
        "gereksinimler oluştur",
        "gereksinim yaz",
    ])
    def test_turkish_planning_detected(self, query):
        assert self._detect(query) is True, f"Expected True for: {query!r}"

    # ── Command prompts — must resolve False ──────────────────────────────────

    @pytest.mark.parametrize("query", [
        "hikaye oluştur: login ekranı",  # colon → explicit command
        "S-001 öncelik high",
        "backlog listele",
        "backlog status",
        "review completed",
    ])
    def test_turkish_command_not_rerouted(self, query):
        assert self._detect(query) is False, f"Expected False for: {query!r}"
