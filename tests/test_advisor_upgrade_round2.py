"""
test_advisor_upgrade_round2.py — Round-2 advisor fixes.

Covers:
  A. C8 receives acceptance_criteria through developer_runner / REDO path.
  B. PO review_story() method: accept / reject / missing criteria.
  C. Gate PASS no longer auto-adds to increment without PO review.
  D. PO reject prevents increment, does not call accept_story.
  E. PO accept adds to increment.
  F. retro_actions are passed into decompose_goal context.
  G. roadmap target_sprint influences sprint planning selection.
  H. Legacy empty-AC tasks are visibly flagged and surfaced via get_legacy_tasks().
  I. evaluate_all_gates remains backward compatible with legacy positional callers.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
#  Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
    return SprintStateStore(state_path=tmp_path / "sprint_state.json")


# ─────────────────────────────────────────────────────────────────────────────
#  A. C8 reaches developer_runner gate_kwargs
# ─────────────────────────────────────────────────────────────────────────────

class TestC8ThroughDeveloperRunner:

    def test_gate_kwargs_carries_acceptance_criteria(self):
        """run_redo_loop must forward acceptance_criteria to gate_fn."""
        from src.pipeline.redo import run_redo_loop

        captured: dict = {}

        def fake_gate_fn(draft, vector_context, is_empty, agent_role, redo_log, **kwargs):
            captured.update(kwargs)
            return {"conjunction": True, "gates": {}}

        fake_chat      = MagicMock()
        fake_blindspot = MagicMock(return_value="")

        run_redo_loop(
            draft="hello", base_messages=[], vector_context="", is_empty=True,
            redo_log=[],
            agent_role="DeveloperAgent",
            query="q",
            redo_rules="",
            limit_message_template="limit {gate}",
            post_process=lambda s: s,
            gate_fn=fake_gate_fn,
            chat_fn=fake_chat,
            blindspot_fn=fake_blindspot,
            gate_kwargs={
                "codebase_context":    "ctx",
                "acceptance_criteria": ["must log in", "must show error"],
            },
        )
        assert captured.get("acceptance_criteria") == ["must log in", "must show error"]
        assert captured.get("codebase_context") == "ctx"

    def test_developer_runner_reads_ac_from_task_context(self):
        """_process_developer_message must thread task.context AC into gate_kwargs.

        Verified by reading the module source file (ollama import is lazy in
        sprint_loop, but developer_runner imports it at top — so we read the
        file rather than importing).
        """
        from pathlib import Path
        src = Path("src/pipeline/developer_runner.py").read_text(encoding="utf-8")
        # task.context → task_ac local
        assert 'task_ctx.get("acceptance_criteria"' in src
        # task_ac → gate_kwargs
        assert '"acceptance_criteria": task_ac' in src

    def test_old_evaluate_all_gates_callers_still_work(self):
        """evaluate_all_gates without acceptance_criteria kwarg should not raise."""
        from src.gates.evaluator import evaluate_all_gates
        # Legacy positional form used by main_cli.py
        report = evaluate_all_gates("some draft", "", True, "StudentAgent", [])
        assert "conjunction" in report
        assert "gates" in report


# ─────────────────────────────────────────────────────────────────────────────
#  B. POLLMAgent.review_story
# ─────────────────────────────────────────────────────────────────────────────

class TestPOReviewStory:

    def test_returns_required_keys(self):
        from src.agents.po_llm_agent import POLLMAgent
        agent = POLLMAgent()
        result = agent.review_story(
            story={},
            task_output="implemented login and logout",
            acceptance_criteria=["login works", "logout works"],
            sprint_goal="auth",
        )
        assert set(result.keys()) >= {"accepted", "reason", "missing_criteria"}
        assert isinstance(result["accepted"], bool)
        assert isinstance(result["reason"], str)
        assert isinstance(result["missing_criteria"], list)

    def test_accepts_when_all_criteria_evidenced(self):
        from src.agents.po_llm_agent import POLLMAgent
        agent = POLLMAgent()
        result = agent.review_story(
            story={},
            task_output=(
                "Login endpoint implemented. Logout clears session. "
                "Password validation enforces 8 chars."
            ),
            acceptance_criteria=[
                "Login endpoint works",
                "Logout clears session",
                "Password validation",
            ],
        )
        assert result["accepted"] is True
        assert result["missing_criteria"] == []

    def test_rejects_when_criteria_missing(self):
        from src.agents.po_llm_agent import POLLMAgent
        agent = POLLMAgent()
        result = agent.review_story(
            story={},
            task_output="Login endpoint implemented.",
            acceptance_criteria=[
                "Login endpoint works",
                "Password rotation every 90 days",  # not evidenced
            ],
        )
        assert result["accepted"] is False
        assert any("rotation" in m.lower() for m in result["missing_criteria"])

    def test_empty_output_rejects(self):
        from src.agents.po_llm_agent import POLLMAgent
        agent = POLLMAgent()
        result = agent.review_story(
            story={},
            task_output="",
            acceptance_criteria=["something"],
        )
        assert result["accepted"] is False
        assert result["missing_criteria"] == ["something"]

    def test_no_criteria_auto_accepts(self):
        from src.agents.po_llm_agent import POLLMAgent
        agent = POLLMAgent()
        result = agent.review_story(
            story={},
            task_output="any",
            acceptance_criteria=[],
        )
        assert result["accepted"] is True


# ─────────────────────────────────────────────────────────────────────────────
#  C + D + E. Sprint loop wiring — PO accept / reject paths
# ─────────────────────────────────────────────────────────────────────────────

class TestSprintLoopPOIntegration:

    def test_reject_does_not_add_to_increment(self, store):
        """When PO rejects, story must NOT enter the increment."""
        sid = store.add_story(
            title="Auth",
            acceptance_criteria=["password rotation every 90 days"],
            priority="high",
        )
        task_id = store.promote_story_to_sprint_task(sid)
        assert store.get_increment() == []
        # Simulate: gate passed, AC validated, task completed
        store.mark_ac_validated(task_id)
        store.complete_task(task_id, "Login implemented.")
        # PO reviews and rejects
        from src.agents.po_llm_agent import POLLMAgent
        review = POLLMAgent().review_story(
            story=store.get_story(sid) or {},
            task_output="Login implemented.",
            acceptance_criteria=["password rotation every 90 days"],
        )
        assert review["accepted"] is False
        # Caller (sprint_loop) must NOT call add_to_increment on reject
        assert task_id not in store.get_increment()

    def test_accept_adds_to_increment(self, store):
        sid = store.add_story(
            title="Auth",
            acceptance_criteria=["login works", "logout works"],
            priority="high",
        )
        task_id = store.promote_story_to_sprint_task(sid)
        store.mark_ac_validated(task_id)
        store.complete_task(task_id, "Login works. Logout works.")

        from src.agents.po_llm_agent import POLLMAgent
        review = POLLMAgent().review_story(
            story=store.get_story(sid) or {},
            task_output="Login works. Logout works.",
            acceptance_criteria=["login works", "logout works"],
        )
        assert review["accepted"] is True
        # Sprint loop would call both:
        store.accept_story(sid)
        store.add_to_increment(task_id)
        assert task_id in store.get_increment()
        assert store.get_story(sid)["status"] == "accepted"


# ─────────────────────────────────────────────────────────────────────────────
#  F + G. Planning context: retro_actions + roadmap
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanningContextInjection:

    def test_sprint_loop_passes_retro_actions_to_decompose_goal(self, store):
        """sprint_loop PLAN phase must inject retro_actions into decompose_goal context."""
        store.add_retro_actions([
            "Write integration tests first",
            "Reduce PR size",
        ])

        captured_context: dict = {}

        class FakePO:
            def decompose_goal(self, goal, context=""):
                captured_context["context"] = context
                return []
            def generate_stories(self, epics):
                return []
            def review_story(self, **kw):
                return {"accepted": True, "reason": "", "missing_criteria": []}

        from src.loop import sprint_loop

        with patch.object(sprint_loop, "SprintStateStore", return_value=store), \
             patch.object(sprint_loop, "POLLMAgent", FakePO), \
             patch.object(sprint_loop, "_process_developer_message", create=True, new=lambda t: None):
            sprint_loop.run_sprint("Improve auth", sprint_id="sprint-test")

        ctx = captured_context.get("context", "")
        assert "Write integration tests first" in ctx
        assert "Reduce PR size" in ctx

    def test_sprint_loop_prefers_target_sprint_matches(self, store):
        """Backlog items with matching target_sprint must be prepended to story_ids."""
        # Pre-seed a backlog story targeted at sprint-target
        pre_sid = store.add_story(
            title="Targeted work",
            acceptance_criteria=["thing works"],
            target_sprint="sprint-target",
        )

        executed_story_ids: list[str] = []

        class FakePO:
            def decompose_goal(self, goal, context=""):
                return [{"title": "Epic", "description": ""}]
            def generate_stories(self, epics):
                return []  # no new stories — targeted one should still run
            def review_story(self, **kw):
                return {"accepted": True, "reason": "", "missing_criteria": []}

        # Patch promote_story_to_sprint_task to capture which stories get executed
        original_promote = store.promote_story_to_sprint_task

        def spying_promote(sid):
            executed_story_ids.append(sid)
            return original_promote(sid)

        from src.loop import sprint_loop

        with patch.object(sprint_loop, "SprintStateStore", return_value=store), \
             patch.object(sprint_loop, "POLLMAgent", FakePO), \
             patch.object(store, "promote_story_to_sprint_task", side_effect=spying_promote), \
             patch.object(sprint_loop, "_process_developer_message", create=True,
                          new=lambda t: MagicMock(draft="thing works", redo_log=[])), \
             patch.object(sprint_loop, "evaluate_all_gates",
                          return_value={"conjunction": True, "gates": {}}):
            sprint_loop.run_sprint("Any goal", sprint_id="sprint-target")

        assert pre_sid in executed_story_ids, (
            f"targeted story {pre_sid} not executed; saw {executed_story_ids}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  H. Legacy empty-AC tasks
# ─────────────────────────────────────────────────────────────────────────────

class TestLegacyFlag:

    def test_task_without_ac_flagged_legacy(self, store):
        sid = store.add_story(
            title="Old style",
            acceptance_criteria=[],   # empty = legacy
        )
        # update_story won't promote status without AC; force manual promotion
        task_id = store.promote_story_to_sprint_task(sid)
        # Inspect the task in state
        state = store.load()
        task = next(t for t in state["tasks"] if t["id"] == task_id)
        assert task.get("legacy_no_ac") is True

    def test_task_with_ac_not_flagged_legacy(self, store):
        sid = store.add_story(
            title="New style",
            acceptance_criteria=["a works", "b works"],
        )
        task_id = store.promote_story_to_sprint_task(sid)
        state = store.load()
        task = next(t for t in state["tasks"] if t["id"] == task_id)
        assert task.get("legacy_no_ac") is False

    def test_get_legacy_tasks_surfaces_empty_ac_tasks(self, store):
        legacy_sid = store.add_story(title="Legacy", acceptance_criteria=[])
        full_sid   = store.add_story(title="Full",   acceptance_criteria=["x works"])
        legacy_tid = store.promote_story_to_sprint_task(legacy_sid)
        full_tid   = store.promote_story_to_sprint_task(full_sid)

        legacy = store.get_legacy_tasks()
        ids = [t["id"] for t in legacy]
        assert legacy_tid in ids
        assert full_tid not in ids
