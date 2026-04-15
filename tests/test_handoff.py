"""
test_handoff.py — Integration tests for agent-to-agent state handoff.

Covers:
  - ScrumMaster write → Developer read via shared SprintStateStore
  - State consistency after mutations (add task, update status, set goal)
  - Concurrent access safety (thread-level)
  - Context block formatting for LLM injection
"""

import json
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules.setdefault("ollama", MagicMock())
sys.modules.setdefault("chromadb", MagicMock())

from src.agents.product_owner_agent import ProductOwnerAgent
from src.agents.scrum_master_agent import ScrumMasterAgent
from src.pipeline.scrum_team.sprint_state_store import SprintStateStore


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _base_state():
    return {
        "sprint": {
            "id": "sprint-handoff",
            "goal": "Test handoff",
            "start": "2026-04-07",
            "end": "2026-04-21",
            "velocity": 20,
        },
        "tasks": [
            {
                "id": "T-001",
                "title": "Existing task",
                "type": "story",
                "status": "in_progress",
                "assignee": "developer-default",
                "priority": "high",
                "story_points": 5,
                "blocker": None,
                "created_at": "2026-04-09T00:00:00",
                "assigned_at": "2026-04-09T10:00:00",
            },
        ],
        "team": [
            {"id": "developer-default", "role": "Developer", "capacity": 8},
        ],
    }


@pytest.fixture
def shared_store(tmp_path):
    state_file = tmp_path / "sprint_state.json"
    store = SprintStateStore(state_path=state_file)
    store.save(_base_state())
    return store


@pytest.fixture
def scrum_agent(shared_store):
    return ScrumMasterAgent(state_store=shared_store)


# ─────────────────────────────────────────────────────────────────────────────
#  ScrumMaster Write → Developer Read
# ─────────────────────────────────────────────────────────────────────────────

class TestScrumMasterToDeveloperHandoff:

    def test_added_task_visible_after_assignment(self, scrum_agent, shared_store):
        """SM adds a task and assigns it → Developer's context block should include it."""
        scrum_agent.handle_query("görev ekle: API refactoring")
        scrum_agent.handle_query("T-002 developer-default üzerine ata")

        context = shared_store.read_context_block()
        # Assigned task appears in active assignments
        assert "T-002" in context
        # T-001 is still visible (already assigned in base state)
        assert "T-001" in context

    def test_updated_status_reflected_in_context(self, scrum_agent, shared_store):
        """SM marks task done → Developer sees updated status."""
        scrum_agent.handle_query("T-001 durumunu done olarak güncelle")

        # Done tasks with assignee don't appear in assignments (get_assignments filters status!=done)
        assignments = shared_store.get_assignments()
        assert len(assignments) == 0

    def test_blocked_task_appears_in_developer_context(self, scrum_agent, shared_store):
        """SM blocks a task → Developer sees it in blocked section."""
        # First update task to blocked
        state = shared_store.load()
        state["tasks"][0]["status"] = "blocked"
        state["tasks"][0]["blocker"] = "CI pipeline broken"
        shared_store.save(state)

        context = shared_store.read_context_block()
        assert "Blocked Tasks:" in context
        assert "CI pipeline broken" in context

    def test_sprint_goal_change_visible(self, scrum_agent, shared_store):
        """SM sets new goal → Developer reads updated goal."""
        scrum_agent.handle_query("sprint hedef: Deploy v2.0")

        goal = shared_store.get_sprint_goal()
        assert goal == "Deploy v2.0"

        context = shared_store.read_context_block()
        assert "Deploy v2.0" in context

    def test_assignment_visible_in_developer_context(self, scrum_agent, shared_store):
        """SM assigns task → Developer sees assignment in context."""
        # Add unassigned task first
        scrum_agent.handle_query("görev ekle: New feature")
        scrum_agent.handle_query("T-002 developer-default üzerine ata")

        context = shared_store.read_context_block()
        assert "T-002" in context
        assert "developer-default" in context


# ─────────────────────────────────────────────────────────────────────────────
#  Context Block Format
# ─────────────────────────────────────────────────────────────────────────────

class TestContextBlockFormat:

    def test_context_block_has_boundaries(self, shared_store):
        context = shared_store.read_context_block()
        assert context.startswith("=== SPRINT CONTEXT ===")
        assert context.endswith("=== END SPRINT CONTEXT ===")

    def test_context_block_contains_goal(self, shared_store):
        context = shared_store.read_context_block()
        assert "Sprint Goal:" in context
        assert "Test handoff" in context

    def test_context_block_shows_assignments(self, shared_store):
        context = shared_store.read_context_block()
        assert "Active Assignments:" in context
        assert "T-001" in context
        assert "developer-default" in context

    def test_context_block_empty_blocked(self, shared_store):
        context = shared_store.read_context_block()
        assert "Blocked Tasks:" in context
        assert "(none)" in context  # no blocked tasks in base state

    def test_context_block_on_empty_state(self, tmp_path):
        store = SprintStateStore(state_path=tmp_path / "empty.json")
        # Will create default state
        context = store.read_context_block()
        assert "=== SPRINT CONTEXT ===" in context
        assert "(none)" in context  # no tasks in default state


# ─────────────────────────────────────────────────────────────────────────────
#  State Consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestStateConsistency:

    def test_multiple_mutations_are_consistent(self, scrum_agent, shared_store):
        """Multiple SM operations produce consistent final state."""
        scrum_agent.handle_query("görev ekle: Task A")
        scrum_agent.handle_query("görev ekle: Task B")
        scrum_agent.handle_query("T-002 developer-default üzerine ata")
        scrum_agent.handle_query("T-001 durumunu done olarak güncelle")

        state = shared_store.load()
        assert len(state["tasks"]) == 3  # T-001 + T-002 + T-003

        t001 = next(t for t in state["tasks"] if t["id"] == "T-001")
        assert t001["status"] == "done"

        t002 = next(t for t in state["tasks"] if t["id"] == "T-002")
        assert t002["assignee"] == "developer-default"

    def test_state_persists_across_agent_instances(self, shared_store):
        """State written by one agent instance is readable by another."""
        agent1 = ScrumMasterAgent(state_store=shared_store)
        agent1.handle_query("görev ekle: Persisted task")

        agent2 = ScrumMasterAgent(state_store=shared_store)
        state = shared_store.load()
        # Verify the new task exists in state (it won't show in status text by title)
        task_ids = [t["id"] for t in state["tasks"]]
        assert "T-002" in task_ids
        new_task = next(t for t in state["tasks"] if t["id"] == "T-002")
        assert "Persisted task" in new_task["title"]


# ─────────────────────────────────────────────────────────────────────────────
#  Concurrent Access (Thread Safety)
# ─────────────────────────────────────────────────────────────────────────────

class TestConcurrentAccess:

    def test_concurrent_reads_dont_corrupt(self, shared_store):
        """Multiple threads reading simultaneously should not cause errors."""
        errors = []

        def reader():
            try:
                for _ in range(10):
                    shared_store.read_context_block()
                    shared_store.get_sprint_goal()
                    shared_store.get_assignments()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Concurrent reads failed: {errors}"

    def test_concurrent_write_and_read(self, shared_store):
        """Writer and readers operating concurrently should not corrupt state."""
        errors = []
        agent = ScrumMasterAgent(state_store=shared_store)

        def writer():
            try:
                for i in range(5):
                    agent.handle_query(f"görev ekle: Concurrent task {i}")
            except Exception as e:
                errors.append(("writer", e))

        def reader():
            try:
                for _ in range(10):
                    shared_store.read_context_block()
            except Exception as e:
                errors.append(("reader", e))

        writer_thread = threading.Thread(target=writer)
        reader_threads = [threading.Thread(target=reader) for _ in range(3)]

        writer_thread.start()
        for t in reader_threads:
            t.start()

        writer_thread.join(timeout=10)
        for t in reader_threads:
            t.join(timeout=10)

        assert errors == [], f"Concurrent access failed: {errors}"

        # Verify state is valid JSON
        state = shared_store.load()
        assert isinstance(state, dict)
        assert "tasks" in state


# ─────────────────────────────────────────────────────────────────────────────
#  ProductOwner → ScrumMaster → Developer  (Scrum team handoff)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def scrum_store(tmp_path):
    """Shared store with backlog support for Scrum team handoff tests."""
    state_file = tmp_path / "sprint_state.json"
    store = SprintStateStore(state_path=state_file)
    store.save({
        "sprint": {
            "id": "sprint-handoff-po",
            "goal": "PO handoff test",
            "start": "2026-04-07",
            "end": "2026-04-21",
            "velocity": 20,
        },
        "tasks": [],
        "backlog": [],
        "team": [
            {"id": "developer-default", "role": "Developer", "capacity": 8},
        ],
    })
    return store


@pytest.fixture
def po_agent(scrum_store):
    return ProductOwnerAgent(state_store=scrum_store)


@pytest.fixture
def sm_agent(scrum_store):
    return ScrumMasterAgent(state_store=scrum_store)


class TestProductOwnerToScrumMasterHandoff:
    """PO creates stories in backlog → SM promotes to sprint tasks → Dev reads context."""

    def test_po_story_visible_in_backlog(self, po_agent, scrum_store):
        """PO creates a story → it's in the backlog."""
        po_agent.handle_query("hikaye oluştur: User authentication")

        backlog = scrum_store.get_backlog()
        assert len(backlog) == 1
        assert backlog[0]["story_id"] == "S-001"
        assert backlog[0]["title"] == "User authentication"

    def test_po_story_not_in_sprint_context(self, po_agent, scrum_store):
        """PO stories should NOT appear in sprint context (only in backlog)."""
        po_agent.handle_query("hikaye oluştur: User authentication")

        sprint_context = scrum_store.read_context_block()
        assert "User authentication" not in sprint_context
        assert "S-001" not in sprint_context

    def test_sm_promotes_story_to_sprint(self, po_agent, sm_agent, scrum_store):
        """PO creates story → SM promotes it → task appears in sprint."""
        po_agent.handle_query("hikaye oluştur: User authentication")
        result = sm_agent.handle_query("S-001 sprint'e ekle")

        assert "T-001" in result
        assert "S-001" in result

        state = scrum_store.load()
        assert len(state["tasks"]) == 1
        task = state["tasks"][0]
        assert task["id"] == "T-001"
        assert task["title"] == "User authentication"
        assert task["source_story_id"] == "S-001"
        assert task["po_status"] == "pending_review"

    def test_promoted_story_status_updated(self, po_agent, sm_agent, scrum_store):
        """After promotion, the backlog story status becomes 'in_sprint'."""
        po_agent.handle_query("hikaye oluştur: User authentication")
        sm_agent.handle_query("S-001 sprint'e ekle")

        story = scrum_store.get_story("S-001")
        assert story["status"] == "in_sprint"

    def test_promoted_task_in_developer_context(self, po_agent, sm_agent, scrum_store):
        """After SM promotes and assigns → Developer sees the task in context block."""
        po_agent.handle_query("hikaye oluştur: User authentication")
        sm_agent.handle_query("S-001 sprint'e ekle")
        sm_agent.handle_query("T-001 developer-default üzerine ata")

        context = scrum_store.read_context_block()
        assert "T-001" in context
        assert "developer-default" in context

    def test_acceptance_criteria_propagated(self, po_agent, sm_agent, scrum_store):
        """PO defines criteria → SM promotes → task inherits criteria."""
        po_agent.handle_query("hikaye oluştur: User authentication")
        po_agent.handle_query("S-001 kabul kriterleri: Login works, Logout works")
        sm_agent.handle_query("S-001 sprint'e ekle")

        state = scrum_store.load()
        task = state["tasks"][0]
        assert task["acceptance_criteria"] == ["Login works", "Logout works"]

    def test_priority_propagated(self, po_agent, sm_agent, scrum_store):
        """PO sets priority → SM promotes → task inherits priority."""
        po_agent.handle_query("hikaye oluştur: User authentication")
        po_agent.handle_query("S-001 öncelik high")
        sm_agent.handle_query("S-001 sprint'e ekle")

        state = scrum_store.load()
        task = state["tasks"][0]
        assert task["priority"] == "high"

    def test_full_scrum_flow(self, po_agent, sm_agent, scrum_store):
        """End-to-end: PO creates → PO refines → SM promotes → SM assigns → SM completes → PO accepts."""
        # PO creates story
        po_agent.handle_query("hikaye oluştur: API endpoint")
        po_agent.handle_query("S-001 öncelik high")
        po_agent.handle_query("S-001 kabul kriterleri: Returns 200, Validates input")

        # SM promotes and assigns
        sm_agent.handle_query("S-001 sprint'e ekle")
        sm_agent.handle_query("T-001 developer-default üzerine ata")

        # Verify developer sees the task
        context = scrum_store.read_context_block()
        assert "T-001" in context

        # SM marks done
        sm_agent.handle_query("T-001 durumunu done olarak güncelle")

        # PO accepts the story
        po_agent.handle_query("S-001 kabul et")

        story = scrum_store.get_story("S-001")
        assert story["status"] == "accepted"

    def test_backlog_isolation_from_sprint_context(self, po_agent, scrum_store):
        """Backlog context and sprint context are separate blocks."""
        po_agent.handle_query("hikaye oluştur: Story A")
        po_agent.handle_query("hikaye oluştur: Story B")

        backlog_ctx = scrum_store.read_backlog_context_block()
        sprint_ctx = scrum_store.read_context_block()

        # Backlog context shows stories
        assert "S-001" in backlog_ctx
        assert "S-002" in backlog_ctx
        assert "=== BACKLOG CONTEXT ===" in backlog_ctx

        # Sprint context does NOT show stories
        assert "S-001" not in sprint_ctx
        assert "S-002" not in sprint_ctx
        assert "=== SPRINT CONTEXT ===" in sprint_ctx
