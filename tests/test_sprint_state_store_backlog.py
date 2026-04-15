"""
test_sprint_state_store_backlog.py — Tests for backlog methods on SprintStateStore.

Covers: add_story, get_story, get_backlog, update_story, accept/reject,
promote_to_sprint, read_backlog_context_block, story ID generation.
"""

import pytest
from pathlib import Path

from src.pipeline.scrum_team.sprint_state_store import SprintStateStore


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    state_file = tmp_path / "sprint_state.json"
    s = SprintStateStore(state_path=state_file)
    # Will create default state on first load
    return s


# ─────────────────────────────────────────────────────────────────────────────
#  Story ID Generation
# ─────────────────────────────────────────────────────────────────────────────

class TestStoryIdGeneration:

    def test_first_story_gets_s001(self, store):
        sid = store.add_story(title="First story")
        assert sid == "S-001"

    def test_sequential_ids(self, store):
        s1 = store.add_story(title="Story 1")
        s2 = store.add_story(title="Story 2")
        s3 = store.add_story(title="Story 3")
        assert s1 == "S-001"
        assert s2 == "S-002"
        assert s3 == "S-003"


# ─────────────────────────────────────────────────────────────────────────────
#  Add / Get Story
# ─────────────────────────────────────────────────────────────────────────────

class TestAddGetStory:

    def test_add_story_returns_id(self, store):
        sid = store.add_story(title="Test story")
        assert sid.startswith("S-")

    def test_add_story_persists(self, store):
        sid = store.add_story(
            title="Test story",
            description="Description here",
            priority="high",
            acceptance_criteria=["Criterion 1", "Criterion 2"],
            source_request="original query",
        )
        story = store.get_story(sid)
        assert story is not None
        assert story["title"] == "Test story"
        assert story["description"] == "Description here"
        assert story["priority"] == "high"
        assert story["acceptance_criteria"] == ["Criterion 1", "Criterion 2"]
        assert story["source_request"] == "original query"
        assert story["status"] == "draft"

    def test_get_nonexistent_story(self, store):
        assert store.get_story("S-999") is None

    def test_get_backlog_empty(self, store):
        assert store.get_backlog() == []

    def test_get_backlog_with_stories(self, store):
        store.add_story(title="A")
        store.add_story(title="B")
        backlog = store.get_backlog()
        assert len(backlog) == 2

    def test_title_truncated(self, store):
        long_title = "x" * 200
        sid = store.add_story(title=long_title)
        story = store.get_story(sid)
        assert len(story["title"]) == 120

    def test_invalid_priority_defaults_medium(self, store):
        sid = store.add_story(title="Test", priority="invalid")
        story = store.get_story(sid)
        assert story["priority"] == "medium"


# ─────────────────────────────────────────────────────────────────────────────
#  Update Story
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateStory:

    def test_update_priority(self, store):
        sid = store.add_story(title="Test")
        result = store.update_story(sid, priority="high")
        assert result is True
        assert store.get_story(sid)["priority"] == "high"

    def test_update_sets_updated_at(self, store):
        sid = store.add_story(title="Test")
        store.update_story(sid, title="Updated")
        story = store.get_story(sid)
        assert story["updated_at"] is not None

    def test_update_nonexistent_returns_false(self, store):
        assert store.update_story("S-999", title="X") is False

    def test_update_ignores_disallowed_fields(self, store):
        sid = store.add_story(title="Test")
        # 'created_at' is not in the allowed set — should be silently ignored
        store.update_story(sid, created_at="2020-01-01")
        story = store.get_story(sid)
        assert story["created_at"] != "2020-01-01"


# ─────────────────────────────────────────────────────────────────────────────
#  Accept / Reject
# ─────────────────────────────────────────────────────────────────────────────

class TestAcceptReject:

    def test_accept_story(self, store):
        sid = store.add_story(title="Test")
        result = store.accept_story(sid)
        assert result is True
        assert store.get_story(sid)["status"] == "accepted"

    def test_reject_story(self, store):
        sid = store.add_story(title="Test")
        result = store.reject_story(sid, reason="Too broad")
        assert result is True
        story = store.get_story(sid)
        assert story["status"] == "rejected"
        assert story["rejection_reason"] == "Too broad"

    def test_accept_nonexistent(self, store):
        assert store.accept_story("S-999") is False

    def test_reject_nonexistent(self, store):
        assert store.reject_story("S-999") is False


# ─────────────────────────────────────────────────────────────────────────────
#  Promote to Sprint
# ─────────────────────────────────────────────────────────────────────────────

class TestPromoteToSprint:

    def test_promote_creates_task(self, store):
        sid = store.add_story(
            title="API feature",
            description="Build the API",
            priority="high",
            acceptance_criteria=["Returns 200"],
        )
        task_id = store.promote_to_sprint(sid)
        assert task_id == "T-001"

        state = store.load()
        assert len(state["tasks"]) == 1
        task = state["tasks"][0]
        assert task["id"] == "T-001"
        assert task["title"] == "API feature"
        assert task["description"] == "Build the API"
        assert task["priority"] == "high"
        assert task["acceptance_criteria"] == ["Returns 200"]
        assert task["source_story_id"] == sid
        assert task["po_status"] == "pending_review"
        assert task["status"] == "todo"

    def test_promote_updates_story_status(self, store):
        sid = store.add_story(title="Test")
        store.promote_to_sprint(sid)
        story = store.get_story(sid)
        assert story["status"] == "in_sprint"

    def test_promote_nonexistent_returns_none(self, store):
        assert store.promote_to_sprint("S-999") is None

    def test_promote_multiple_stories(self, store):
        s1 = store.add_story(title="Story 1")
        s2 = store.add_story(title="Story 2")
        t1 = store.promote_to_sprint(s1)
        t2 = store.promote_to_sprint(s2)
        assert t1 == "T-001"
        assert t2 == "T-002"

    def test_promote_preserves_existing_tasks(self, store):
        # Pre-populate a task
        state = store.load()
        state["tasks"] = [{"id": "T-001", "title": "Existing", "status": "todo"}]
        store.save(state)

        sid = store.add_story(title="New story")
        task_id = store.promote_to_sprint(sid)
        assert task_id == "T-002"

        state = store.load()
        assert len(state["tasks"]) == 2


# ─────────────────────────────────────────────────────────────────────────────
#  Backlog Context Block
# ─────────────────────────────────────────────────────────────────────────────

class TestBacklogContextBlock:

    def test_empty_backlog(self, store):
        block = store.read_backlog_context_block()
        assert "=== BACKLOG CONTEXT ===" in block
        assert "(none)" in block
        assert "=== END BACKLOG CONTEXT ===" in block

    def test_with_active_stories(self, store):
        store.add_story(title="Active story", priority="high")
        store.add_story(title="Another story")

        block = store.read_backlog_context_block()
        assert "S-001" in block
        assert "S-002" in block
        assert "high" in block

    def test_accepted_stories_excluded(self, store):
        sid = store.add_story(title="Will be accepted")
        store.accept_story(sid)

        block = store.read_backlog_context_block()
        assert sid not in block

    def test_default_state_has_backlog_key(self, tmp_path):
        """A fresh store should include 'backlog' in its default state."""
        store = SprintStateStore(state_path=tmp_path / "fresh.json")
        state = store.load()
        assert "backlog" in state
        assert state["backlog"] == []
