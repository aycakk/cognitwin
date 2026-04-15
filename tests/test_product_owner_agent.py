"""
test_product_owner_agent.py — Unit tests for ProductOwnerAgent.

Covers: intent detection (7 intents), all rule handlers, edge cases,
backlog state mutations, and acceptance/rejection workflows.

No external I/O required — SprintStateStore is pointed at a tmp file.
"""

import json
import pytest
from pathlib import Path

from src.agents.product_owner_agent import ProductOwnerAgent
from src.pipeline.scrum_team.sprint_state_store import SprintStateStore


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_state(backlog=None, tasks=None):
    return {
        "sprint": {
            "id": "sprint-test",
            "goal": "Test sprint",
            "start": "2026-04-07",
            "end": "2026-04-21",
            "velocity": 20,
        },
        "tasks": tasks if tasks is not None else [],
        "backlog": backlog if backlog is not None else [],
        "team": [
            {"id": "developer-default", "role": "Developer", "capacity": 8},
        ],
    }


def _make_story(
    story_id="S-001",
    title="Test hikaye",
    status="draft",
    priority="medium",
    acceptance_criteria=None,
):
    return {
        "story_id":            story_id,
        "title":               title,
        "description":         "",
        "priority":            priority,
        "acceptance_criteria": acceptance_criteria or [],
        "source_request":      "",
        "status":              status,
        "created_at":          "2026-04-10T10:00:00",
        "updated_at":          None,
    }


@pytest.fixture
def tmp_store(tmp_path):
    state_file = tmp_path / "sprint_state.json"
    store = SprintStateStore(state_path=state_file)
    store.save(_make_state())
    return store


@pytest.fixture
def agent(tmp_store):
    return ProductOwnerAgent(state_store=tmp_store)


@pytest.fixture
def store_with_stories(tmp_path):
    state_file = tmp_path / "sprint_state.json"
    store = SprintStateStore(state_path=state_file)
    store.save(_make_state(backlog=[
        _make_story("S-001", "API refactoring", "draft", "high"),
        _make_story("S-002", "Auth module", "ready", "medium",
                    acceptance_criteria=["Login works", "Logout works"]),
        _make_story("S-003", "Accepted story", "accepted"),
    ]))
    return store


@pytest.fixture
def agent_with_stories(store_with_stories):
    return ProductOwnerAgent(state_store=store_with_stories)


# ─────────────────────────────────────────────────────────────────────────────
#  Intent Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestIntentDetection:

    def test_create_story_turkish(self, agent):
        assert agent.detect_intent("hikaye oluştur: API refactoring") == "create_story"

    def test_create_story_english(self, agent):
        assert agent.detect_intent("create story: new feature") == "create_story"

    def test_list_backlog_turkish(self, agent):
        assert agent.detect_intent("backlog listele") == "list_backlog"

    def test_list_backlog_english(self, agent):
        assert agent.detect_intent("show backlog") == "list_backlog"

    def test_prioritize(self, agent):
        assert agent.detect_intent("S-001 öncelik high") == "prioritize"

    def test_define_criteria_turkish(self, agent):
        assert agent.detect_intent("S-001 kabul kriterleri: k1, k2") == "define_criteria"

    def test_define_criteria_english(self, agent):
        assert agent.detect_intent("S-001 acceptance criteria: c1, c2") == "define_criteria"

    def test_accept_turkish(self, agent):
        assert agent.detect_intent("S-001 kabul et") == "accept_story"

    def test_accept_english(self, agent):
        assert agent.detect_intent("accept S-001") == "accept_story"

    def test_reject_turkish(self, agent):
        assert agent.detect_intent("S-001 reddet") == "reject_story"

    def test_reject_english(self, agent):
        assert agent.detect_intent("reject S-001") == "reject_story"

    def test_backlog_status(self, agent):
        assert agent.detect_intent("backlog durumu") == "backlog_status"

    def test_unknown_intent(self, agent):
        assert agent.detect_intent("random gibberish xyz") == "unknown"


# ─────────────────────────────────────────────────────────────────────────────
#  Create Story
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateStory:

    def test_create_story_turkish(self, agent, tmp_store):
        result = agent.handle_query("hikaye oluştur: API refactoring")
        assert "S-001" in result
        assert "API refactoring" in result

        state = tmp_store.load()
        assert len(state["backlog"]) == 1
        assert state["backlog"][0]["story_id"] == "S-001"
        assert state["backlog"][0]["title"] == "API refactoring"
        assert state["backlog"][0]["status"] == "draft"

    def test_create_story_english(self, agent, tmp_store):
        result = agent.handle_query("create story: User login")
        assert "S-001" in result
        assert "User login" in result

    def test_create_multiple_stories(self, agent, tmp_store):
        agent.handle_query("hikaye oluştur: First story")
        agent.handle_query("story ekle: Second story")

        state = tmp_store.load()
        assert len(state["backlog"]) == 2
        assert state["backlog"][0]["story_id"] == "S-001"
        assert state["backlog"][1]["story_id"] == "S-002"

    def test_create_story_stores_source_request(self, agent, tmp_store):
        query = "hikaye oluştur: API refactoring"
        agent.handle_query(query)

        state = tmp_store.load()
        assert state["backlog"][0]["source_request"] == query


# ─────────────────────────────────────────────────────────────────────────────
#  List Backlog
# ─────────────────────────────────────────────────────────────────────────────

class TestListBacklog:

    def test_empty_backlog(self, agent):
        result = agent.handle_query("backlog listele")
        assert "boş" in result.lower()

    def test_list_active_stories(self, agent_with_stories):
        result = agent_with_stories.handle_query("backlog listele")
        assert "S-001" in result
        assert "S-002" in result
        # Accepted stories should not appear in active list
        assert "S-003" not in result

    def test_list_shows_priority(self, agent_with_stories):
        result = agent_with_stories.handle_query("show backlog")
        assert "high" in result


# ─────────────────────────────────────────────────────────────────────────────
#  Prioritize
# ─────────────────────────────────────────────────────────────────────────────

class TestPrioritize:

    def test_set_priority(self, agent_with_stories, store_with_stories):
        result = agent_with_stories.handle_query("S-001 öncelik low")
        assert "low" in result

        story = store_with_stories.get_story("S-001")
        assert story["priority"] == "low"

    def test_turkish_priority_names(self, agent_with_stories, store_with_stories):
        result = agent_with_stories.handle_query("S-001 öncelik yüksek")
        assert "high" in result

        story = store_with_stories.get_story("S-001")
        assert story["priority"] == "high"

    def test_missing_story_id(self, agent):
        result = agent.handle_query("öncelik high")
        assert "ID" in result

    def test_missing_priority_level(self, agent_with_stories):
        result = agent_with_stories.handle_query("S-001 öncelik")
        assert "belirtilmedi" in result or "high/medium/low" in result

    def test_nonexistent_story(self, agent):
        result = agent.handle_query("S-999 öncelik high")
        assert "bulunamadı" in result


# ─────────────────────────────────────────────────────────────────────────────
#  Acceptance Criteria
# ─────────────────────────────────────────────────────────────────────────────

class TestAcceptanceCriteria:

    def test_define_criteria(self, agent_with_stories, store_with_stories):
        result = agent_with_stories.handle_query(
            "S-001 kabul kriterleri: API 200 döner, hata durumunda 400 döner"
        )
        assert "2 kriter" in result

        story = store_with_stories.get_story("S-001")
        assert len(story["acceptance_criteria"]) == 2
        assert "API 200 döner" in story["acceptance_criteria"][0]

    def test_define_criteria_english(self, agent_with_stories, store_with_stories):
        result = agent_with_stories.handle_query(
            "S-001 acceptance criteria: Login succeeds, Error shown on failure"
        )
        assert "2 kriter" in result

    def test_missing_story_id(self, agent):
        result = agent.handle_query("kabul kriterleri: k1, k2")
        assert "ID" in result

    def test_missing_criteria_text(self, agent_with_stories):
        result = agent_with_stories.handle_query("S-001 kabul kriterleri:")
        # Should show usage or error about empty criteria
        assert "S-001" in result


# ─────────────────────────────────────────────────────────────────────────────
#  Accept / Reject
# ─────────────────────────────────────────────────────────────────────────────

class TestAcceptReject:

    def test_accept_story(self, agent_with_stories, store_with_stories):
        result = agent_with_stories.handle_query("S-001 kabul et")
        assert "kabul edildi" in result

        story = store_with_stories.get_story("S-001")
        assert story["status"] == "accepted"

    def test_accept_english(self, agent_with_stories, store_with_stories):
        result = agent_with_stories.handle_query("accept S-002")
        assert "kabul edildi" in result or "accepted" in result

    def test_reject_story(self, agent_with_stories, store_with_stories):
        result = agent_with_stories.handle_query("S-001 reddet")
        assert "reddedildi" in result

        story = store_with_stories.get_story("S-001")
        assert story["status"] == "rejected"

    def test_reject_with_reason(self, agent_with_stories, store_with_stories):
        result = agent_with_stories.handle_query("S-001 reddet: Kapsam çok geniş")
        assert "reddedildi" in result
        assert "Kapsam" in result

        story = store_with_stories.get_story("S-001")
        assert story["rejection_reason"] == "Kapsam çok geniş"

    def test_accept_nonexistent(self, agent):
        result = agent.handle_query("S-999 kabul et")
        assert "bulunamadı" in result

    def test_reject_nonexistent(self, agent):
        result = agent.handle_query("S-999 reddet")
        assert "bulunamadı" in result

    def test_missing_id_accept(self, agent):
        result = agent.handle_query("kabul et")
        assert "ID" in result

    def test_missing_id_reject(self, agent):
        result = agent.handle_query("reddet")
        assert "ID" in result


# ─────────────────────────────────────────────────────────────────────────────
#  Backlog Status
# ─────────────────────────────────────────────────────────────────────────────

class TestBacklogStatus:

    def test_empty_backlog_status(self, agent):
        result = agent.handle_query("backlog durumu")
        assert "boş" in result.lower()

    def test_status_with_stories(self, agent_with_stories):
        result = agent_with_stories.handle_query("backlog durumu")
        assert "3" in result  # total count
        assert "draft" in result
        assert "ready" in result
        assert "accepted" in result


# ─────────────────────────────────────────────────────────────────────────────
#  Unknown Intent
# ─────────────────────────────────────────────────────────────────────────────

class TestUnknownIntent:

    def test_unknown_shows_help(self, agent):
        result = agent.handle_query("random gibberish")
        assert "Product Owner" in result
        assert "hikaye oluştur" in result
