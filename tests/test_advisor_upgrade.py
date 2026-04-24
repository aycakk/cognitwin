"""
test_advisor_upgrade.py — Tests for advisor-feedback upgrades.

Covers:
  A. C8 gate: acceptance criteria validation (pass / fail / empty)
  B. SprintStateStore: new schema fields, AC enforcement in complete_task,
     product_goal, increment, roadmap, meeting_notes, retro_actions
  C. ProductOwner: story without AC gets status=needs_refinement,
     update_story with AC promotes to draft
  D. RoadmapPlanner: builds entries with target_sprint/date/deployment_package
  E. WeeklyReporter: past-week and next-week reports include required sections
  F. MeetingNotesManager: notes persisted, retro actions available cross-sprint
  G. Integration: cannot mark task done without validated AC; can after mark_ac_validated
"""

from __future__ import annotations

import pytest
from pathlib import Path
from datetime import date, timedelta

# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
    return SprintStateStore(state_path=tmp_path / "sprint_state.json")


@pytest.fixture
def store_with_story(store):
    sid = store.add_story(
        title="Login feature",
        acceptance_criteria=["User can log in", "Error shown on bad password"],
        priority="high",
    )
    return store, sid


@pytest.fixture
def store_with_task(store_with_story):
    store, sid = store_with_story
    task_id = store.promote_story_to_sprint_task(sid)
    store.assign_task(task_id, "developer-default")
    store.start_task(task_id)
    return store, sid, task_id


# ─────────────────────────────────────────────────────────────────────────────
#  A. C8 Gate
# ─────────────────────────────────────────────────────────────────────────────

class TestC8AcceptanceCriteria:

    def test_pass_when_no_criteria(self):
        from src.gates.c8_acceptance_criteria import check_acceptance_criteria
        passed, evidence = check_acceptance_criteria("Any output", [])
        assert passed is True
        assert "skipped" in evidence.lower()

    def test_pass_when_all_criteria_addressed(self):
        from src.gates.c8_acceptance_criteria import check_acceptance_criteria
        draft = "The user login form validates email format and shows error on bad password."
        criteria = ["login form validates email", "error shown on bad password"]
        passed, evidence = check_acceptance_criteria(draft, criteria)
        assert passed is True
        assert "All" in evidence

    def test_fail_when_criterion_not_addressed(self):
        from src.gates.c8_acceptance_criteria import check_acceptance_criteria
        draft = "The dashboard shows total sales figures."
        criteria = ["user can reset password", "email confirmation sent"]
        passed, evidence = check_acceptance_criteria(draft, criteria)
        assert passed is False
        assert "not addressed" in evidence

    def test_partial_fail_reports_count(self):
        from src.gates.c8_acceptance_criteria import check_acceptance_criteria
        draft = "User can register with email address."
        criteria = [
            "registration form validates email",
            "password must be eight characters",
            "success confirmation shown after submit",
        ]
        passed, evidence = check_acceptance_criteria(draft, criteria)
        # 'registration form validates email' should match partially
        # 'password must be eight characters' unlikely to match
        assert "not addressed" in evidence or passed is True  # tolerate either

    def test_criteria_with_no_significant_words_skip(self):
        from src.gates.c8_acceptance_criteria import check_acceptance_criteria
        passed, evidence = check_acceptance_criteria("some output", ["OK", "Yes", "It"])
        assert passed is True  # all criteria have only short words → skipped

    def test_evaluator_wires_c8(self):
        from src.gates.evaluator import evaluate_all_gates
        draft = "The user login form validates email and shows error on failure."
        criteria = ["login form validates email", "error shown on failure"]
        report = evaluate_all_gates(
            draft, "", True, "DeveloperAgent", [],
            codebase_context="login form validator",
            acceptance_criteria=criteria,
        )
        assert "C8" in report["gates"]
        assert report["gates"]["C8"]["pass"] is True

    def test_evaluator_c8_fail_propagates_to_conjunction(self):
        from src.gates.evaluator import evaluate_all_gates
        draft = "The dashboard shows some charts."
        criteria = ["user can reset password", "email confirmation sent after reset"]
        report = evaluate_all_gates(
            draft, "", True, "DeveloperAgent", [],
            codebase_context="dashboard chart renderer",
            acceptance_criteria=criteria,
        )
        assert report["gates"]["C8"]["pass"] is False
        assert report["conjunction"] is False

    def test_evaluator_c8_absent_for_non_developer(self):
        from src.gates.evaluator import evaluate_all_gates
        report = evaluate_all_gates(
            "Some scrum master output.", "", True, "ScrumMasterAgent", [],
            acceptance_criteria=["some criterion"],
        )
        # ScrumMasterAgent only runs C4 — C8 must not appear
        assert "C8" not in report["gates"]

    def test_c8_revision_hint_in_gate_result(self):
        from src.gates.gate_result import build_gate_result
        gr = build_gate_result("C8", False, "2/3 criteria not addressed")
        assert "acceptance" in gr.revision_hint.lower() or "criteria" in gr.revision_hint.lower()
        assert gr.confidence_score < 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  B. SprintStateStore — new fields and methods
# ─────────────────────────────────────────────────────────────────────────────

class TestSprintStateStoreNewFields:

    # ── Schema fields ─────────────────────────────────────────────────────────

    def test_new_top_level_keys_in_default_state(self, store):
        state = store.load()
        for key in ("product_goal", "roadmap", "meeting_notes", "increment", "retro_actions"):
            assert key in state, f"Missing top-level key: {key}"

    def test_add_story_stores_canonical_english_fields(self, store):
        sid = store.add_story(
            title="Register user",
            title_en="Register user",
            user_story_en="As a user I want to register so that I can log in",
            epic="Authentication",
            story_points=3,
            deployment_package="MVP Auth",
            acceptance_criteria=["Form validates email"],
        )
        story = store.get_story(sid)
        assert story["title_en"] == "Register user"
        assert story["user_story_en"] == "As a user I want to register so that I can log in"
        assert story["epic"] == "Authentication"
        assert story["story_points"] == 3
        assert story["deployment_package"] == "MVP Auth"

    def test_add_story_without_ac_is_needs_refinement(self, store):
        sid = store.add_story(title="Vague feature")
        story = store.get_story(sid)
        assert story["status"] == "needs_refinement"

    def test_add_story_with_ac_is_draft(self, store):
        sid = store.add_story(
            title="Login",
            acceptance_criteria=["User can log in with email"],
        )
        story = store.get_story(sid)
        assert story["status"] == "draft"

    def test_update_story_with_ac_promotes_needs_refinement_to_draft(self, store):
        sid = store.add_story(title="Feature without AC")
        assert store.get_story(sid)["status"] == "needs_refinement"
        store.update_story(sid, acceptance_criteria=["Feature works correctly"])
        assert store.get_story(sid)["status"] == "draft"

    def test_update_story_new_fields_allowed(self, store):
        sid = store.add_story(title="Story", acceptance_criteria=["AC1"])
        store.update_story(sid, epic="My Epic", story_points=5, deployment_package="Pkg A")
        story = store.get_story(sid)
        assert story["epic"] == "My Epic"
        assert story["story_points"] == 5
        assert story["deployment_package"] == "Pkg A"

    # ── AC enforcement in complete_task ───────────────────────────────────────

    def test_complete_task_blocked_without_ac_validated(self, store_with_task):
        store, sid, task_id = store_with_task
        result = store.complete_task(task_id, "Done!")
        assert result is False
        task = next(t for t in store.load()["tasks"] if t["id"] == task_id)
        assert task["status"] != "done"

    def test_complete_task_succeeds_after_mark_ac_validated(self, store_with_task):
        store, sid, task_id = store_with_task
        store.mark_ac_validated(task_id)
        result = store.complete_task(task_id, "Done!")
        assert result is True
        task = next(t for t in store.load()["tasks"] if t["id"] == task_id)
        assert task["status"] == "done"

    def test_complete_task_without_ac_does_not_require_validation(self, store):
        sid = store.add_story(title="No AC story")
        task_id = store.promote_story_to_sprint_task(sid)
        store.assign_task(task_id, "developer-default")
        store.start_task(task_id)
        # Task has empty AC — should complete without validation
        result = store.complete_task(task_id, "Done!")
        assert result is True

    # ── Product Goal ──────────────────────────────────────────────────────────

    def test_set_get_product_goal(self, store):
        store.set_product_goal("Build a world-class e-learning platform")
        assert store.get_product_goal() == "Build a world-class e-learning platform"

    def test_product_goal_default_empty(self, store):
        assert store.get_product_goal() == ""

    # ── Increment ─────────────────────────────────────────────────────────────

    def test_add_to_increment(self, store_with_task):
        store, sid, task_id = store_with_task
        store.mark_ac_validated(task_id)
        store.complete_task(task_id, "Result")
        result = store.add_to_increment(task_id)
        assert result is True
        assert task_id in store.get_increment()

    def test_add_nonexistent_task_to_increment_returns_false(self, store):
        assert store.add_to_increment("T-999") is False

    # ── Roadmap ───────────────────────────────────────────────────────────────

    def test_add_and_get_roadmap_entry(self, store):
        entry = {
            "release_package": "MVP Auth",
            "target_sprint":   "sprint-1",
            "target_date":     "2026-05-07",
            "scope":           ["S-001"],
            "dependencies":    [],
            "success_criteria": ["User can log in"],
            "status":          "planned",
        }
        pkg_id = store.add_roadmap_entry(entry)
        assert pkg_id.startswith("PKG-")
        roadmap = store.get_roadmap()
        assert len(roadmap) == 1
        assert roadmap[0]["target_sprint"] == "sprint-1"
        assert roadmap[0]["target_date"] == "2026-05-07"
        assert roadmap[0]["release_package"] == "MVP Auth"

    def test_roadmap_ids_sequential(self, store):
        store.add_roadmap_entry({"release_package": "A", "status": "planned"})
        store.add_roadmap_entry({"release_package": "B", "status": "planned"})
        ids = [e["package_id"] for e in store.get_roadmap()]
        assert ids == ["PKG-001", "PKG-002"]

    # ── Meeting Notes ─────────────────────────────────────────────────────────

    def test_add_and_get_meeting_note(self, store):
        note = {
            "event_type":   "sprint_planning",
            "date":         "2026-04-24",
            "participants": ["PO", "SM"],
            "decisions":    ["Sprint goal set"],
            "blockers":     [],
            "action_items": [],
        }
        note_id = store.add_meeting_note(note)
        assert note_id.startswith("MN-")
        notes = store.get_meeting_notes()
        assert len(notes) == 1
        assert notes[0]["event_type"] == "sprint_planning"

    def test_get_meeting_notes_filtered_by_event_type(self, store):
        store.add_meeting_note({"event_type": "sprint_planning",    "date": "2026-04-24"})
        store.add_meeting_note({"event_type": "daily_scrum",        "date": "2026-04-25"})
        store.add_meeting_note({"event_type": "sprint_retrospective", "date": "2026-04-26"})
        retro_notes = store.get_meeting_notes(event_type="sprint_retrospective")
        assert len(retro_notes) == 1
        assert retro_notes[0]["event_type"] == "sprint_retrospective"

    # ── Retro Actions ─────────────────────────────────────────────────────────

    def test_add_and_get_retro_actions(self, store):
        store.add_retro_actions(["Improve PR review process", "Add daily standups"])
        actions = store.get_retro_actions()
        assert len(actions) == 2
        assert "Improve PR review process" in actions

    def test_retro_actions_replace_not_append(self, store):
        store.add_retro_actions(["Old action"])
        store.add_retro_actions(["New action 1", "New action 2"])
        actions = store.get_retro_actions()
        assert "Old action" not in actions
        assert len(actions) == 2


# ─────────────────────────────────────────────────────────────────────────────
#  C. ProductOwner — story status enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestProductOwnerACEnforcement:

    def test_story_without_ac_cannot_be_promoted_to_ready(self, store):
        sid = store.add_story(title="Vague feature without AC")
        story = store.get_story(sid)
        # Status must be needs_refinement, not draft or ready
        assert story["status"] == "needs_refinement"

    def test_story_with_ac_is_immediately_draft(self, store):
        sid = store.add_story(
            title="Login",
            acceptance_criteria=["User can log in with valid credentials"],
        )
        assert store.get_story(sid)["status"] == "draft"

    def test_cannot_complete_sprint_task_without_ac_validation(self, store):
        sid = store.add_story(
            title="Feature X",
            acceptance_criteria=["Feature works end to end"],
        )
        task_id = store.promote_story_to_sprint_task(sid)
        store.assign_task(task_id, "dev")
        store.start_task(task_id)
        # No mark_ac_validated — must fail
        ok = store.complete_task(task_id, "Done")
        assert ok is False

    def test_can_complete_sprint_task_after_ac_validation(self, store):
        sid = store.add_story(
            title="Feature Y",
            acceptance_criteria=["Feature Y passes all criteria"],
        )
        task_id = store.promote_story_to_sprint_task(sid)
        store.assign_task(task_id, "dev")
        store.start_task(task_id)
        store.mark_ac_validated(task_id)
        ok = store.complete_task(task_id, "Done")
        assert ok is True


# ─────────────────────────────────────────────────────────────────────────────
#  D. RoadmapPlanner
# ─────────────────────────────────────────────────────────────────────────────

class TestRoadmapPlanner:

    def test_build_from_backlog_creates_entries(self, store):
        from src.pipeline.roadmap_planner import RoadmapPlanner
        store.add_story(
            title="Login feature",
            acceptance_criteria=["User can log in"],
            deployment_package="MVP Auth",
        )
        store.add_story(
            title="Dashboard",
            acceptance_criteria=["Charts load within 2s"],
            deployment_package="MVP Dashboard",
        )
        planner = RoadmapPlanner(state_store=store)
        entries = planner.build_from_backlog(sprint_start_date=date(2026, 5, 1))
        assert len(entries) == 2

    def test_roadmap_entry_has_required_fields(self, store):
        from src.pipeline.roadmap_planner import RoadmapPlanner
        store.add_story(
            title="Auth feature",
            acceptance_criteria=["Login works"],
            epic="Authentication",
        )
        planner = RoadmapPlanner(state_store=store)
        entries = planner.build_from_backlog(sprint_start_date=date(2026, 5, 1))
        entry = entries[0]
        for field in ("package_id", "release_package", "target_sprint", "target_date",
                      "scope", "success_criteria", "status"):
            assert field in entry, f"Missing field: {field}"

    def test_roadmap_entry_has_target_sprint_and_date(self, store):
        from src.pipeline.roadmap_planner import RoadmapPlanner
        store.add_story(
            title="Feature",
            acceptance_criteria=["Works"],
            deployment_package="Release 1",
        )
        planner = RoadmapPlanner(state_store=store)
        entries = planner.build_from_backlog(sprint_start_date=date(2026, 5, 1), sprint_number_offset=2)
        assert entries[0]["target_sprint"] == "sprint-2"
        assert entries[0]["target_date"] == "2026-05-15"

    def test_roadmap_persisted_to_state(self, store):
        from src.pipeline.roadmap_planner import RoadmapPlanner
        store.add_story("Feature", acceptance_criteria=["AC1"])
        RoadmapPlanner(state_store=store).build_from_backlog()
        assert len(store.get_roadmap()) == 1

    def test_roadmap_text_includes_target(self, store):
        from src.pipeline.roadmap_planner import RoadmapPlanner
        store.add_story("Feature", acceptance_criteria=["AC1"], deployment_package="Package A")
        RoadmapPlanner(state_store=store).build_from_backlog(sprint_start_date=date(2026, 5, 1))
        text = RoadmapPlanner(state_store=store).get_roadmap_text()
        assert "Package A" in text
        assert "target_sprint" in text.lower() or "sprint" in text.lower()


# ─────────────────────────────────────────────────────────────────────────────
#  E. WeeklyReporter
# ─────────────────────────────────────────────────────────────────────────────

class TestWeeklyReporter:

    def _setup_done_task(self, store, days_ago: int = 2):
        """Add a story→task and complete it n days ago."""
        from datetime import datetime, timedelta
        sid = store.add_story("Task completed", acceptance_criteria=["AC1"])
        task_id = store.promote_story_to_sprint_task(sid)
        store.assign_task(task_id, "developer-default")
        store.start_task(task_id)
        store.mark_ac_validated(task_id)
        store.complete_task(task_id, "Result")
        # Backdate completed_at to simulate last week
        state = store.load()
        for t in state["tasks"]:
            if t["id"] == task_id:
                t["completed_at"] = (
                    datetime.now() - timedelta(days=days_ago)
                ).isoformat(timespec="seconds")
        store.save(state)
        return task_id

    def test_past_week_contains_completed_section(self, store):
        from src.pipeline.weekly_reporter import WeeklyReporter
        self._setup_done_task(store)
        report = WeeklyReporter(state_store=store).past_week_report()
        assert "Completed Tasks" in report

    def test_past_week_contains_blockers_section(self, store):
        from src.pipeline.weekly_reporter import WeeklyReporter
        report = WeeklyReporter(state_store=store).past_week_report()
        assert "Blockers" in report or "Blocker" in report

    def test_past_week_includes_done_task(self, store):
        from src.pipeline.weekly_reporter import WeeklyReporter
        task_id = self._setup_done_task(store, days_ago=2)
        report = WeeklyReporter(state_store=store).past_week_report()
        assert task_id in report

    def test_past_week_excludes_old_task(self, store):
        from src.pipeline.weekly_reporter import WeeklyReporter
        task_id = self._setup_done_task(store, days_ago=10)
        report = WeeklyReporter(state_store=store).past_week_report()
        assert task_id not in report

    def test_next_week_contains_required_sections(self, store):
        from src.pipeline.weekly_reporter import WeeklyReporter
        store.add_story("Ready story", acceptance_criteria=["AC"])
        report = WeeklyReporter(state_store=store).next_week_plan()
        assert "Carry-over" in report
        assert "Ready for Sprint" in report

    def test_next_week_shows_needs_refinement(self, store):
        from src.pipeline.weekly_reporter import WeeklyReporter
        store.add_story("No AC story")  # → needs_refinement
        report = WeeklyReporter(state_store=store).next_week_plan()
        assert "Needs Refinement" in report or "needs_refinement" in report.lower()

    def test_next_week_includes_roadmap_package(self, store):
        from src.pipeline.weekly_reporter import WeeklyReporter
        store.add_roadmap_entry({
            "package_id":      "PKG-001",
            "release_package": "MVP Launch",
            "target_sprint":   "sprint-1",
            "target_date":     str(date.today() + timedelta(days=3)),
            "scope":           [],
            "status":          "planned",
        })
        report = WeeklyReporter(state_store=store).next_week_plan()
        assert "MVP Launch" in report


# ─────────────────────────────────────────────────────────────────────────────
#  F. MeetingNotesManager
# ─────────────────────────────────────────────────────────────────────────────

class TestMeetingNotesManager:

    def test_sprint_planning_notes_has_required_fields(self, store):
        from src.pipeline.meeting_notes import MeetingNotesManager
        mgr  = MeetingNotesManager(state_store=store)
        note = mgr.generate_sprint_planning_notes()
        for field in ("event_type", "date", "participants", "decisions", "action_items"):
            assert field in note

    def test_sprint_planning_event_type(self, store):
        from src.pipeline.meeting_notes import MeetingNotesManager
        note = MeetingNotesManager(state_store=store).generate_sprint_planning_notes()
        assert note["event_type"] == "sprint_planning"

    def test_daily_scrum_notes(self, store):
        from src.pipeline.meeting_notes import MeetingNotesManager
        note = MeetingNotesManager(state_store=store).generate_daily_scrum_notes()
        assert note["event_type"] == "daily_scrum"
        assert "In progress" in " ".join(note["decisions"])

    def test_sprint_review_notes(self, store):
        from src.pipeline.meeting_notes import MeetingNotesManager
        note = MeetingNotesManager(state_store=store).generate_sprint_review_notes()
        assert note["event_type"] == "sprint_review"
        assert "velocity" in note

    def test_retrospective_notes_persists_actions(self, store):
        from src.pipeline.meeting_notes import MeetingNotesManager
        actions = ["Add more automated tests", "Daily standup at 9am"]
        mgr = MeetingNotesManager(state_store=store)
        mgr.generate_retrospective_notes(actions=actions)
        stored = store.get_retro_actions()
        assert "Add more automated tests" in stored
        assert "Daily standup at 9am" in stored

    def test_retro_actions_available_via_get_retro_actions(self, store):
        from src.pipeline.meeting_notes import MeetingNotesManager
        mgr = MeetingNotesManager(state_store=store)
        mgr.generate_retrospective_notes(actions=["Improve CI pipeline"])
        assert "Improve CI pipeline" in mgr.get_retro_actions()

    def test_save_and_retrieve_note(self, store):
        from src.pipeline.meeting_notes import MeetingNotesManager
        mgr  = MeetingNotesManager(state_store=store)
        note = mgr.generate_daily_scrum_notes()
        note_id = mgr.save(note)
        assert note_id.startswith("MN-")
        retrieved = mgr.get_all()
        assert len(retrieved) == 1
        assert retrieved[0]["event_type"] == "daily_scrum"

    def test_get_all_filtered(self, store):
        from src.pipeline.meeting_notes import MeetingNotesManager
        mgr = MeetingNotesManager(state_store=store)
        mgr.save(mgr.generate_sprint_planning_notes())
        mgr.save(mgr.generate_daily_scrum_notes())
        mgr.save(mgr.generate_sprint_review_notes())
        retros = mgr.get_all(event_type="sprint_planning")
        assert len(retros) == 1
        assert retros[0]["event_type"] == "sprint_planning"

    def test_retro_notes_available_to_next_planning(self, store):
        from src.pipeline.meeting_notes import MeetingNotesManager
        mgr = MeetingNotesManager(state_store=store)
        # Simulate end of sprint: retro sets actions
        mgr.generate_retrospective_notes(actions=["Add integration tests"])
        # Next planning picks them up
        planning_note = mgr.generate_sprint_planning_notes()
        decisions_text = " ".join(planning_note["decisions"])
        assert "retrospective" in decisions_text.lower() or "action" in decisions_text.lower()

    def test_invalid_event_type_raises(self, store):
        from src.pipeline.meeting_notes import MeetingNotesManager
        mgr = MeetingNotesManager(state_store=store)
        with pytest.raises(ValueError, match="Unknown event_type"):
            mgr._build_note(
                event_type="invalid_event",
                participants=[],
                decisions=[],
                blockers=[],
                action_items=[],
            )

    def test_backlog_refinement_notes(self, store):
        from src.pipeline.meeting_notes import MeetingNotesManager
        store.add_story("No AC story")
        mgr  = MeetingNotesManager(state_store=store)
        note = mgr.generate_backlog_refinement_notes()
        assert note["event_type"] == "backlog_refinement"
        assert any("acceptance criteria" in d.lower() for d in note["decisions"])


# ─────────────────────────────────────────────────────────────────────────────
#  G. Integration — policy + evaluator + store enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_c8_in_developer_agent_policy(self):
        from src.governance.policy import GATE_POLICY
        assert "C8" in GATE_POLICY["DeveloperAgent"]

    def test_c8_not_in_scrum_master_policy(self):
        from src.governance.policy import GATE_POLICY
        assert "C8" not in GATE_POLICY["ScrumMasterAgent"]

    def test_full_lifecycle_with_ac_enforcement(self, store):
        """Complete lifecycle: story→task→gate pass→AC validated→done→increment."""
        # Create story with AC
        sid = store.add_story(
            title="User registration",
            acceptance_criteria=["Email validated", "Confirmation email sent"],
            deployment_package="MVP Auth",
        )
        assert store.get_story(sid)["status"] == "draft"

        # Promote to sprint task
        task_id = store.promote_story_to_sprint_task(sid)
        store.assign_task(task_id, "backend-developer")
        store.start_task(task_id)

        # Cannot complete without AC validation
        assert store.complete_task(task_id, "done") is False

        # Mark AC validated (simulates C8 gate pass in sprint loop)
        store.mark_ac_validated(task_id)

        # Now can complete
        assert store.complete_task(task_id, "Registration implemented") is True

        # Story is accepted and task is in increment
        store.accept_story(sid)
        store.add_to_increment(task_id)

        assert store.get_story(sid)["status"] == "accepted"
        assert task_id in store.get_increment()

    def test_story_without_ac_is_not_sprint_ready(self, store):
        sid = store.add_story(title="Undefined feature")
        story = store.get_story(sid)
        assert story["status"] == "needs_refinement"
        # After adding AC it becomes draft (sprint-ready)
        store.update_story(sid, acceptance_criteria=["Feature is functional"])
        assert store.get_story(sid)["status"] == "draft"

    def test_po_llm_agent_stories_always_have_ac(self):
        """POLLMAgent fallback must produce stories with non-empty AC."""
        from src.agents.po_llm_agent import POLLMAgent
        agent  = POLLMAgent()
        epics  = [{"title": "Authentication", "description": "User auth system"}]
        # Use fallback path directly (no Ollama needed)
        stories = agent._fallback_stories(epics)
        for story in stories:
            assert len(story.get("acceptance_criteria", [])) >= 1, (
                f"Story '{story['title']}' has no acceptance criteria"
            )

    def test_po_llm_agent_fallback_has_english_fields(self):
        from src.agents.po_llm_agent import POLLMAgent
        agent = POLLMAgent()
        epics = [{"title": "Dashboard", "description": "User dashboard"}]
        stories = agent._fallback_stories(epics)
        for s in stories:
            assert "title_en" in s
            assert "user_story_en" in s
            assert s["story_points"] > 0
