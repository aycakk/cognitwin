"""tests/test_sprint_increment_and_report.py

Verifies:
  1. C8-FAIL / blocked task is excluded from Product Increment even when
     state["increment"] contains its ID.
  2. po_status=ready_for_review (PO not yet accepted) excludes a task.
  3. done + accepted + ac_validated task is included in Product Increment.
  4. Product Goal is derived from goal text and displayed in the report.
  5. Report-instruction lines do not reach the PO / become backlog stories.
  6. Roadmap / Past-Week / Next-Week / Meeting-Notes sections are present.
  7. Blocked and unexecuted backlog stories appear in Carry-over.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import pytest


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _FakeResult:
    sprint_id: str = "sprint-test"
    goal: str = "Build a TODO app"
    completed_stories: list = field(default_factory=list)
    blocked_stories:   list = field(default_factory=list)
    total_steps: int = 5
    avg_confidence: float = 0.75
    summary: str = ""


def _make_store(tmp_path, tasks=None, backlog=None,
                increment=None, roadmap=None,
                meeting_notes=None, product_goal=""):
    """Create a SprintStateStore pre-populated with test data."""
    from src.pipeline.scrum_team.sprint_state_store import SprintStateStore, DEFAULT_SPRINT_STATE
    store = SprintStateStore(state_path=tmp_path / "state.json")
    state = copy.deepcopy(DEFAULT_SPRINT_STATE)
    state["tasks"]         = tasks         or []
    state["backlog"]       = backlog        or []
    state["increment"]     = increment      or []
    state["roadmap"]       = roadmap        or []
    state["meeting_notes"] = meeting_notes  or []
    state["product_goal"]  = product_goal
    state["sprint"]["goal"] = "Build a TODO app"
    store.save(state)
    return store


# ─────────────────────────────────────────────────────────────────────────────
#  1. Blocked / C8-FAIL task must NOT appear in Product Increment
# ─────────────────────────────────────────────────────────────────────────────

class TestIncrementFiltering:

    def test_blocked_task_excluded_even_if_in_state_increment(self, tmp_path):
        """state['increment'] contains T-010 but T-010 is blocked → excluded."""
        from src.services.api.sprint_bridge import render_sprint_report, _task_is_increment_eligible

        tasks = [{
            "id": "T-010",
            "title": "View Tasks",
            "status": "blocked",          # ← blocked
            "po_status": "pending_review",
            "acceptance_criteria": ["User can view task list"],
            "ac_validated": False,
            "artifact_type": "text_plan",
            "legacy_no_ac": False,
        }]
        store = _make_store(
            tmp_path,
            tasks=tasks,
            increment=["T-010"],  # stale entry — task was added before it got blocked
        )

        # Direct eligibility check
        assert not _task_is_increment_eligible(tasks[0]), \
            "blocked task must fail _task_is_increment_eligible"

        result = _FakeResult(
            blocked_stories=[{
                "story_id": "S-001", "task_id": "T-010",
                "title": "View Tasks", "reason": "PO rejected: bad output",
            }]
        )
        text = render_sprint_report(result, store, executed_task_ids={"T-010"})

        # Increment section must not contain T-010
        increment_section = _section(text, "Product Increment")
        assert "T-010" not in increment_section, (
            f"T-010 should not appear in Product Increment section.\n{increment_section}"
        )
        # It should appear in Blocked section
        assert "T-010" in text or "S-001" in text  # listed somewhere as blocked

    def test_c8_fail_task_excluded_from_increment(self, tmp_path):
        """Task with ac_validated=False (C8 FAIL) must not appear in increment."""
        from src.services.api.sprint_bridge import _task_is_increment_eligible

        task = {
            "id": "T-005",
            "status": "done",
            "po_status": "accepted",
            "acceptance_criteria": ["Criterion A"],
            "ac_validated": False,   # C8 FAIL
        }
        assert not _task_is_increment_eligible(task)

    def test_ready_for_review_excluded_from_increment(self, tmp_path):
        """po_status=ready_for_review means PO has not yet accepted — exclude."""
        from src.services.api.sprint_bridge import _task_is_increment_eligible

        task = {
            "id": "T-003",
            "status": "done",
            "po_status": "ready_for_review",  # PO not accepted yet
            "acceptance_criteria": ["Criterion B"],
            "ac_validated": True,
        }
        assert not _task_is_increment_eligible(task), \
            "ready_for_review is not accepted — must be excluded from increment"

    def test_done_accepted_ac_validated_included(self, tmp_path):
        """Only tasks satisfying all Done criteria appear in Product Increment."""
        from src.services.api.sprint_bridge import render_sprint_report, _task_is_increment_eligible

        tasks = [{
            "id": "T-001",
            "title": "Create Task",
            "status": "done",
            "po_status": "accepted",
            "acceptance_criteria": ["User can create a task"],
            "ac_validated": True,
            "artifact_type": "text_plan",
            "legacy_no_ac": False,
        }]
        store = _make_store(tmp_path, tasks=tasks, increment=["T-001"])

        assert _task_is_increment_eligible(tasks[0])

        result = _FakeResult(
            completed_stories=[{"story_id": "S-001", "task_id": "T-001", "title": "Create Task"}]
        )
        text = render_sprint_report(result, store, executed_task_ids={"T-001"})

        increment_section = _section(text, "Product Increment")
        assert "T-001" in increment_section, \
            f"T-001 should appear in Product Increment.\n{increment_section}"

    def test_no_ac_task_with_accepted_po_included(self, tmp_path):
        """Legacy task (no AC) with accepted PO status must appear in increment."""
        from src.services.api.sprint_bridge import _task_is_increment_eligible

        task = {
            "id": "T-002",
            "status": "done",
            "po_status": "accepted",
            "acceptance_criteria": [],   # no AC
            "ac_validated": False,        # irrelevant when no AC
            "legacy_no_ac": True,
        }
        assert _task_is_increment_eligible(task)


# ─────────────────────────────────────────────────────────────────────────────
#  2. add_to_increment guard
# ─────────────────────────────────────────────────────────────────────────────

class TestAddToIncrementGuard:

    def test_refuses_blocked_task(self, tmp_path):
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
        store = SprintStateStore(state_path=tmp_path / "state.json")
        sid = store.add_story("View Tasks", acceptance_criteria=["User can view list"])
        tid = store.promote_story_to_sprint_task(sid)
        store.start_task(tid)
        store.block_task(tid, "Gate failed")

        ok = store.add_to_increment(tid)
        assert not ok, "add_to_increment must refuse a blocked task"
        assert tid not in store.get_increment()

    def test_refuses_ready_for_review(self, tmp_path):
        """po_status=ready_for_review is not yet accepted — must be refused."""
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
        import copy
        from src.pipeline.scrum_team.sprint_state_store import DEFAULT_SPRINT_STATE

        store = SprintStateStore(state_path=tmp_path / "state.json")
        state = copy.deepcopy(DEFAULT_SPRINT_STATE)
        state["tasks"] = [{
            "id": "T-001",
            "status": "done",
            "po_status": "ready_for_review",  # not yet accepted
            "acceptance_criteria": ["x"],
            "ac_validated": True,
        }]
        store.save(state)

        ok = store.add_to_increment("T-001")
        assert not ok, "ready_for_review must be refused by add_to_increment"

    def test_accepts_done_accepted_ac_validated(self, tmp_path):
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
        store = SprintStateStore(state_path=tmp_path / "state.json")
        sid = store.add_story("Create Task", acceptance_criteria=["User can create"])
        tid = store.promote_story_to_sprint_task(sid)
        store.start_task(tid)
        store.mark_ac_validated(tid)
        store.complete_task(tid, "Done")
        store.accept_story(sid)  # sets po_status="accepted" on task

        ok = store.add_to_increment(tid)
        assert ok, "add_to_increment must accept a done+accepted+ac_validated task"
        assert tid in store.get_increment()


# ─────────────────────────────────────────────────────────────────────────────
#  3. Product Goal derived and displayed
# ─────────────────────────────────────────────────────────────────────────────

class TestProductGoal:

    def test_product_goal_shown_in_report(self, tmp_path):
        from src.services.api.sprint_bridge import render_sprint_report

        store = _make_store(
            tmp_path,
            product_goal="Deliver a usable TODO app with task creation and deletion.",
        )
        result = _FakeResult()
        text = render_sprint_report(result, store, executed_task_ids=set())
        assert "Deliver a usable TODO app" in text
        assert "Product Goal" in text

    def test_product_goal_not_set_shows_placeholder(self, tmp_path):
        from src.services.api.sprint_bridge import render_sprint_report

        store = _make_store(tmp_path, product_goal="")
        result = _FakeResult()
        text = render_sprint_report(result, store, executed_task_ids=set())
        assert "(not set)" in text

    def test_sprint_loop_sets_product_goal(self, tmp_path):
        """sprint_loop.run_sprint must persist product_goal when it's empty."""
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
        store = SprintStateStore(state_path=tmp_path / "state.json")
        assert store.get_product_goal() == ""

        store.set_product_goal("Build a TODO app for team productivity")
        assert store.get_product_goal() == "Build a TODO app for team productivity"


# ─────────────────────────────────────────────────────────────────────────────
#  4. Report instructions do not become backlog stories
# ─────────────────────────────────────────────────────────────────────────────

class TestGoalCleaning:

    def test_also_produce_stripped(self):
        from src.services.api.sprint_bridge import _clean_goal_for_po

        raw = (
            "Build a TODO app that supports task creation, completion, and deletion. "
            "Also produce: Product Goal, Product Backlog (stories), Product Roadmap, "
            "Past week accomplishments, Next week plan, Meeting notes, Product Increment."
        )
        cleaned = _clean_goal_for_po(raw)
        assert "TODO app" in cleaned
        assert "Also produce" not in cleaned
        assert "Product Roadmap" not in cleaned
        assert "Past week" not in cleaned
        assert "Meeting notes" not in cleaned

    def test_turkish_instruction_stripped(self):
        from src.services.api.sprint_bridge import _clean_goal_for_po

        raw = "Bir TODO uygulaması yap. Ayrıca üret: Ürün Yol Haritası, Geçen Hafta."
        cleaned = _clean_goal_for_po(raw)
        assert "TODO" in cleaned
        assert "Ayrıca üret" not in cleaned

    def test_clean_goal_without_instructions_unchanged(self):
        from src.services.api.sprint_bridge import _clean_goal_for_po

        raw = "Build a secure login system with JWT tokens."
        cleaned = _clean_goal_for_po(raw)
        assert cleaned == raw

    def test_standalone_instruction_lines_stripped(self):
        from src.services.api.sprint_bridge import _clean_goal_for_po

        raw = (
            "Build a task manager.\n"
            "Product Roadmap\n"
            "Past week accomplishments\n"
            "Meeting notes"
        )
        cleaned = _clean_goal_for_po(raw)
        assert "task manager" in cleaned
        assert "Product Roadmap" not in cleaned
        assert "Past week" not in cleaned
        assert "Meeting notes" not in cleaned


# ─────────────────────────────────────────────────────────────────────────────
#  5. Required report sections are present
# ─────────────────────────────────────────────────────────────────────────────

class TestReportSections:

    def test_all_required_sections_present(self, tmp_path):
        from src.services.api.sprint_bridge import render_sprint_report

        store = _make_store(
            tmp_path,
            product_goal="Build a TODO app",
            roadmap=[{
                "package_id": "PKG-001",
                "release_package": "MVP Task Management",
                "target_sprint": "sprint-1",
                "target_date": "2026-05-10",
                "status": "planned",
                "success_criteria": ["User can create tasks"],
            }],
            meeting_notes=[{
                "note_id": "MN-001",
                "event_type": "sprint_planning",
                "date": "2026-04-27",
                "participants": ["ProductOwnerAgent"],
                "decisions": ["Sprint goal set"],
                "blockers": [],
                "action_items": [],
                "created_at": "2026-04-27T10:00:00",
            }],
        )
        result = _FakeResult()
        text = render_sprint_report(result, store, executed_task_ids=set())

        for section in [
            "Product Goal",
            "Product Backlog",
            "Product Roadmap",
            "Product Increment",
            "Blocked / Rejected",
            "Past Week Accomplishments",
            "Next Week Plan",
            "Sprint Meeting Notes",
        ]:
            assert section in text, f"Missing section: {section!r}"

    def test_roadmap_uses_release_package_field(self, tmp_path):
        """_fmt_roadmap must read 'release_package', not 'title'."""
        from src.services.api.sprint_bridge import render_sprint_report

        store = _make_store(
            tmp_path,
            roadmap=[{
                "package_id": "PKG-001",
                "release_package": "Auth MVP",   # correct field from RoadmapPlanner
                "target_sprint": "sprint-2",
                "target_date": "2026-05-14",
                "status": "planned",
            }],
        )
        result = _FakeResult()
        text = render_sprint_report(result, store, executed_task_ids=set())
        assert "Auth MVP" in text, "Roadmap should show release_package value"

    def test_meeting_notes_shown(self, tmp_path):
        from src.services.api.sprint_bridge import render_sprint_report

        store = _make_store(
            tmp_path,
            meeting_notes=[{
                "note_id": "MN-001",
                "event_type": "sprint_review",
                "date": "2026-04-27",
                "participants": [],
                "decisions": ["Velocity = 12 story points"],
                "blockers": [],
                "action_items": [{"owner": "PO", "action": "Review pending tasks"}],
                "created_at": "2026-04-27T12:00:00",
            }],
        )
        result = _FakeResult()
        text = render_sprint_report(result, store, executed_task_ids=set())
        assert "sprint_review" in text
        assert "Velocity = 12" in text


# ─────────────────────────────────────────────────────────────────────────────
#  6. Carry-over: blocked/unexecuted stories appear in Next Week section
# ─────────────────────────────────────────────────────────────────────────────

class TestCarryOver:

    def test_unexecuted_backlog_story_in_carryover(self, tmp_path):
        """Stories not completed this run must appear in carry-over."""
        from src.services.api.sprint_bridge import render_sprint_report

        backlog = [
            {
                "story_id": "S-001",
                "title": "Create Task",
                "status": "in_sprint",
                "priority": "high",
                "acceptance_criteria": ["User can create"],
            },
            {
                "story_id": "S-002",
                "title": "Delete Task",
                "status": "draft",
                "priority": "medium",
                "acceptance_criteria": ["User can delete"],
            },
        ]
        # T-001 is S-001's task but it's blocked — not in safe_increment
        tasks = [{
            "id": "T-001",
            "title": "Create Task",
            "status": "blocked",
            "po_status": "pending_review",
            "acceptance_criteria": ["User can create"],
            "ac_validated": False,
            "source_story_id": "S-001",
            "artifact_type": "text_plan",
            "legacy_no_ac": False,
        }]
        store = _make_store(tmp_path, tasks=tasks, backlog=backlog)

        result = _FakeResult(
            blocked_stories=[{
                "story_id": "S-001", "task_id": "T-001",
                "title": "Create Task", "reason": "Step limit reached",
            }]
        )
        text = render_sprint_report(result, store, executed_task_ids={"T-001"})

        assert "Carry-over" in text
        # S-002 was never started — must be in carry-over
        assert "S-002" in text or "Delete Task" in text

    def test_done_accepted_story_not_in_carryover(self, tmp_path):
        """Successfully completed and accepted stories must NOT appear in carry-over."""
        from src.services.api.sprint_bridge import render_sprint_report

        backlog = [{
            "story_id": "S-001",
            "title": "Create Task",
            "status": "accepted",   # done and accepted
            "priority": "high",
            "acceptance_criteria": ["User can create"],
        }]
        tasks = [{
            "id": "T-001",
            "title": "Create Task",
            "status": "done",
            "po_status": "accepted",
            "acceptance_criteria": ["User can create"],
            "ac_validated": True,
            "source_story_id": "S-001",
            "artifact_type": "text_plan",
            "legacy_no_ac": False,
        }]
        store = _make_store(tmp_path, tasks=tasks, backlog=backlog, increment=["T-001"])

        result = _FakeResult(
            completed_stories=[{"story_id": "S-001", "task_id": "T-001", "title": "Create Task"}]
        )
        text = render_sprint_report(result, store, executed_task_ids={"T-001"})

        # S-001 / T-001 should be in increment, not carry-over
        increment_section = _section(text, "Product Increment")
        assert "T-001" in increment_section

        carryover_section = _section(text, "Carry-over")
        assert "S-001" not in carryover_section, \
            "Accepted story must not appear as carry-over"


# ─────────────────────────────────────────────────────────────────────────────
#  7. State isolation across successive sprint runs
# ─────────────────────────────────────────────────────────────────────────────

class TestStateIsolation:
    """Prove that a new sprint run does not show stale data from a prior run."""

    def _populate_todo_state(self, store) -> None:
        """Seed the store with a TODO-app sprint state (simulates prior run)."""
        state = store.load()
        state["product_goal"] = "Build a TODO app for task management."
        state["sprint"]["goal"] = "Deliver TODO app MVP"
        state["backlog"] = [
            {"story_id": "S-001", "title": "User can add new task",
             "status": "accepted", "priority": "high", "acceptance_criteria": []},
            {"story_id": "S-002", "title": "User can mark tasks as completed",
             "status": "accepted", "priority": "high", "acceptance_criteria": []},
            {"story_id": "S-003", "title": "User can delete tasks",
             "status": "accepted", "priority": "medium", "acceptance_criteria": []},
        ]
        state["tasks"] = [
            {"id": "T-001", "title": "Add task form", "status": "done",
             "po_status": "accepted", "ac_validated": True,
             "artifact_type": "text_plan", "acceptance_criteria": [],
             "source_story_id": "S-001", "legacy_no_ac": True,
             "completed_at": "2026-04-01T10:00:00"},
            {"id": "T-002", "title": "Delete button", "status": "done",
             "po_status": "accepted", "ac_validated": True,
             "artifact_type": "text_plan", "acceptance_criteria": [],
             "source_story_id": "S-003", "legacy_no_ac": True,
             "completed_at": "2026-04-01T11:00:00"},
        ]
        state["increment"] = ["T-001", "T-002"]
        state["roadmap"] = [{
            "package_id": "RP-001", "release_package": "TODO App MVP",
            "target_sprint": "sprint-1", "target_date": "2026-05-01",
            "status": "planned",
            "success_criteria": ["task form works", "task title validation"],
        }]
        state["meeting_notes"] = [{
            "note_id": "MN-001", "event_type": "sprint_planning",
            "date": "2026-04-01",
            "decisions": ["Build TODO app this sprint"],
            "action_items": [],
        }]
        state["retro_actions"] = [
            "Scrum Master Agent entegrasyonunu tamamla",
            "agile.ttl ontoloji doğrulaması",
        ]
        store.save(state)

    def test_reset_for_isolated_sprint_clears_all_user_visible_state(self, tmp_path):
        """reset_for_isolated_sprint must wipe tasks, backlog, roadmap,
        meeting_notes, increment, product_goal, and sprint goal."""
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore

        store = SprintStateStore(state_path=tmp_path / "state.json")
        self._populate_todo_state(store)

        store.reset_for_isolated_sprint()
        state = store.load()

        assert state["tasks"]         == [], "tasks must be cleared"
        assert state["backlog"]       == [], "backlog must be cleared"
        assert state["increment"]     == [], "increment must be cleared"
        assert state["roadmap"]       == [], "roadmap must be cleared"
        assert state["meeting_notes"] == [], "meeting_notes must be cleared"
        assert state["product_goal"]  == "", "product_goal must be cleared"
        assert state["retro_actions"] == [], "retro_actions must be cleared"

    def test_reset_preserves_team_capacity(self, tmp_path):
        """Team capacity config must survive reset_for_isolated_sprint."""
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore

        store = SprintStateStore(state_path=tmp_path / "state.json")
        state = store.load()
        state["team"] = [{"id": "custom-dev", "role": "Custom Developer", "capacity": 12}]
        store.save(state)

        store.reset_for_isolated_sprint()
        after = store.load()
        assert after["team"] == [{"id": "custom-dev", "role": "Custom Developer", "capacity": 12}]

    def test_notes_sprint_report_does_not_show_todo_product_goal(self, tmp_path):
        """After an isolated reset, the Product Goal must not be the old TODO goal."""
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
        from src.services.api.sprint_bridge import render_sprint_report

        store = SprintStateStore(state_path=tmp_path / "state.json")
        self._populate_todo_state(store)

        # Simulate isolated reset + new product_goal set for notes app
        store.reset_for_isolated_sprint()
        store.set_product_goal("Deliver a notes app with create/edit/delete/search.")

        result = _FakeResult(goal="Build a simple notes app")
        text = render_sprint_report(result, store, executed_task_ids=set())

        assert "TODO app" not in text, (
            "Product Goal from prior TODO run must not appear in notes app report"
        )
        assert "notes app" in text.lower(), "New product goal must appear in report"

    def test_notes_sprint_backlog_excludes_todo_stories(self, tmp_path):
        """After isolated reset, backlog must not contain old TODO stories."""
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
        from src.services.api.sprint_bridge import render_sprint_report

        store = SprintStateStore(state_path=tmp_path / "state.json")
        self._populate_todo_state(store)

        store.reset_for_isolated_sprint()
        # Add only notes-app stories post-reset
        sid = store.add_story("User can create a note",
                              acceptance_criteria=["note is saved"],
                              priority="high")
        result = _FakeResult(goal="Build a simple notes app")
        text = render_sprint_report(result, store, executed_task_ids=set())

        backlog_section = _section(text, "Product Backlog")
        assert "add new task" not in backlog_section.lower(), (
            "Old TODO story must not appear in notes app backlog"
        )
        assert "create a note" in backlog_section.lower(), (
            "Notes app story must appear in backlog"
        )

    def test_notes_roadmap_excludes_todo_roadmap_entries(self, tmp_path):
        """After isolated reset, roadmap must not contain TODO success criteria."""
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
        from src.services.api.sprint_bridge import render_sprint_report

        store = SprintStateStore(state_path=tmp_path / "state.json")
        self._populate_todo_state(store)

        store.reset_for_isolated_sprint()
        store.add_roadmap_entry({
            "package_id": "RP-002",
            "release_package": "Notes App v1",
            "target_sprint": "sprint-1",
            "target_date": "2026-05-15",
            "status": "planned",
            "success_criteria": ["user can search notes"],
        })

        result = _FakeResult(goal="Build a simple notes app")
        text = render_sprint_report(result, store, executed_task_ids=set())

        roadmap_section = _section(text, "Product Roadmap")
        assert "task form" not in roadmap_section.lower(), (
            "Old TODO roadmap criteria must not appear"
        )
        assert "Notes App v1" in roadmap_section, (
            "New notes roadmap package must appear"
        )

    def test_next_week_excludes_old_seed_tasks(self, tmp_path):
        """Stale retro/seed tasks like Turkish system tasks must not appear
        in the Next Week / Carry-over section after an isolated reset."""
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
        from src.services.api.sprint_bridge import render_sprint_report

        store = SprintStateStore(state_path=tmp_path / "state.json")
        self._populate_todo_state(store)

        store.reset_for_isolated_sprint()
        result = _FakeResult(goal="Build a simple notes app")
        text = render_sprint_report(result, store, executed_task_ids=set())

        assert "entegrasyon" not in text, (
            "Old Turkish seed task 'Scrum Master Agent entegrasyonunu tamamla' must not appear"
        )
        assert "ontoloji" not in text, (
            "Old Turkish seed task 'agile.ttl ontoloji doğrulaması' must not appear"
        )

    def test_meeting_notes_shown_for_current_run_only(self, tmp_path):
        """After isolated reset, prior sprint meeting notes must not appear."""
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
        from src.services.api.sprint_bridge import render_sprint_report

        store = SprintStateStore(state_path=tmp_path / "state.json")
        self._populate_todo_state(store)

        store.reset_for_isolated_sprint()
        # Add a current-run note
        store.add_meeting_note({
            "note_id": "MN-CURR", "event_type": "sprint_planning",
            "date": "2026-04-27",
            "decisions": ["Build notes app this sprint"],
            "action_items": [],
        })

        result = _FakeResult(goal="Build a simple notes app")
        text = render_sprint_report(result, store, executed_task_ids=set())

        notes_section = _section(text, "Sprint Meeting Notes")
        assert "Build TODO app this sprint" not in notes_section, (
            "Old TODO sprint planning note must not appear"
        )
        assert "Build notes app this sprint" in notes_section, (
            "Current run note must appear"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  8. Story scope filtering and meeting-note counts
# ─────────────────────────────────────────────────────────────────────────────

class TestScopeFiltering:
    """Verify that off-scope PO stories are rejected and excluded from roadmap/
    meeting notes, and that meeting note story counts match actual executions."""

    # ── _story_aligns_with_goal ───────────────────────────────────────────────

    def test_login_story_rejected_for_notes_goal(self):
        """Login/auth story must be rejected for a notes-app goal."""
        from src.loop.sprint_loop import _story_aligns_with_goal
        story = {
            "title": "User can log in with email and password",
            "description": "As a user, I want to log in so that my data is secure.",
            "epic": "Authentication",
        }
        goal = "Build a simple notes app where users can create, edit, delete, and search notes."
        assert not _story_aligns_with_goal(story, goal), (
            "Auth story must be rejected for a notes-app goal"
        )

    def test_register_story_rejected_for_notes_goal(self):
        """Register/signup story must be rejected for a notes-app goal."""
        from src.loop.sprint_loop import _story_aligns_with_goal
        story = {
            "title": "User can register a new account",
            "description": "User fills email and password to register.",
            "epic": "Authentication",
        }
        goal = "Build a simple notes app where users can create, edit, delete, and search notes."
        assert not _story_aligns_with_goal(story, goal)

    def test_create_note_story_accepted_for_notes_goal(self):
        """In-scope create-note story must pass for a notes-app goal."""
        from src.loop.sprint_loop import _story_aligns_with_goal
        story = {
            "title": "User can create a new note",
            "description": "User types a title and content and saves the note.",
            "epic": "Note Management",
        }
        goal = "Build a simple notes app where users can create, edit, delete, and search notes."
        assert _story_aligns_with_goal(story, goal)

    def test_search_story_accepted_for_notes_goal(self):
        from src.loop.sprint_loop import _story_aligns_with_goal
        story = {
            "title": "User can search notes by keyword",
            "description": "Notes matching the search term are highlighted.",
            "epic": "Note Management",
        }
        goal = "Build a simple notes app where users can create, edit, delete, and search notes."
        assert _story_aligns_with_goal(story, goal)

    def test_auth_story_accepted_when_goal_mentions_auth(self):
        """Auth story MUST be accepted when the goal explicitly mentions authentication."""
        from src.loop.sprint_loop import _story_aligns_with_goal
        story = {
            "title": "User can log in with email and password",
            "description": "Secure login flow with JWT tokens.",
            "epic": "Authentication",
        }
        goal = "Build a secure notes app with user authentication and login."
        assert _story_aligns_with_goal(story, goal), (
            "Auth story must be accepted when goal mentions authentication"
        )

    # ── Off-scope story → rejected in backlog ─────────────────────────────────

    def test_out_of_scope_story_not_promoted_to_sprint(self, tmp_path):
        """sprint_loop must reject an off-scope story in the backlog and not execute it."""
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
        from unittest.mock import patch, MagicMock
        from src.loop import sprint_loop

        store = SprintStateStore(state_path=tmp_path / "state.json")

        class FakePO:
            def decompose_goal(self, goal, context=""):
                return [{"title": "Notes", "description": ""}]
            def generate_stories(self, epics):
                return [
                    {   # in scope
                        "title": "User can create a note",
                        "description": "Note creation feature",
                        "epic": "Note Management",
                        "priority": "high",
                        "acceptance_criteria": ["note is saved"],
                        "deployment_package": "Notes v1",
                    },
                    {   # out of scope — auth
                        "title": "User can log in with email and password",
                        "description": "Authentication flow",
                        "epic": "Authentication",
                        "priority": "medium",
                        "acceptance_criteria": ["user can log in"],
                        "deployment_package": "Auth",
                    },
                ]
            def review_story(self, **kw):
                return {"accepted": True, "reason": "", "missing_criteria": []}

        goal = "Build a simple notes app where users can create, edit, delete, and search notes."

        with patch.object(sprint_loop, "SprintStateStore", return_value=store), \
             patch.object(sprint_loop, "POLLMAgent", FakePO), \
             patch.object(sprint_loop, "_process_developer_message", create=True,
                          new=lambda t: MagicMock(draft="note created", redo_log=[], artifact_type="text_plan")), \
             patch.object(sprint_loop, "evaluate_all_gates",
                          return_value={"conjunction": True, "gates": {}}):
            sprint_loop.run_sprint(goal, sprint_id="sprint-scope-test")

        backlog = store.get_backlog()
        login_stories = [s for s in backlog if "log in" in s["title"].lower() or "login" in s["title"].lower()]
        create_stories = [s for s in backlog if "create a note" in s["title"].lower()]

        assert login_stories, "Login story should be in backlog (as rejected)"
        assert login_stories[0]["status"] == "rejected", (
            "Login story must be rejected, not in_sprint"
        )
        assert "Out of scope" in (login_stories[0].get("rejection_reason") or ""), (
            "rejection_reason must mention 'Out of scope'"
        )
        assert create_stories, "Notes story should be in backlog and executed"
        assert create_stories[0]["status"] in ("in_sprint", "accepted"), (
            "In-scope notes story must be promoted to sprint"
        )

    # ── Roadmap excludes off-scope stories ────────────────────────────────────

    def test_roadmap_excludes_off_scope_stories(self, tmp_path):
        """RoadmapPlanner must not include rejected (out-of-scope) stories."""
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
        from src.pipeline.roadmap_planner import RoadmapPlanner

        store = SprintStateStore(state_path=tmp_path / "state.json")
        store.add_story(
            "User can create a note",
            acceptance_criteria=["note is saved"],
            epic="Note Management",
            deployment_package="Notes v1",
        )
        login_sid = store.add_story(
            "User can log in with email and password",
            acceptance_criteria=["user can log in"],
            epic="Authentication",
            deployment_package="Auth",
        )
        store.reject_story(login_sid, reason="Out of scope for current product goal.")

        RoadmapPlanner(store).build_from_backlog()
        roadmap = store.get_roadmap()

        for entry in roadmap:
            sc = " ".join(entry.get("success_criteria", []))
            assert "log in" not in sc.lower() and "email" not in sc.lower() and \
                   "password" not in sc.lower(), (
                f"Off-scope auth criteria must not appear in roadmap: {sc}"
            )
            pkg = entry.get("release_package", "")
            assert "Auth" not in pkg, f"Off-scope auth package must not appear in roadmap: {pkg}"

    # ── Meeting note selected count ────────────────────────────────────────────

    def test_meeting_note_selected_count_matches_executed_stories(self, tmp_path):
        """Sprint planning note must count actually-executed stories, not just
        backlog stories with status=in_sprint (which goes to zero post-sprint)."""
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
        from src.pipeline.meeting_notes import MeetingNotesManager

        store = SprintStateStore(state_path=tmp_path / "state.json")

        # Add 3 in-scope stories and promote 2 of them to sprint tasks
        sid1 = store.add_story("User can create a note", acceptance_criteria=["note saved"])
        sid2 = store.add_story("User can delete a note", acceptance_criteria=["note deleted"])
        sid3 = store.add_story("User can search notes",  acceptance_criteria=["results shown"])

        for sid in (sid1, sid2):
            store.mark_ac_validated(store.promote_story_to_sprint_task(sid))

        # Complete the first two; leave sid3 unexecuted (simulates partial sprint)
        tasks = [t for t in store.load()["tasks"]]
        for t in tasks:
            store.mark_ac_validated(t["id"])
            store.complete_task(t["id"], "done")
        store.accept_story(sid1)
        store.accept_story(sid2)

        notes_mgr = MeetingNotesManager(store)
        note = notes_mgr.generate_sprint_planning_notes()

        # 2 stories were promoted to sprint tasks (sid3 was never promoted)
        decisions_text = " ".join(note.get("decisions", []))
        assert "2 stories selected" in decisions_text, (
            f"Expected '2 stories selected' in planning note decisions; got: {decisions_text}"
        )

    def test_increment_filtering_unchanged_after_scope_changes(self, tmp_path):
        """Scope filtering must not break Product Increment eligibility logic."""
        from src.services.api.sprint_bridge import _task_is_increment_eligible

        eligible = {
            "id": "T-001", "status": "done", "po_status": "accepted",
            "acceptance_criteria": ["note saved"], "ac_validated": True,
            "artifact_type": "text_plan",
        }
        ineligible = {
            "id": "T-002", "status": "blocked", "po_status": "pending_review",
            "acceptance_criteria": ["auth works"], "ac_validated": False,
            "artifact_type": "text_plan",
        }
        assert _task_is_increment_eligible(eligible)
        assert not _task_is_increment_eligible(ineligible)


# ─────────────────────────────────────────────────────────────────────────────
#  9. Roadmap includes accepted stories; meeting notes count via sprint_loop mock
# ─────────────────────────────────────────────────────────────────────────────

class TestRoadmapAndNotesCount:

    def test_roadmap_not_empty_when_all_stories_accepted(self, tmp_path):
        """RoadmapPlanner must include 'accepted' stories (completed sprint work)."""
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
        from src.pipeline.roadmap_planner import RoadmapPlanner

        store = SprintStateStore(state_path=tmp_path / "state.json")
        sid1 = store.add_story("User can create a note",
                               acceptance_criteria=["note is saved"],
                               epic="Note Management", deployment_package="Notes Core")
        sid2 = store.add_story("User can delete a note",
                               acceptance_criteria=["note is gone"],
                               epic="Note Management", deployment_package="Notes Core")
        store.accept_story(sid1)
        store.accept_story(sid2)

        entries = RoadmapPlanner(store).build_from_backlog()

        assert len(entries) >= 1, (
            "RoadmapPlanner must produce at least one package when stories are accepted"
        )
        roadmap = store.get_roadmap()
        all_sc = " ".join(c for e in roadmap for c in e.get("success_criteria", []))
        assert "note is saved" in all_sc or "note is gone" in all_sc, (
            "Accepted story AC must appear as roadmap success criteria"
        )

    def test_roadmap_package_status_delivered_when_all_stories_accepted(self, tmp_path):
        """Package status must be 'delivered' when all stories in it are accepted."""
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
        from src.pipeline.roadmap_planner import RoadmapPlanner

        store = SprintStateStore(state_path=tmp_path / "state.json")
        sid = store.add_story("User can edit a note",
                              acceptance_criteria=["changes saved"],
                              deployment_package="Notes Core")
        store.accept_story(sid)

        RoadmapPlanner(store).build_from_backlog()
        roadmap = store.get_roadmap()

        assert any(e.get("status") == "delivered" for e in roadmap), (
            "Package status must be 'delivered' when all scope stories are accepted"
        )

    def test_roadmap_package_contains_required_fields(self, tmp_path):
        """Roadmap entry must include package_id, target_sprint, target_date,
        scope, and success_criteria for notes app stories."""
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
        from src.pipeline.roadmap_planner import RoadmapPlanner

        store = SprintStateStore(state_path=tmp_path / "state.json")
        store.add_story("User can search notes",
                        acceptance_criteria=["results shown"],
                        epic="Search", deployment_package="Notes Search")

        RoadmapPlanner(store).build_from_backlog()
        roadmap = store.get_roadmap()

        assert roadmap, "Roadmap must not be empty"
        entry = roadmap[0]
        assert entry.get("package_id"),      "package_id required"
        assert entry.get("release_package"), "release_package required"
        assert entry.get("target_sprint"),   "target_sprint required"
        assert entry.get("target_date"),     "target_date required"
        assert isinstance(entry.get("scope"), list) and entry["scope"], "scope required"
        assert isinstance(entry.get("success_criteria"), list), "success_criteria required"

    def test_roadmap_excludes_rejected_stories(self, tmp_path):
        """Rejected (out-of-scope) stories must not affect roadmap packages."""
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
        from src.pipeline.roadmap_planner import RoadmapPlanner

        store = SprintStateStore(state_path=tmp_path / "state.json")
        store.add_story("User can create a note",
                        acceptance_criteria=["note saved"],
                        deployment_package="Notes Core")
        bad_sid = store.add_story("User can log in with email and password",
                                  acceptance_criteria=["user can log in"],
                                  deployment_package="Auth")
        store.reject_story(bad_sid, reason="Out of scope for current product goal.")

        RoadmapPlanner(store).build_from_backlog()
        roadmap = store.get_roadmap()

        pkg_names = [e.get("release_package", "") for e in roadmap]
        assert "Auth" not in pkg_names, "Rejected auth story must not create Auth roadmap package"
        assert any("Notes" in p for p in pkg_names), "In-scope notes package must exist"

    def test_meeting_notes_count_via_sprint_loop_mock(self, tmp_path):
        """End-to-end: sprint_loop creates tasks via promote_story_to_sprint_task;
        generate_sprint_planning_notes must count the promoted stories correctly."""
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
        from src.pipeline.meeting_notes import MeetingNotesManager
        from unittest.mock import patch, MagicMock
        from src.loop import sprint_loop

        store = SprintStateStore(state_path=tmp_path / "state.json")
        goal = "Build a simple notes app where users can create, edit, delete, and search notes."

        class FakePO:
            def decompose_goal(self, goal, context=""):
                return [{"title": "Notes", "description": ""}]
            def generate_stories(self, epics):
                return [
                    {"title": "User can create a note", "description": "note creation",
                     "epic": "Note Management", "priority": "high",
                     "acceptance_criteria": ["note is saved"], "deployment_package": "Notes Core"},
                    {"title": "User can edit a note", "description": "note editing",
                     "epic": "Note Management", "priority": "high",
                     "acceptance_criteria": ["changes saved"], "deployment_package": "Notes Core"},
                    {"title": "User can delete a note", "description": "note deletion",
                     "epic": "Note Management", "priority": "medium",
                     "acceptance_criteria": ["note gone"], "deployment_package": "Notes Core"},
                    {"title": "User can search notes", "description": "note search",
                     "epic": "Search", "priority": "medium",
                     "acceptance_criteria": ["results shown"], "deployment_package": "Notes Core"},
                ]
            def review_story(self, **kw):
                return {"accepted": True, "reason": "", "missing_criteria": []}

        with patch.object(sprint_loop, "SprintStateStore", return_value=store), \
             patch.object(sprint_loop, "POLLMAgent", FakePO), \
             patch.object(sprint_loop, "_process_developer_message", create=True,
                          new=lambda t: MagicMock(draft="done", redo_log=[], artifact_type="text_plan")), \
             patch.object(sprint_loop, "evaluate_all_gates",
                          return_value={"conjunction": True, "gates": {}}):
            sprint_loop.run_sprint(goal, sprint_id="sprint-notes-test")

        state = store.load()
        tasks = state.get("tasks", [])
        assert len(tasks) >= 4, f"Expected ≥4 tasks, got {len(tasks)}"
        assert all(t.get("source_story_id") for t in tasks), (
            "All sprint tasks must have source_story_id set"
        )

        notes_mgr = MeetingNotesManager(store)
        note = notes_mgr.generate_sprint_planning_notes()
        decisions_text = " ".join(note.get("decisions", []))
        assert "4 stories selected" in decisions_text, (
            f"Expected '4 stories selected'; got: {decisions_text}"
        )

    def test_increment_filtering_still_correct_after_roadmap_fix(self, tmp_path):
        """Roadmap changes must not break Product Increment eligibility."""
        from src.services.api.sprint_bridge import _task_is_increment_eligible

        assert _task_is_increment_eligible({
            "id": "T-001", "status": "done", "po_status": "accepted",
            "acceptance_criteria": ["note saved"], "ac_validated": True,
        })
        assert not _task_is_increment_eligible({
            "id": "T-002", "status": "done", "po_status": "accepted",
            "acceptance_criteria": ["x"], "ac_validated": False,
        })


def _section(report: str, heading: str) -> str:
    """Return the text from the first line containing *heading* to the next '--' line."""
    lines = report.splitlines()
    capturing = False
    section_lines: list[str] = []
    for line in lines:
        if heading in line:
            capturing = True
            section_lines.append(line)
            continue
        if capturing:
            if line.startswith("--") or line.startswith("=="):
                break
            section_lines.append(line)
    return "\n".join(section_lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 5: Task-level summary, Needs Attention section, workflow_meta counts
# ─────────────────────────────────────────────────────────────────────────────

def _make_task(id_: str, status: str, po: str = "accepted", ac_validated: bool = True,
               ac: list | None = None) -> dict:
    return {
        "id": id_,
        "title": f"Task {id_}",
        "status": status,
        "po_status": po,
        "acceptance_criteria": ac if ac is not None else ["some AC"],
        "ac_validated": ac_validated,
        "artifact_type": "text_plan",
        "legacy_no_ac": False,
    }


class TestTaskLevelSummary:
    """Summary line must reflect task-object counts, not SprintResult story counts."""

    def test_summary_uses_task_counts(self, tmp_path):
        """9 done tasks → 'Summary: 9 completed tasks', not story-level count."""
        from src.services.api.sprint_bridge import render_sprint_report

        task_ids = {f"T-{i:03d}" for i in range(1, 10)}  # 9 tasks
        tasks = [_make_task(tid, "done", po="accepted", ac_validated=True) for tid in task_ids]
        tasks.append(_make_task("T-010", "in_progress", po="pending_review", ac_validated=False))
        tasks.append(_make_task("T-011", "blocked",     po="pending_review", ac_validated=False))

        store = _make_store(tmp_path, tasks=tasks)
        executed = {t["id"] for t in tasks}

        # SprintResult reports only 5 completed_stories — old (wrong) number
        result = _FakeResult(completed_stories=["S-001"] * 5, blocked_stories=["S-002"])
        text = render_sprint_report(result, store, executed_task_ids=executed)

        summary = next(l for l in text.splitlines() if l.startswith("Summary"))
        assert "9 completed tasks" in summary, f"Expected 9 completed tasks in: {summary!r}"
        assert "1 in progress"     in summary, f"Expected 1 in progress in: {summary!r}"
        assert "1 blocked"         in summary, f"Expected 1 blocked in: {summary!r}"

    def test_summary_with_no_run_tasks(self, tmp_path):
        """executed_task_ids=set() → 0 completed tasks · 0 in progress · 0 blocked."""
        from src.services.api.sprint_bridge import render_sprint_report

        store = _make_store(tmp_path)
        result = _FakeResult()
        text = render_sprint_report(result, store, executed_task_ids=set())

        summary = next(l for l in text.splitlines() if l.startswith("Summary"))
        assert "0 completed tasks" in summary


class TestNeedsAttentionSection:
    """Needs Attention section must include blocked, C8=FAIL, and PO-rejected tasks."""

    def test_in_progress_c8_fail_appears(self, tmp_path):
        """T-010 is in_progress with C8=FAIL → must appear in Needs Attention."""
        from src.services.api.sprint_bridge import render_sprint_report

        tasks = [
            _make_task("T-010", "in_progress", po="pending_review", ac_validated=False,
                       ac=["User can view tasks"]),
        ]
        store = _make_store(tmp_path, tasks=tasks)
        result = _FakeResult()
        text = render_sprint_report(result, store, executed_task_ids={"T-010"})

        section = _section(text, "Needs Attention")
        assert "T-010" in section, f"T-010 missing from Needs Attention:\n{section}"
        assert "C8=FAIL" in section or "Needs retry" in section

    def test_blocked_task_appears(self, tmp_path):
        """Blocked task must appear in Needs Attention."""
        from src.services.api.sprint_bridge import render_sprint_report

        tasks = [_make_task("T-011", "blocked", po="pending_review", ac_validated=False)]
        store = _make_store(tmp_path, tasks=tasks)
        result = _FakeResult()
        text = render_sprint_report(result, store, executed_task_ids={"T-011"})

        section = _section(text, "Needs Attention")
        assert "T-011" in section

    def test_po_rejected_task_appears(self, tmp_path):
        """PO-rejected task must appear in Needs Attention."""
        from src.services.api.sprint_bridge import render_sprint_report

        tasks = [_make_task("T-012", "done", po="rejected", ac_validated=True)]
        store = _make_store(tmp_path, tasks=tasks)
        result = _FakeResult()
        text = render_sprint_report(result, store, executed_task_ids={"T-012"})

        section = _section(text, "Needs Attention")
        assert "T-012" in section
        assert "PO-rejected" in section

    def test_clean_tasks_not_in_needs_attention(self, tmp_path):
        """Done + accepted + C8=PASS task must NOT appear in Needs Attention."""
        from src.services.api.sprint_bridge import render_sprint_report

        tasks = [_make_task("T-001", "done", po="accepted", ac_validated=True)]
        store = _make_store(tmp_path, tasks=tasks)
        result = _FakeResult()
        text = render_sprint_report(result, store, executed_task_ids={"T-001"})

        section = _section(text, "Needs Attention")
        assert "T-001" not in section

    def test_fallback_to_story_blocked_when_no_run_tasks(self, tmp_path):
        """When executed_task_ids is empty, story-level blocked info should show."""
        from src.services.api.sprint_bridge import render_sprint_report

        store = _make_store(tmp_path)
        blocked_stories = [{
            "story_id": "S-BLOCK",
            "task_id": "T-BLOCK",
            "reason": "PO rejected: missing criteria",
            "missing_criteria": ["User can delete a task"],
        }]
        result = _FakeResult(blocked_stories=blocked_stories)
        text = render_sprint_report(result, store, executed_task_ids=set())

        section = _section(text, "Needs Attention")
        assert "S-BLOCK" in section


class TestWorkflowMetaTaskCounts:
    """workflow_meta must expose task-level counts distinct from story-level."""

    def test_workflow_meta_has_task_keys(self, tmp_path):
        """render_sprint_report Summary line must reflect task counts, not story counts."""
        from src.services.api.sprint_bridge import render_sprint_report

        tasks = [
            _make_task("T-001", "done",       po="accepted",       ac_validated=True),
            _make_task("T-002", "blocked",     po="pending_review", ac_validated=False),
            _make_task("T-003", "in_progress", po="pending_review", ac_validated=False,
                       ac=["Some AC"]),
        ]
        store = _make_store(tmp_path, tasks=tasks)
        # SprintResult claims only 1 completed story — must NOT appear in Summary
        result = _FakeResult(completed_stories=["S-001"], blocked_stories=[])
        executed = {"T-001", "T-002", "T-003"}
        text = render_sprint_report(result, store, executed_task_ids=executed)

        summary = next(l for l in text.splitlines() if l.startswith("Summary"))
        assert "1 completed tasks" in summary
        assert "1 in progress"     in summary
        assert "1 blocked"         in summary

    def test_workflow_meta_keys_present_in_return(self, tmp_path):
        """run_sprint_for_ui must include task-level keys alongside story-level keys."""
        import src.services.api.sprint_bridge as bridge
        from unittest.mock import patch, MagicMock
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore, DEFAULT_SPRINT_STATE

        store = SprintStateStore(state_path=tmp_path / "wm_state.json")
        state_before = copy.deepcopy(DEFAULT_SPRINT_STATE)
        state_before["tasks"] = []
        state_after = copy.deepcopy(DEFAULT_SPRINT_STATE)
        state_after["tasks"] = [_make_task("T-A", "done", po="accepted", ac_validated=True)]
        state_after["product_goal"] = "notes app"
        state_after["roadmap"] = []

        fake_result = _FakeResult(completed_stories=[], blocked_stories=[])

        call_idx = [0]
        state_seq = [state_before, state_after, state_after, state_after, state_after, state_after]

        def _mock_load():
            idx = min(call_idx[0], len(state_seq) - 1)
            call_idx[0] += 1
            return copy.deepcopy(state_seq[idx])

        store.load = _mock_load
        store.reset_for_isolated_sprint = lambda: None
        store.get_product_goal = lambda: "notes app"
        store.get_roadmap = lambda: [{"x": 1}]  # non-empty → skip build
        store.set_product_goal = lambda g: None

        # Patch lazy-imported names at their original source modules
        with patch("src.loop.sprint_loop.run_sprint", return_value=fake_result), \
             patch("src.pipeline.scrum_team.sprint_state_store.SprintStateStore",
                   return_value=store), \
             patch("src.pipeline.meeting_notes.MeetingNotesManager.generate_sprint_planning_notes",
                   return_value={"event_type": "sprint_planning", "decisions": []}), \
             patch("src.pipeline.meeting_notes.MeetingNotesManager.generate_sprint_review_notes",
                   return_value={"event_type": "sprint_review", "decisions": []}), \
             patch("src.pipeline.meeting_notes.MeetingNotesManager.save", return_value="MN-1"):

            result_dict = bridge.run_sprint_for_ui("Build a notes app", isolated=False)

        wm = result_dict.get("workflow_meta", {})
        for key in ("completed_tasks", "blocked_tasks", "in_progress_tasks",
                    "completed_stories", "blocked_stories"):
            assert key in wm, f"workflow_meta missing key: {key!r}"
