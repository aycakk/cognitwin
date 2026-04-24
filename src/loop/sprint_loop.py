"""loop/sprint_loop.py — Autonomous sprint runner (MVP Phase 1).

Public interface:  run_sprint(goal, sprint_id=None) → SprintResult

Sprint phases
─────────────
  ANALYZE  — classify goal type, load cross-sprint history context
  PLAN     — POLLMAgent decomposes goal into epics + user stories, persisted to backlog
  EXECUTE  — each backlog story is promoted to a sprint task, developer is called,
             gate results checked; ComposerOrchestrator decides reroute on failure
  VALIDATE — compute average confidence, persist snapshot to SprintMemoryStore

Safety guards
─────────────
  MAX_STEPS_PER_SPRINT = 20  (hard ceiling, imported from ComposerOrchestrator)
  MAX_REROUTE_PER_TASK = 3   (per-task retry budget, imported from ComposerOrchestrator)

If either limit is reached the sprint returns a partial SprintResult rather
than looping forever.

Reuse
─────
  - SprintStateStore  — existing, used for all state mutations
  - _process_developer_message  — existing developer pipeline path
  - evaluate_all_gates          — existing gate evaluator
  - REDO loop                   — already embedded inside developer runner
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.agents.composer_orchestrator import (
    ComposerOrchestrator,
    RerouteAction,
    MAX_STEPS_PER_SPRINT,
    MAX_REROUTE_PER_TASK,
)
from src.agents.po_llm_agent import POLLMAgent
from src.core.schemas import AgentTask, AgentRole
from src.gates.evaluator import evaluate_all_gates
from src.gates.gate_result import build_gate_result
from src.memory.sprint_memory_store import SprintMemoryStore
from src.pipeline.scrum_team.sprint_state_store import SprintStateStore

# developer_runner is imported lazily inside run_sprint() to avoid pulling in
# the ollama module at import time (keeps non-LLM tests importable without Ollama).
_process_developer_message = None  # populated on first run_sprint() call

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SprintResult:
    sprint_id:         str
    goal:              str
    completed_stories: list[dict[str, Any]] = field(default_factory=list)
    blocked_stories:   list[dict[str, Any]] = field(default_factory=list)
    total_steps:       int   = 0
    avg_confidence:    float = 0.0
    summary:           str   = ""


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _avg_confidence_from_report(gate_report: dict) -> float:
    """Compute mean confidence score across all gates in a report."""
    scores = [
        build_gate_result(gid, info.get("pass", False), info.get("evidence", "")).confidence_score
        for gid, info in gate_report.get("gates", {}).items()
    ]
    return round(sum(scores) / len(scores), 2) if scores else 0.0


def _build_dev_query(story: dict | None, task_id: str) -> str:
    """Build a developer-facing query string from a backlog story dict."""
    if not story:
        return f"Implement task {task_id}."

    title       = story.get("title", f"task {task_id}")
    description = story.get("description", "")
    criteria    = story.get("acceptance_criteria", [])

    parts = [f"Implement the following: {title}"]
    if description and description.strip() and description.strip() != title:
        parts.append(f"Description: {description.strip()}")
    if criteria:
        parts.append("Acceptance criteria: " + "; ".join(str(c) for c in criteria[:3]))
    return ". ".join(parts)


def _format_summary(
    goal:      str,
    completed: list[dict],
    blocked:   list[dict],
    avg_conf:  float,
    steps:     int,
    prefix:    str = "",
) -> str:
    lines = [
        f"{prefix}=== SPRINT RUN SUMMARY ===",
        f"Goal       : {goal}",
        f"Completed  : {len(completed)} {'story' if len(completed) == 1 else 'stories'}",
        f"Blocked    : {len(blocked)} {'story' if len(blocked) == 1 else 'stories'}",
        f"Confidence : {avg_conf:.0%}",
        f"Steps used : {steps}/{MAX_STEPS_PER_SPRINT}",
    ]
    if completed:
        lines.append("Completed:")
        for s in completed:
            lines.append(
                f"  ✓ {s.get('story_id', '?')} [{s.get('task_id', '?')}]"
                f"  {s.get('title', '-')[:60]}"
            )
    if blocked:
        lines.append("Blocked:")
        for s in blocked:
            lines.append(
                f"  ✗ {s.get('story_id', '?')}"
                f"  {s.get('title', s.get('reason', '-'))[:60]}"
                + (f"  — {s.get('reason', '')[:50]}" if s.get("reason") else "")
            )
    lines.append("==========================")
    return "\n".join(lines)


def _persist_and_build_result(
    sprint_id:    str,
    goal:         str,
    completed:    list[dict],
    blocked:      list[dict],
    steps:        int,
    conf_scores:  list[float],
    memory_store: SprintMemoryStore,
    prefix:       str = "",
) -> SprintResult:
    avg_conf = round(sum(conf_scores) / len(conf_scores), 2) if conf_scores else 0.0

    memory_store.append_sprint({
        "sprint_id":         sprint_id,
        "goal":              goal,
        "completed_stories": completed,
        "blocked_stories":   blocked,
        "avg_confidence":    avg_conf,
        "step_count":        steps,
    })

    summary = _format_summary(goal, completed, blocked, avg_conf, steps, prefix=prefix)
    return SprintResult(
        sprint_id=sprint_id,
        goal=goal,
        completed_stories=completed,
        blocked_stories=blocked,
        total_steps=steps,
        avg_confidence=avg_conf,
        summary=summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_sprint(goal: str, sprint_id: str | None = None) -> SprintResult:
    """
    Run a single autonomous sprint from a high-level goal string.

    Phases: analyze → plan → execute → validate

    Parameters
    ----------
    goal : str
        Human-readable goal, e.g. "Add login screen" or "Fix auth bug".
    sprint_id : str | None
        Optional explicit ID. Auto-generated as sprint-auto-YYYYMMDD-HHMMSS
        when not provided.

    Returns
    -------
    SprintResult
        Summary of what was completed, what was blocked, step count, and
        average gate confidence.
    """
    # Lazy import — avoids pulling ollama into the module namespace at import time
    global _process_developer_message
    if _process_developer_message is None:
        from src.pipeline.developer_runner import _process_developer_message as _pdm  # noqa: PLC0415
        _process_developer_message = _pdm

    state_store  = SprintStateStore()
    memory_store = SprintMemoryStore()
    po_agent     = POLLMAgent()
    orchestrator = ComposerOrchestrator(state_store)

    sprint_id  = sprint_id or f"sprint-auto-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    step_count = 0

    completed_stories: list[dict[str, Any]] = []
    blocked_stories:   list[dict[str, Any]] = []
    conf_scores:       list[float]          = []

    logger.info("sprint_loop: ── START sprint=%s goal=%r", sprint_id, goal[:60])

    # ─────────────────────────────────────────────────────────────────
    #  PHASE 1 — ANALYZE
    # ─────────────────────────────────────────────────────────────────
    history_context = memory_store.get_history_context()
    analysis        = orchestrator.analyze(goal, history_context)

    # Stamp the goal into sprint state so downstream agents can see it
    state_store.set_sprint_goal(f"[AUTO] {goal[:180]}")
    step_count += 1

    logger.info("sprint_loop: ANALYZE done — goal_type=%s step=%d", analysis["goal_type"], step_count)

    if step_count >= MAX_STEPS_PER_SPRINT:
        logger.warning("sprint_loop: step limit hit during ANALYZE")
        return _persist_and_build_result(
            sprint_id, goal, completed_stories, blocked_stories,
            step_count, conf_scores, memory_store,
            prefix="[STEP LIMIT — ANALYZE] ",
        )

    # ─────────────────────────────────────────────────────────────────
    #  PHASE 2 — PLAN
    # ─────────────────────────────────────────────────────────────────
    logger.info("sprint_loop: PLAN — decomposing goal into epics")
    # Inject retrospective actions and roadmap into the PO planning context
    # so the next sprint actually learns from the previous one.
    retro_actions = state_store.get_retro_actions()
    roadmap       = state_store.get_roadmap()
    extra_context_parts: list[str] = []
    if analysis.get("context"):
        extra_context_parts.append(analysis["context"])
    if retro_actions:
        extra_context_parts.append(
            "Retrospective action items from previous sprint:\n- "
            + "\n- ".join(retro_actions)
        )
    if roadmap:
        roadmap_summary = "; ".join(
            f"{e.get('package_id', '?')}→sprint {e.get('target_sprint', '?')}"
            for e in roadmap[:5]
        )
        extra_context_parts.append(f"Current roadmap packages: {roadmap_summary}")
    plan_context = "\n\n".join(extra_context_parts)
    epics = po_agent.decompose_goal(goal, plan_context)
    step_count += 1

    logger.info("sprint_loop: PLAN — %d epics; generating stories", len(epics))

    if step_count >= MAX_STEPS_PER_SPRINT:
        logger.warning("sprint_loop: step limit hit after epic decomposition")
        return _persist_and_build_result(
            sprint_id, goal, completed_stories, blocked_stories,
            step_count, conf_scores, memory_store,
            prefix="[STEP LIMIT — PLAN/epics] ",
        )

    stories = po_agent.generate_stories(epics)
    step_count += 1

    logger.info("sprint_loop: PLAN — %d stories generated", len(stories))

    # Persist each story to the backlog
    story_ids: list[str] = []
    for story_data in stories:
        if step_count >= MAX_STEPS_PER_SPRINT:
            break
        sid = state_store.add_story(
            title=story_data.get("title", "Untitled story"),
            description=story_data.get("description", ""),
            priority=story_data.get("priority", "medium"),
            acceptance_criteria=story_data.get("acceptance_criteria", []),
        )
        story_ids.append(sid)
        step_count += 1

    logger.info("sprint_loop: PLAN done — %d stories added to backlog (step=%d)", len(story_ids), step_count)

    # Prefer pre-existing backlog items whose target_sprint matches this
    # sprint id (roadmap-driven planning). When no such items exist,
    # fall back to the newly-generated stories above.
    targeted_story_ids = [
        s.get("story_id")
        for s in state_store.get_backlog()
        if s.get("target_sprint") == sprint_id
        and s.get("status") in ("draft", "ready", "needs_refinement")
        and s.get("story_id")
    ]
    if targeted_story_ids:
        logger.info(
            "sprint_loop: PLAN — roadmap targets %d existing stories for sprint=%s",
            len(targeted_story_ids), sprint_id,
        )
        # Prepend so roadmap-targeted work runs first, then any new stories
        seen = set(targeted_story_ids)
        story_ids = targeted_story_ids + [sid for sid in story_ids if sid not in seen]

    # ─────────────────────────────────────────────────────────────────
    #  PHASE 3 — EXECUTE
    # ─────────────────────────────────────────────────────────────────
    logger.info("sprint_loop: EXECUTE — processing %d stories", len(story_ids))

    for story_id in story_ids:
        story       = state_store.get_story(story_id)
        story_title = story.get("title", story_id) if story else story_id

        if step_count >= MAX_STEPS_PER_SPRINT:
            logger.warning("sprint_loop: step limit hit, deferring story=%s", story_id)
            blocked_stories.append({
                "story_id": story_id,
                "title":    story_title,
                "reason":   "Step limit reached before execution",
            })
            continue

        # Promote backlog story → sprint task
        task_id = state_store.promote_story_to_sprint_task(story_id)
        if not task_id:
            logger.error("sprint_loop: promote failed for story=%s", story_id)
            blocked_stories.append({
                "story_id": story_id,
                "title":    story_title,
                "reason":   "Failed to promote story to sprint task",
            })
            continue

        # Assign task to the default developer and mark in-progress
        state_store.assign_task(task_id, "developer-default")
        state_store.start_task(task_id)
        step_count += 1

        dev_query      = _build_dev_query(story, task_id)
        sprint_context = state_store.read_context_block()

        logger.info(
            "sprint_loop: EXECUTE story=%s task=%s step=%d",
            story_id, task_id, step_count,
        )

        # ── Reroute loop ──────────────────────────────────────────────
        task_completed = False
        final_result   = ""
        reroute_count  = 0

        while reroute_count <= MAX_REROUTE_PER_TASK and step_count < MAX_STEPS_PER_SPRINT:
            dev_task = AgentTask(
                role=AgentRole.DEVELOPER,
                masked_input=dev_query,
                metadata={"strategy": "auto"},
                context={
                    "acceptance_criteria": story.get("acceptance_criteria", []) if story else [],
                    "source_story_id":     story_id,
                },
            )

            response   = _process_developer_message(dev_task)
            step_count += 1

            gate_report = evaluate_all_gates(
                response.draft,
                "",       # no academic vector context on developer path
                True,     # is_empty=True (C2 uses academic namespace, not active here)
                "DeveloperAgent",
                response.redo_log,
                codebase_context=sprint_context,
                acceptance_criteria=story.get("acceptance_criteria", []) if story else [],
            )

            conf = _avg_confidence_from_report(gate_report)
            logger.debug(
                "sprint_loop: gate check story=%s reroute=%d pass=%s conf=%.2f",
                story_id, reroute_count, gate_report["conjunction"], conf,
            )

            if gate_report["conjunction"]:
                # All gates passed — mark AC validated so complete_task allows done
                if story and story.get("acceptance_criteria"):
                    state_store.mark_ac_validated(task_id)

                # PO review: gate-pass alone is not acceptance.
                # The PO must confirm the task output evidences the criteria.
                review = po_agent.review_story(
                    story               = story or {},
                    task_output         = response.draft,
                    acceptance_criteria = story.get("acceptance_criteria", []) if story else [],
                    sprint_goal         = state_store.get_sprint_goal(),
                )
                if review.get("accepted"):
                    conf_scores.append(conf)
                    task_completed = True
                    final_result   = response.draft
                    break

                # PO rejected — route back to Developer with the rejection reason.
                if reroute_count >= MAX_REROUTE_PER_TASK:
                    # Budget exhausted — treat as escalation
                    if story_id:
                        state_store.reject_story(story_id, review.get("reason", "")[:200])
                    state_store.block_task(task_id, f"PO rejected: {review.get('reason', '')[:150]}")
                    blocked_stories.append({
                        "story_id":         story_id,
                        "task_id":          task_id,
                        "title":            story_title,
                        "reason":           f"PO rejected: {review.get('reason', '')}",
                        "missing_criteria": review.get("missing_criteria", []),
                    })
                    break

                reroute_count += 1
                step_count    += 1
                missing = review.get("missing_criteria") or []
                dev_query = dev_query + (
                    f"\n[PO REJECTED — {review.get('reason', 'AC not met')}]"
                    + (f"\nMissing criteria: {'; '.join(missing[:3])}" if missing else "")
                )
                logger.info(
                    "sprint_loop: PO rejected story=%s reroute=%d reason=%r",
                    story_id, reroute_count, review.get("reason", ""),
                )
                continue

            # Ask orchestrator what to do with the failure
            reroute = orchestrator.reroute(story_id, gate_report, reroute_count)
            logger.info(
                "sprint_loop: reroute story=%s action=%s reason=%r",
                story_id, reroute.action.value, reroute.reason,
            )

            if reroute.action == RerouteAction.RETRY:
                reroute_count += 1
                step_count    += 1
                # Inject revision hint into the query for the next attempt
                failing = [
                    gid for gid, info in gate_report["gates"].items()
                    if not info.get("pass", True)
                ]
                if failing:
                    from src.gates.gate_result import build_gate_result as _bgr
                    hint_gr   = _bgr(failing[0], False, gate_report["gates"][failing[0]].get("evidence", ""))
                    dev_query = dev_query + f"\n[REVISION REQUIRED — {failing[0]}]: {hint_gr.revision_hint}"
                continue

            elif reroute.action == RerouteAction.SKIP:
                # Accept imperfect output (e.g. no failing gates found defensively)
                conf_scores.append(conf)
                task_completed = True
                final_result   = response.draft
                break

            else:  # ESCALATE
                state_store.block_task(task_id, reroute.reason)
                blocked_stories.append({
                    "story_id": story_id,
                    "task_id":  task_id,
                    "title":    story_title,
                    "reason":   reroute.reason,
                })
                break

        # ── Finalise task state ───────────────────────────────────────
        if task_completed:
            ok = state_store.complete_task(task_id, final_result[:500])
            if not ok:
                # AC enforcement blocked completion — treat as escalation
                logger.warning(
                    "sprint_loop: complete_task blocked for task=%s (AC not validated)",
                    task_id,
                )
                state_store.block_task(task_id, "Acceptance criteria not validated")
                blocked_stories.append({
                    "story_id": story_id,
                    "task_id":  task_id,
                    "title":    story_title,
                    "reason":   "Acceptance criteria not validated",
                })
            else:
                # PO already accepted inside the loop — finalise increment.
                if story_id:
                    state_store.accept_story(story_id)
                    state_store.add_to_increment(task_id)
                completed_stories.append({
                    "story_id": story_id,
                    "task_id":  task_id,
                    "title":    story_title,
                })
                logger.info("sprint_loop: ✓ completed story=%s task=%s", story_id, task_id)

        elif not any(b.get("story_id") == story_id for b in blocked_stories):
            # Reroute budget or step budget exhausted without an explicit ESCALATE
            reason = (
                "Max reroutes exceeded"
                if reroute_count > MAX_REROUTE_PER_TASK
                else "Step limit reached during execution"
            )
            state_store.block_task(task_id, reason)
            blocked_stories.append({
                "story_id": story_id,
                "task_id":  task_id,
                "title":    story_title,
                "reason":   reason,
            })
            logger.warning("sprint_loop: ✗ blocked story=%s reason=%s", story_id, reason)

    # ─────────────────────────────────────────────────────────────────
    #  PHASE 4 — VALIDATE
    # ─────────────────────────────────────────────────────────────────
    avg_conf = round(sum(conf_scores) / len(conf_scores), 2) if conf_scores else 0.0

    logger.info(
        "sprint_loop: VALIDATE — completed=%d blocked=%d avg_conf=%.2f steps=%d/%d",
        len(completed_stories), len(blocked_stories), avg_conf,
        step_count, MAX_STEPS_PER_SPRINT,
    )

    return _persist_and_build_result(
        sprint_id, goal, completed_stories, blocked_stories,
        step_count, conf_scores, memory_store,
    )
