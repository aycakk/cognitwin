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
import os
import re as _re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# HUMAN_TASK_REVIEW_MODE — controls task-level story acceptance.
#
#   False (default) — PO Agent reviews every task after gates pass.
#                     accept  → agent_accepted → Done
#                     reject  → Developer retry → blocked
#                     Tasks never sit in waiting_review during normal execution.
#
#   True  (debug)   — Gate PASS → waiting_review → Human Supervisor acts.
#                     Set HUMAN_TASK_REVIEW_MODE=true in the environment.
#                     For experiments and manual inspection only.
#
# Human involvement in the sprint is limited to the Sprint Review stage (end of sprint).
def _resolve_human_task_review_mode() -> bool:
    """Return True only when HUMAN_TASK_REVIEW_MODE=true is set explicitly.

    AUTO_PO_ACCEPT is deprecated and ignored.  Setting it will emit a warning
    so operators know to migrate, but it will NOT enable task-level human review.
    """
    raw = os.getenv("HUMAN_TASK_REVIEW_MODE", "").strip().lower()
    if raw in ("true", "1", "yes"):
        return True
    if raw in ("false", "0", "no"):
        return False
    # Warn about deprecated flag but do NOT honour it.
    if os.getenv("AUTO_PO_ACCEPT", "").strip():
        logger.warning(
            "sprint_loop: AUTO_PO_ACCEPT is deprecated and has no effect. "
            "Use HUMAN_TASK_REVIEW_MODE=true to enable task-level human review."
        )
    return False  # default: PO Agent handles story acceptance, no human task review


_HUMAN_TASK_REVIEW_MODE: bool = _resolve_human_task_review_mode()
# Aliases kept for external callers that import the old names.
_ACCEPTANCE_MODE: str  = "human_required" if _HUMAN_TASK_REVIEW_MODE else "supervised"
_AUTO_PO_ACCEPT:  bool = not _HUMAN_TASK_REVIEW_MODE

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
from src.loop.sprint_compliance import validate_sprint_compliance
from src.memory.sprint_memory_store import SprintMemoryStore
from src.pipeline.scrum_team.sprint_state_store import SprintStateStore

# developer_runner is imported lazily inside run_sprint() to avoid pulling in
# the ollama module at import time (keeps non-LLM tests importable without Ollama).
_process_developer_message = None  # populated on first run_sprint() call

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Story scope validation
# ─────────────────────────────────────────────────────────────────────────────

_SCOPE_STOP_WORDS: frozenset[str] = frozenset({
    "user", "users", "the", "and", "for", "with", "that", "this", "from",
    "will", "their", "which", "have", "been", "when", "also", "should",
})

# Domain keyword sets — a story using any of these is ONLY in-scope when the
# product goal also mentions at least one term from the same set.
_GATED_DOMAINS: tuple[frozenset[str], ...] = (
    frozenset({                                            # infra / devops
        "kubernetes", "devops", "cicd", "terraform",
    }),
    frozenset({                                            # payments / billing
        "stripe", "invoice", "billing", "subscription",
    }),
    frozenset({                                            # backend / server-side API
        # Reject stories whose dominant subject is server-side infra when the goal
        # is frontend-only.  Only use unambiguous infrastructure nouns; avoid auth
        # terms (jwt, session) because they legitimately appear in frontend stories.
        "endpoint", "persistence", "backend",
        "postgres", "mysql", "sqlite", "mongodb",
    }),
)


def _story_aligns_with_goal(story: dict, goal: str) -> bool:
    """Permissive scope check: only reject stories whose dominant subject is
    a clearly unrelated technical domain (devops, billing) the goal never
    mentions. Keyword-overlap rejection has been removed — the PO's review
    step is the proper place to reject off-topic work.
    """
    def _words(text: str) -> frozenset[str]:
        return frozenset(_re.findall(r"\b[a-z]{3,}\b", (text or "").lower()))

    goal_words  = _words(goal)
    story_words = _words(
        " ".join([
            story.get("title", ""),
            story.get("description", ""),
            story.get("epic", ""),
        ])
    )
    for term_set in _GATED_DOMAINS:
        if story_words & term_set and not (goal_words & term_set):
            logger.warning(
                "sprint_loop: story off-domain (gated terms %s): %r",
                story_words & term_set, story.get("title", "?")[:60],
            )
            return False
    return True


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
    agile_compliance:  dict | None = None


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
    sprint_id:        str,
    goal:             str,
    completed:        list[dict],
    blocked:          list[dict],
    steps:            int,
    conf_scores:      list[float],
    memory_store:     SprintMemoryStore,
    prefix:           str = "",
    agile_compliance: dict | None = None,
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
        agile_compliance=agile_compliance,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_sprint(
    goal: str,
    sprint_id: str | None = None,
    event_callback=None,  # Optional[Callable[[str, str, str, Optional[str]], None]]
) -> SprintResult:
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
    # Shorthand so we don't repeat None-guard throughout
    def _emit(agent: str, etype: str, message: str, task_id=None) -> None:
        if event_callback is not None:
            try:
                event_callback(agent, etype, message, task_id)
            except Exception:
                pass  # never let event emission crash the sprint

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
    _emit("system", "status", f"Sprint started — goal: {goal[:120]}")

    # ─────────────────────────────────────────────────────────────────
    #  PHASE 1 — ANALYZE
    # ─────────────────────────────────────────────────────────────────
    _emit("po", "thought", f"Analyzing product goal: {goal[:100]}")
    history_context = memory_store.get_history_context()
    analysis        = orchestrator.analyze(goal, history_context)

    # Stamp the goal into sprint state so downstream agents can see it
    state_store.set_sprint_goal(f"[AUTO] {goal[:180]}")

    # Persist the product goal on first run (or if it was previously cleared).
    # sprint_bridge._clean_goal_for_po() has already stripped report instructions,
    # so `goal` here is the actual product scope, not formatting directives.
    if not state_store.get_product_goal() and goal:
        state_store.set_product_goal(goal[:300])
        logger.info("sprint_loop: persisted product_goal=%r", goal[:60])

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
    _emit("po", "action", "Decomposing goal into epics and user stories")
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
    _emit("po", "action", f"Decomposed into {len(epics)} epic(s) — generating user stories")

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
    _emit("po", "action", f"{len(stories)} user story/stories generated — adding to backlog")

    logger.info("sprint_loop: PLAN — %d stories generated", len(stories))

    # Persist each story to the backlog; reject out-of-scope stories immediately.
    story_ids: list[str] = []
    for story_data in stories:
        if step_count >= MAX_STEPS_PER_SPRINT:
            break
        sid = state_store.add_story(
            title=story_data.get("title", "Untitled story"),
            description=story_data.get("description", ""),
            priority=story_data.get("priority", "medium"),
            acceptance_criteria=story_data.get("acceptance_criteria", []),
            epic=story_data.get("epic", ""),
            deployment_package=story_data.get("deployment_package", ""),
        )
        if not _story_aligns_with_goal(story_data, goal):
            state_store.reject_story(sid, reason="Out of scope for current product goal.")
            logger.info(
                "sprint_loop: PLAN — story rejected (out of scope): %r",
                story_data.get("title", "?")[:60],
            )
            _emit("po", "action", f"Story rejected (out of scope): {story_data.get('title', '?')[:60]}")
            continue  # do not execute; do not count as a planning step
        story_ids.append(sid)
        _emit("po", "action", f"Backlog: [{sid}] {story_data.get('title', 'Untitled')[:70]}")
        step_count += 1

    logger.info("sprint_loop: PLAN done — %d in-scope stories queued (step=%d)", len(story_ids), step_count)
    _emit("sm", "ceremony", f"Sprint Planning — {len(story_ids)} story/stories queued for this sprint")

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
        _emit("sm", "action", f"Task {task_id} assigned — Developer starting work", task_id)
        _emit("dev", "thought", f"Working on [{task_id}]: {story_title[:70]}", task_id)

        dev_query      = _build_dev_query(story, task_id)
        sprint_context = state_store.read_context_block()

        logger.info(
            "sprint_loop: EXECUTE story=%s task=%s step=%d",
            story_id, task_id, step_count,
        )

        # ── Reroute loop ──────────────────────────────────────────────
        task_completed       = False
        final_result         = ""
        reroute_count        = 0
        _agent_review_result: dict | None = None

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
                _emit("gate", "gate", f"All gates passed for {task_id}", task_id)
                # All gates passed — mark AC validated so complete_task allows done
                if story and story.get("acceptance_criteria"):
                    state_store.mark_ac_validated(task_id)

                # Gate passed.
                if _HUMAN_TASK_REVIEW_MODE:
                    # Debug / experiment mode: route to Human Supervisor.
                    _emit("po", "thought", f"Task {task_id} passed gates — awaiting Human Supervisor review [debug mode]", task_id)
                    conf_scores.append(conf)
                    task_completed = True
                    final_result   = response.draft
                    break

                # Default: PO Agent reviews immediately — no human involvement.
                _emit("po", "thought", f"Product Owner Agent reviewing output for {task_id}", task_id)
                _agent_review_result = po_agent.review_story(
                    story               = story or {},
                    task_output         = response.draft,
                    acceptance_criteria = story.get("acceptance_criteria", []) if story else [],
                    sprint_goal         = state_store.get_sprint_goal(),
                )
                review = _agent_review_result
                if review.get("accepted"):
                    _emit("po", "action", f"Product Owner Agent accepted {task_id}", task_id)
                    conf_scores.append(conf)
                    task_completed = True
                    final_result   = response.draft
                    break

                # PO rejected — route back to Developer with the rejection reason.
                _emit("po", "action", f"Product Owner Agent rejected {task_id}: {review.get('reason', 'AC not met')[:80]}", task_id)
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
                try:
                    state_store.set_retry_count(task_id, reroute_count)
                except Exception:
                    pass
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

            # Gate failed — build human-readable detail records.
            failing = [gid for gid, info in gate_report["gates"].items() if not info.get("pass", True)]
            from src.gates.gate_result import build_gate_result as _bgr_detail  # noqa: PLC0415
            _GATE_TITLES = {
                "A1": "REDO audit cycle integrity",
                "C1": "PII leak detection",
                "C2": "Vector memory grounding",
                "C2_DEV": "Codebase grounding",
                "C3": "Ontology compliance",
                "C4": "Hallucination check",
                "C5": "Role permission scope",
                "C6": "Anti-sycophancy",
                "C7": "Blindspot disclosure",
                "C8": "Acceptance criteria coverage",
            }
            gate_detail_items: list[dict] = []
            for gid in failing[:5]:
                info = gate_report["gates"].get(gid, {})
                evidence = (info.get("evidence") or "")[:300]
                gr = _bgr_detail(gid, False, evidence)
                gate_detail_items.append({
                    "gate_id":          gid,
                    "title":            _GATE_TITLES.get(gid, gid),
                    "status":           "fail",
                    "reason":           evidence or "Gate evaluation failed.",
                    "suggested_action": gr.revision_hint,
                })
            try:
                state_store.attach_gate_failures(task_id, gate_detail_items)
            except Exception:
                pass
            _emit(
                "gate", "gate",
                f"Gate(s) failed for {task_id}: " + ", ".join(
                    f"{d['gate_id']} — {d['title']}" for d in gate_detail_items[:3]
                ),
                task_id,
            )

            # Ask orchestrator what to do with the failure
            reroute = orchestrator.reroute(story_id, gate_report, reroute_count)
            logger.info(
                "sprint_loop: reroute story=%s action=%s reason=%r",
                story_id, reroute.action.value, reroute.reason,
            )

            if reroute.action == RerouteAction.RETRY:
                reroute_count += 1
                step_count    += 1
                try:
                    state_store.set_retry_count(task_id, reroute_count)
                except Exception:
                    pass
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
            # Write any code files produced by the developer
            try:
                from src.loop.sprint_files import write_developer_files  # noqa: PLC0415
                written = write_developer_files(sprint_id, task_id, final_result)
                if written:
                    _emit("dev", "artifact", f"Generated files: {', '.join(written)}", task_id)
            except Exception as _fe:
                logger.warning("sprint_loop: sprint_files write failed: %s", _fe)

            ok = state_store.complete_task(
                task_id,
                final_result[:500],
                artifact_type=getattr(response, "artifact_type", "text_plan"),
            )
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
                _emit("dev", "action", f"Task {task_id} blocked — acceptance criteria not validated", task_id)
            else:
                if _HUMAN_TASK_REVIEW_MODE:
                    # Debug mode: complete_task already set po_status="ready_for_review";
                    # the board shows this in "Waiting Review" until a human acts.
                    _emit("dev", "artifact", f"Task {task_id} in Waiting Review — Human Supervisor review needed [debug mode]", task_id)
                    logger.info("sprint_loop: ⏳ waiting_review story=%s task=%s (debug)", story_id, task_id)
                else:
                    # Default: PO Agent accepted inside the reroute loop — record and close.
                    _rev = _agent_review_result or {}
                    if story_id:
                        state_store.apply_agent_review(
                            task_id          = task_id,
                            accepted         = True,
                            reason           = _rev.get("reason", "All acceptance criteria evidenced in task output."),
                            confidence       = _rev.get("confidence", 1.0),
                            missing_criteria = _rev.get("missing_criteria", []),
                        )
                        state_store.add_to_increment(task_id)
                    completed_stories.append({
                        "story_id": story_id,
                        "task_id":  task_id,
                        "title":    story_title,
                    })
                    _emit("dev", "artifact", f"Task {task_id} done — Product Owner Agent accepted", task_id)
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

    # Fallback artefact — guarantees cockpit Preview/Code panes always render
    # something when developer output produced no parsable code blocks.
    try:
        from src.loop.sprint_files import (  # noqa: PLC0415
            list_generated_files, write_fallback_artefact,
        )
        if not list_generated_files(sprint_id) and completed_stories:
            accepted_payload = [
                {
                    "title": s.get("title", ""),
                    "story_id": s.get("story_id", ""),
                    "description": (state_store.get_story(s.get("story_id", "")) or {}).get("description", ""),
                }
                for s in completed_stories
            ]
            written = write_fallback_artefact(sprint_id, goal, accepted_payload)
            if written:
                _emit("dev", "artifact", f"Fallback artefact: {', '.join(written)}")
    except Exception as _fe:
        logger.warning("sprint_loop: fallback artefact failed: %s", _fe)

    # Canonical summary — read from state store so the event stream message
    # matches the board API exactly (both use compute_summary on the same tasks).
    from src.loop.sprint_phase import derive_phase      # noqa: PLC0415
    from src.loop.sprint_summary import compute_summary  # noqa: PLC0415

    try:
        from src.loop.sprint_files import list_generated_files as _lgf  # noqa: PLC0415
        _files_n = len(_lgf(sprint_id))
    except Exception:
        _files_n = 0

    _tasks_now = state_store.load().get("tasks", [])
    _cs = compute_summary(_tasks_now, generated_files_count=_files_n)
    _done_n = _cs["done_tasks"]
    _wr_n   = _cs["waiting_review_tasks"]
    _blk_n  = _cs["blocked_tasks"]
    _conf_n = _cs["confidence"]

    final_counts = {
        "total_tasks":          _cs["total_tasks"],
        "done_tasks":           _done_n,
        "blocked_tasks":        _blk_n,
        "in_progress_tasks":    _cs.get("in_progress_tasks", 0),
        "waiting_review_tasks": _wr_n,
    }
    final_phase = derive_phase(final_counts, run_finished=True)

    if final_phase == "complete":
        _emit("sm", "ceremony", f"Sprint Review — {_done_n} done, {_blk_n} blocked")
        _emit("system", "status", f"Sprint complete — confidence {_conf_n}%")
    elif final_phase == "blocked":
        _emit("system", "status",
              f"Sprint blocked — 0 stories accepted, {_blk_n} blocked — confidence {_conf_n}%")
    else:  # waiting_sprint_review (partial increment or gate-pass tasks in waiting_review)
        _emit("sm", "ceremony",
              f"Sprint Review required — {_done_n} done, {_wr_n} waiting review, "
              f"{_blk_n} blocked (Partial Increment)")
        _emit("system", "status",
              f"Sprint review required — partial increment ({_done_n} done, "
              f"{_wr_n} waiting review, {_blk_n} blocked) — confidence {_conf_n}%")

    compliance = validate_sprint_compliance(state_store)
    _emit(
        "sm",
        "gate",
        f"C3_AGILE (advisory, non-blocking): "
        f"{'PASS' if compliance.get('gate_pass') else 'FAIL'} — "
        f"{str(compliance.get('evidence', ''))[:120]}",
    )

    return _persist_and_build_result(
        sprint_id, goal, completed_stories, blocked_stories,
        step_count, conf_scores, memory_store,
        agile_compliance=compliance,
    )
