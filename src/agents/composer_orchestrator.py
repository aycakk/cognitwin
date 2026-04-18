"""agents/composer_orchestrator.py — ComposerOrchestrator: sprint-level meta-agent.

This is a CONTROL-FLOW agent, not a content-generation agent.
It does NOT call any LLM.  All decisions are deterministic rule-based logic
so they are fast, auditable, and safe for MVP Phase 1.

Responsibilities:
  - analyze(goal)           → classify goal type, build planning context
  - reroute(story_id, ...)  → decide RETRY / ESCALATE / SKIP after gate failure
  - synthesize_state()      → snapshot of current sprint state for context injection

The existing ComposerAgent (output merger) is NOT replaced.
ComposerOrchestrator is a new class with a different concern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.pipeline.scrum_team.sprint_state_store import SprintStateStore

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Safety budgets — shared constants used by sprint_loop.py too
# ─────────────────────────────────────────────────────────────────────────────

MAX_STEPS_PER_SPRINT: int = 20
MAX_REROUTE_PER_TASK: int = 3


# ─────────────────────────────────────────────────────────────────────────────
#  Reroute types
# ─────────────────────────────────────────────────────────────────────────────

class RerouteAction(str, Enum):
    RETRY    = "retry"     # re-run the same agent with revision hint
    ESCALATE = "escalate"  # mark task blocked, require human review
    SKIP     = "skip"      # accept imperfect output and move on


@dataclass
class RerouteDecision:
    action: RerouteAction
    reason: str


# ─────────────────────────────────────────────────────────────────────────────
#  Goal classification — keyword sets
# ─────────────────────────────────────────────────────────────────────────────

_BUGFIX_KEYWORDS       = frozenset({"fix", "bug", "error", "crash", "broken", "hata", "düzelt", "düzeltme"})
_REFACTOR_KEYWORDS     = frozenset({"refactor", "clean", "cleanup", "improve", "optimize", "iyileştir", "yeniden"})
_TESTING_KEYWORDS      = frozenset({"test", "spec", "coverage", "doğrula", "assertion"})
_DOCUMENTATION_KEYWORDS = frozenset({"doc", "docs", "readme", "comment", "dokümantasyon", "açıklama"})


class ComposerOrchestrator:
    """
    Meta-agent that owns sprint-level control flow.

    Instantiated once per sprint run by sprint_loop.run_sprint().
    Holds a reference to SprintStateStore for context reads only —
    it never writes to state directly.
    """

    def __init__(self, state_store: SprintStateStore | None = None) -> None:
        self._store = state_store or SprintStateStore()

    # ─────────────────────────────────────────────────────────────────────────
    #  analyze
    # ─────────────────────────────────────────────────────────────────────────

    def analyze(self, goal: str, history_context: str = "") -> dict[str, Any]:
        """
        Classify the goal type and build a context dict for the planning phase.

        Goal types: feature | bugfix | refactor | testing | documentation
        This is deterministic keyword matching — no LLM for MVP.
        """
        goal_lower = goal.lower()
        tokens     = set(goal_lower.split())

        if tokens & _BUGFIX_KEYWORDS:
            goal_type = "bugfix"
        elif tokens & _REFACTOR_KEYWORDS:
            goal_type = "refactor"
        elif tokens & _TESTING_KEYWORDS:
            goal_type = "testing"
        elif tokens & _DOCUMENTATION_KEYWORDS:
            goal_type = "documentation"
        else:
            goal_type = "feature"

        context_parts: list[str] = [f"Goal type: {goal_type}."]
        if history_context:
            context_parts.append(history_context)

        try:
            sprint_goal = self._store.get_sprint_goal()
            if sprint_goal and "tanımlanmamış" not in sprint_goal:
                context_parts.append(f"Current sprint goal: {sprint_goal}")
        except Exception:
            pass

        logger.debug("orchestrator.analyze: goal_type=%s goal=%r", goal_type, goal[:50])
        return {
            "goal":              goal,
            "goal_type":         goal_type,
            "context":           " ".join(context_parts),
            "history_available": bool(history_context),
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  reroute
    # ─────────────────────────────────────────────────────────────────────────

    def reroute(
        self,
        story_id: str,
        gate_report: dict[str, Any],
        reroute_count: int,
    ) -> RerouteDecision:
        """
        Decide what to do when gate evaluation fails for a story's task.

        Decision rules (deterministic for MVP):

        1. Budget exhausted                → ESCALATE
        2. C1 (PII detected)              → ESCALATE immediately (hard safety stop)
        3. C4 / C6 (content quality)      → RETRY (these are LLM-fixable)
        4. Other gates, budget > 0        → RETRY
        5. Other gates, budget almost out → ESCALATE

        SKIP is used only when the orchestrator decides the partial output is
        acceptable (e.g. no failing gate detected in the report — shouldn't
        happen normally, but guard defensively).
        """
        if reroute_count >= MAX_REROUTE_PER_TASK:
            return RerouteDecision(
                action=RerouteAction.ESCALATE,
                reason=(
                    f"Reroute budget exhausted ({reroute_count}/{MAX_REROUTE_PER_TASK}) "
                    f"for story {story_id}"
                ),
            )

        gates         = gate_report.get("gates", {})
        failing_gates = [gid for gid, info in gates.items() if not info.get("pass", True)]

        if not failing_gates:
            # No failing gate found in report — treat as acceptable (defensive skip)
            logger.warning("orchestrator.reroute: called with no failing gates for %s", story_id)
            return RerouteDecision(
                action=RerouteAction.SKIP,
                reason="No failing gates detected in report — accepting output",
            )

        first_fail = failing_gates[0]

        # C1 is a hard safety gate — PII must never be retried, needs human review
        if first_fail == "C1":
            return RerouteDecision(
                action=RerouteAction.ESCALATE,
                reason=(
                    f"PII detected in output for story {story_id} — "
                    "escalating for human review (C1 is a hard stop)"
                ),
            )

        # Content-quality gates (C4 hallucination, C6 anti-sycophancy) are retryable
        if first_fail in ("C4", "C6"):
            return RerouteDecision(
                action=RerouteAction.RETRY,
                reason=(
                    f"Gate {first_fail} failed — retrying with revision hint "
                    f"(attempt {reroute_count + 1}/{MAX_REROUTE_PER_TASK})"
                ),
            )

        # All other gates: retry if budget allows one more attempt, else escalate
        remaining = MAX_REROUTE_PER_TASK - reroute_count - 1
        if remaining > 0:
            return RerouteDecision(
                action=RerouteAction.RETRY,
                reason=(
                    f"Gate {first_fail} failed — retrying "
                    f"(attempt {reroute_count + 1}/{MAX_REROUTE_PER_TASK}, "
                    f"{remaining} remaining)"
                ),
            )

        return RerouteDecision(
            action=RerouteAction.ESCALATE,
            reason=(
                f"Gate {first_fail} failed and retry budget exhausted "
                f"for story {story_id}"
            ),
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  synthesize_state
    # ─────────────────────────────────────────────────────────────────────────

    def synthesize_state(self) -> dict[str, Any]:
        """Return a lightweight snapshot of current sprint state.

        Used for context injection into LLM prompts when needed.
        Returns empty dict on any error — never raises.
        """
        try:
            goal        = self._store.get_sprint_goal()
            assignments = self._store.get_assignments()
            blocked     = self._store.get_blocked_tasks()
            return {
                "sprint_goal":   goal,
                "active_tasks":  len(assignments),
                "blocked_tasks": len(blocked),
            }
        except Exception as exc:
            logger.warning("orchestrator.synthesize_state failed: %s", exc)
            return {}
