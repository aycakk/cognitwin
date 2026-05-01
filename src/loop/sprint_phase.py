"""loop/sprint_phase.py — Sprint phase + increment status derivation.

Pure module (no I/O, no LLM, no state mutations) used by sprint_loop and the
cockpit API to decide:

  - phase            (idle | planning | running | waiting_sprint_review | …)
  - increment_status (none | partial | releasable)
  - product_status   (not_started | partial_increment | potentially_releasable)

A sprint is ONLY 'complete' when every committed task is done. Any blocked,
in-progress, or waiting-review task forces the phase to a review/blocked
variant — never 'complete'.
"""

from __future__ import annotations

from typing import Mapping

# Canonical phase set used across cockpit + sprint_loop.
PHASES: tuple[str, ...] = (
    "idle",
    "planning",
    "waiting_backlog_approval",
    "running",
    "waiting_story_acceptance",
    "waiting_sprint_review",
    "reviewed_partial",
    "complete",
    "blocked",
    "cancelled",
)


def _ints(counts: Mapping[str, int]) -> tuple[int, int, int, int, int]:
    """Extract (total, done, blocked, inprogress, waiting_review) safely."""
    total = int(counts.get("total_tasks") or counts.get("total") or 0)
    done = int(counts.get("done_tasks") or counts.get("done") or 0)
    blocked = int(counts.get("blocked_tasks") or counts.get("blocked") or 0)
    inp = int(counts.get("in_progress_tasks") or counts.get("inprogress") or 0)
    wr = int(counts.get("waiting_review_tasks") or counts.get("waiting_review") or 0)
    return total, done, blocked, inp, wr


def derive_phase(counts: Mapping[str, int], *, run_finished: bool = True) -> str:
    """Decide the sprint phase from task counts.

    Parameters
    ----------
    counts
        Dict with at least total_tasks/done_tasks/blocked_tasks/in_progress_tasks/
        waiting_review_tasks (or short aliases).
    run_finished
        True when the sprint loop has finished all execution work. While the
        loop is still running, return "running" instead of jumping to
        "waiting_sprint_review" prematurely.
    """
    total, done, blocked, inp, wr = _ints(counts)

    if total == 0:
        return "idle" if run_finished else "planning"

    if not run_finished:
        return "running"

    # Run finished — decide the terminal phase.
    if done == total and blocked == 0 and inp == 0 and wr == 0:
        return "complete"
    if done == 0 and blocked > 0:
        return "blocked"
    # Any mix of done + (blocked|waiting_review|inprogress) — needs PO review.
    if blocked > 0 or wr > 0 or inp > 0:
        return "waiting_sprint_review"
    return "complete"


def derive_increment_status(counts: Mapping[str, int]) -> str:
    """Map task counts to none | partial | releasable."""
    total, done, blocked, inp, wr = _ints(counts)
    if total == 0 or done == 0:
        return "none"
    if done == total and blocked == 0 and inp == 0 and wr == 0:
        return "releasable"
    return "partial"


def derive_product_status(counts: Mapping[str, int]) -> str:
    """Map task counts to not_started | partial_increment | potentially_releasable."""
    inc = derive_increment_status(counts)
    return {
        "none": "not_started",
        "partial": "partial_increment",
        "releasable": "potentially_releasable",
    }[inc]
