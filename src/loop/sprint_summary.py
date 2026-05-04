"""loop/sprint_summary.py — Sprint-level summary + confidence calculator.

Pure module. Given the live task list (board API shape) and a few side
signals, produce the summary block that the cockpit renders:

  total_tasks, done_tasks, blocked_tasks, in_progress_tasks,
  waiting_review_tasks, generated_files_count,
  increment_status, product_status, confidence (0..100).

Confidence formula (per product spec):

  base = round((done / total) * 100)

  Penalties (cap, applied LAST, after the base is computed):
    blocked_tasks > 0          → cap at 70
    generated_files_count == 0 → cap at 50
    no_accepted_tasks          → cap at 40
    done_tasks == 0            → cap at 20

Confidence is NEVER 100 when any blocked task exists, by construction.
"""

from __future__ import annotations

from typing import Iterable

from src.loop.sprint_phase import derive_increment_status, derive_product_status


_DONE_STATUSES = {"done", "completed", "complete", "accepted", "finished"}
_BLOCKED_STATUSES = {"blocked", "failed", "rejected", "escalated"}
_INPROG_STATUSES = {"in_progress", "inprogress", "running", "active", "wip"}
_WAITING_REVIEW = {"waiting_review", "ready_for_review"}


def _bucket(task: dict) -> str:
    """Return one of: done | blocked | inprogress | waiting_review | todo."""
    status = str(task.get("status") or "").strip().lower().replace("-", "_").replace(" ", "_")
    po = str(task.get("po_status") or "").strip().lower()
    # Waiting Review takes priority over a "done" status — this represents the
    # "developer is finished, PO has not yet accepted" state.
    if po == "ready_for_review" and status in _DONE_STATUSES:
        return "waiting_review"
    _ACCEPTED_PO = {"accepted", "agent_accepted", "human_accepted"}
    if status in _DONE_STATUSES and po in _ACCEPTED_PO:
        return "done"
    if status in _DONE_STATUSES and not po:
        # Done with no PO status — legacy behaviour treats as done.
        return "done"
    if status in _BLOCKED_STATUSES:
        return "blocked"
    if status in _INPROG_STATUSES:
        return "inprogress"
    if status in _WAITING_REVIEW:
        return "waiting_review"
    return "todo"


def _count_accepted(tasks: Iterable[dict]) -> int:
    """Count tasks that are both done AND accepted (by agent or human)."""
    _ACCEPTED_PO = {"accepted", "agent_accepted", "human_accepted"}
    n = 0
    for t in tasks:
        po = str(t.get("po_status") or "").strip().lower()
        if _bucket(t) == "done" and po in _ACCEPTED_PO:
            n += 1
    return n


def compute_confidence(
    *,
    total: int,
    done: int,
    blocked: int,
    accepted: int,
    generated_files_count: int,
) -> int:
    """Return 0..100 confidence with penalties applied.

    See module docstring for the rules.
    """
    if total <= 0:
        return 0
    base = round((done / total) * 100)
    score = max(0, min(100, base))

    # Caps — applied last so they always win over the base.
    if done == 0:
        score = min(score, 20)
    if accepted == 0:
        score = min(score, 40)
    if generated_files_count == 0:
        score = min(score, 50)
    if blocked > 0:
        score = min(score, 70)

    return int(score)


def compute_summary(
    tasks: list[dict],
    generated_files_count: int = 0,
    *,
    accepted_count: int | None = None,
) -> dict:
    """Produce the cockpit summary block from a list of board tasks.

    Each task dict should expose at least `status` and (optionally) `po_status`.
    """
    total = len(tasks)
    done = sum(1 for t in tasks if _bucket(t) == "done")
    blocked = sum(1 for t in tasks if _bucket(t) == "blocked")
    inprog = sum(1 for t in tasks if _bucket(t) == "inprogress")
    waiting = sum(1 for t in tasks if _bucket(t) == "waiting_review")

    if accepted_count is None:
        accepted = _count_accepted(tasks)
    else:
        accepted = max(0, int(accepted_count))

    counts = {
        "total_tasks": total,
        "done_tasks": done,
        "blocked_tasks": blocked,
        "in_progress_tasks": inprog,
        "waiting_review_tasks": waiting,
    }

    confidence = compute_confidence(
        total=total,
        done=done,
        blocked=blocked,
        accepted=accepted,
        generated_files_count=int(generated_files_count or 0),
    )

    return {
        **counts,
        "generated_files_count": int(generated_files_count or 0),
        "increment_status": derive_increment_status(counts),
        "product_status": derive_product_status(counts),
        "confidence": confidence,
    }
