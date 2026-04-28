"""services/api/cockpit_routes.py — Autonomous Scrum Cockpit API routes.

Provides the real-time polling API consumed by infra/portal/cockpit.html.

Routes
------
POST /api/sprint/start               — start a new sprint run in background
GET  /api/sprint/{sprint_id}/status  — run status from registry
GET  /api/sprint/{sprint_id}/board   — live board (tasks + summary)
GET  /api/sprint/{sprint_id}/events  — event stream list
GET  /api/sprint/{sprint_id}/files   — generated files list with content
GET  /api/sprint/{sprint_id}/preview — best previewable HTML artefact

Security
--------
- sprint_id is validated: no path separators, max 80 chars
- files / preview only read from runtime/sprint_runs/{sprint_id}/generated_files/
  (enforced inside sprint_files.py helpers)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from src.services.api import sprint_run_registry as registry
from src.loop.sprint_files import list_generated_files, get_preview_content

logger = logging.getLogger(__name__)

cockpit_router = APIRouter()

# Shared thread pool — keeps sprint runs off the main asyncio event loop
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="cockpit-sprint")

# Valid sprint_id characters: alphanumeric, hyphen, underscore only
_SPRINT_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,80}$")


# ─────────────────────────────────────────────────────────────────────────────
#  Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class SprintStartRequest(BaseModel):
    goal: str


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_sprint_id(sprint_id: str) -> None:
    if not _SPRINT_ID_RE.match(sprint_id):
        raise HTTPException(status_code=400, detail="Invalid sprint_id.")


def _task_gate_status(task: dict) -> str:
    """Derive gate_status for board display from task fields.

    Pass: done + PO-accepted + ac_validated (or no acceptance criteria)
    Fail: blocked, or PO-rejected, or has AC but ac_validated=False
    Pending: anything else
    """
    status   = task.get("status", "")
    po       = task.get("po_status", "")
    ac       = task.get("acceptance_criteria") or []
    ac_valid = bool(task.get("ac_validated"))

    if status == "blocked" or po == "rejected":
        return "fail"
    if status == "done":
        # Done requires: PO accepted + AC validated (or no AC)
        if po in ("accepted", "human_accepted") and (ac_valid or not ac):
            return "pass"
        return "fail"
    return "pending"


_STATUS_MAP = {
    "todo": "todo", "to_do": "todo", "pending": "todo", "planned": "todo",
    "ready": "todo", "draft": "todo", "new": "todo", "open": "todo",
    "in_progress": "inprogress", "inprogress": "inprogress",
    "running": "inprogress", "active": "inprogress", "wip": "inprogress",
    "done": "done", "completed": "done", "complete": "done",
    "accepted": "done", "finished": "done",
    "blocked": "blocked", "failed": "blocked",
    "rejected": "blocked", "escalated": "blocked",
}


def _normalize_status(raw: str) -> str:
    """Normalize sprint_state task status to board API format."""
    key = (raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _STATUS_MAP.get(key, "todo")


def _build_board_response(sprint_id: str, entry) -> dict:
    """Build the board API response from SprintStateStore + registry."""
    from src.pipeline.scrum_team.sprint_state_store import SprintStateStore  # noqa: PLC0415
    state = SprintStateStore().load()

    # Build a story map for estimate lookup
    story_map = {s.get("story_id"): s for s in state.get("backlog", [])}

    tasks_out = []
    for t in state.get("tasks", []):
        raw_status = t.get("status", "todo")
        norm_status = _normalize_status(raw_status)
        ac = t.get("acceptance_criteria") or []

        # Find estimate from source story
        source_sid = t.get("source_story_id") or ""
        story = story_map.get(source_sid, {})
        estimate = story.get("estimate") or story.get("story_points") or 3

        tasks_out.append({
            "id":                   t.get("id", "?"),
            "title":                t.get("title", "Untitled"),
            "status":               norm_status,
            "priority":             (t.get("priority") or "medium").lower(),
            "estimate":             estimate,
            "acceptance_criteria":  ac,
            "agent":                t.get("assigned_to", "developer"),
            "gate_status":          _task_gate_status(t),
        })

    summary = {
        "todo":       sum(1 for t in tasks_out if t["status"] == "todo"),
        "inprogress": sum(1 for t in tasks_out if t["status"] == "inprogress"),
        "done":       sum(1 for t in tasks_out if t["status"] == "done"),
        "blocked":    sum(1 for t in tasks_out if t["status"] == "blocked"),
        "confidence": 0,
    }

    phase = (entry.phase if entry else "planning")

    return {
        "sprint_id":    sprint_id,
        "goal":         entry.goal if entry else "",
        "phase":        phase,
        "product_goal": state.get("product_goal") or "",
        "sprint_goal":  state.get("sprint", {}).get("goal", ""),
        "tasks":        tasks_out,
        "summary":      summary,
    }


def _run_sprint_background(sprint_id: str, goal: str) -> None:
    """Execute run_sprint in background thread with event_callback wired to registry."""
    registry.set_status(sprint_id, "running", "planning")

    def _cb(agent: str, etype: str, message: str, task_id=None) -> None:
        registry.append_event(sprint_id, agent, etype, message, task_id)

    try:
        from src.services.api.sprint_bridge import _clean_goal_for_po  # noqa: PLC0415
        from src.loop.sprint_loop import run_sprint  # noqa: PLC0415
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore  # noqa: PLC0415

        clean_goal = _clean_goal_for_po(goal)
        store = SprintStateStore()
        store.reset_for_isolated_sprint()

        result = run_sprint(clean_goal, sprint_id=sprint_id, event_callback=_cb)

        # Authoritative completion check: count actually-done tasks in the board.
        try:
            done_tasks = sum(
                1 for t in store.load().get("tasks", [])
                if _normalize_status(t.get("status", "")) == "done"
            )
        except Exception:
            done_tasks = len(result.completed_stories)

        if done_tasks > 0:
            final_status, final_phase = "complete", "complete"
        elif result.blocked_stories:
            final_status, final_phase = "blocked", "blocked"
        else:
            final_status, final_phase = "blocked", "blocked"
            registry.append_event(
                sprint_id, "system", "status",
                "Sprint ended without accepted tasks — needs refinement.",
            )
        registry.set_status(sprint_id, final_status, final_phase)

        logger.info(
            "cockpit: sprint=%s finished  status=%s  completed=%d  blocked=%d",
            sprint_id, final_status,
            len(result.completed_stories), len(result.blocked_stories),
        )

    except Exception as exc:
        logger.error("cockpit: sprint=%s background run failed: %s", sprint_id, exc, exc_info=True)
        registry.append_event(sprint_id, "system", "status", f"Sprint error: {exc}")
        registry.set_status(sprint_id, "error", "blocked")


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@cockpit_router.post("/api/sprint/start")
async def start_sprint(req: SprintStartRequest):
    """Start a sprint run asynchronously and return sprint_id immediately."""
    goal = (req.goal or "").strip()
    if not goal:
        raise HTTPException(status_code=400, detail="goal must not be empty.")

    sprint_id = f"sprint-cockpit-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    registry.register(sprint_id, goal)
    registry.append_event(sprint_id, "system", "status", f"Sprint started — goal: {goal[:120]}")

    # Submit to background thread — returns immediately
    loop = asyncio.get_running_loop()
    loop.run_in_executor(_executor, _run_sprint_background, sprint_id, goal)

    logger.info("cockpit: started sprint=%s goal=%r", sprint_id, goal[:60])
    return {"sprint_id": sprint_id, "goal": goal, "status": "started"}


@cockpit_router.get("/api/sprint/{sprint_id}/status")
async def sprint_status(sprint_id: str):
    _validate_sprint_id(sprint_id)
    entry = registry.get(sprint_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Sprint not found.")
    return entry.as_dict()


@cockpit_router.get("/api/sprint/{sprint_id}/board")
async def sprint_board(sprint_id: str):
    _validate_sprint_id(sprint_id)
    entry = registry.get(sprint_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Sprint not found.")

    loop = asyncio.get_running_loop()
    board = await loop.run_in_executor(None, _build_board_response, sprint_id, entry)
    return board


@cockpit_router.get("/api/sprint/{sprint_id}/events")
async def sprint_events(sprint_id: str):
    _validate_sprint_id(sprint_id)
    entry = registry.get(sprint_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Sprint not found.")
    return {"events": registry.get_events(sprint_id)}


@cockpit_router.get("/api/sprint/{sprint_id}/files")
async def sprint_files(sprint_id: str):
    _validate_sprint_id(sprint_id)
    entry = registry.get(sprint_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Sprint not found.")

    loop = asyncio.get_running_loop()
    files = await loop.run_in_executor(None, list_generated_files, sprint_id)
    return {"files": files}


@cockpit_router.get("/api/sprint/{sprint_id}/preview")
async def sprint_preview(sprint_id: str):
    _validate_sprint_id(sprint_id)
    entry = registry.get(sprint_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Sprint not found.")

    loop = asyncio.get_running_loop()
    html = await loop.run_in_executor(None, get_preview_content, sprint_id)

    if html is None:
        placeholder = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><style>
body{background:#0f1117;color:#64748b;font-family:monospace;display:flex;
align-items:center;justify-content:center;height:100vh;margin:0;font-size:13px;}
</style></head><body>Developer is still working on the artefact…</body></html>"""
        return HTMLResponse(content=placeholder, status_code=200)

    return HTMLResponse(content=html, status_code=200)
