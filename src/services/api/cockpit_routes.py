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
import os
import re
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from src.services.api import sprint_run_registry as registry
from src.loop.sprint_files import list_generated_files, get_preview_content
from src.loop import sprint_index

logger = logging.getLogger(__name__)

cockpit_router = APIRouter()

# When True, the /api/sprint/{id}/review endpoint (task-level human approval) is active.
# Default: False — PO Agent handles story acceptance; humans only act at Sprint Review.
_HUMAN_TASK_REVIEW_MODE: bool = (
    os.getenv("HUMAN_TASK_REVIEW_MODE", "").strip().lower() in ("true", "1", "yes")
)

# Shared thread pool — keeps sprint runs off the main asyncio event loop
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="cockpit-sprint")

# Valid sprint_id characters: alphanumeric, hyphen, underscore only
_SPRINT_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,80}$")


# ─────────────────────────────────────────────────────────────────────────────
#  Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class SprintStartRequest(BaseModel):
    goal: str = ""
    workspace: str = ""
    user_id: str = ""
    user_name: str = ""
    project_id: str = ""
    selected_backlog_item_ids: list[str] = []


class TaskReviewRequest(BaseModel):
    task_id: str
    decision: str  # "approve" | "request_changes" | "reject"
    feedback: str = ""
    reviewer_name: str = ""


class SprintReviewRequest(BaseModel):
    decision: str  # "close_partial" | "recovery_sprint" | "reopen_blocked"
    feedback: str = ""
    reviewer_name: str = ""


class HumanReviewRequest(BaseModel):
    text: str
    reviewer_name: str = ""


class RetroRequest(BaseModel):
    went_well:    list[str] = []
    improve:      list[str] = []
    actions:      list[str] = []
    reviewer_name: str = ""


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_sprint_id(sprint_id: str) -> None:
    if not _SPRINT_ID_RE.match(sprint_id):
        raise HTTPException(status_code=400, detail="Invalid sprint_id.")


def _parse_user_header(request: Request) -> dict:
    """Parse X-Cognitwin-User header.

    Frontend sends encodeURIComponent(JSON.stringify(user)) so that
    non-ISO-8859-1 characters (e.g. Turkish names) survive HTTP headers.
    Falls back to anonymous if decoding fails.
    """
    import json as _json
    from urllib.parse import unquote as _unquote

    raw = request.headers.get("X-Cognitwin-User", "")
    _FALLBACK = {"user_name": "Human Supervisor", "user_role": "Supervisor",
                 "user_email": "", "user_id": ""}
    if not raw:
        return _FALLBACK
    try:
        obj = _json.loads(_unquote(raw))
        return {
            "user_name":  str(obj.get("name",  "") or "Human Supervisor"),
            "user_role":  str(obj.get("role",  "") or "Supervisor"),
            "user_email": str(obj.get("email", "")),
            "user_id":    str(obj.get("id",    "")),
        }
    except Exception:
        return _FALLBACK


def _is_admin(user: dict) -> bool:
    return (user.get("user_role") or "").strip().lower() in ("administrator", "admin")


def _sprint_owner(sprint_id: str) -> dict:
    """Return {owner_user_id, owner_user_email} from sprint_index, or {} if not found."""
    entry = sprint_index.get_entry(sprint_id)
    if entry:
        return {
            "owner_user_id":    entry.get("owner_user_id") or "",
            "owner_user_email": entry.get("owner_user_email") or "",
        }
    return {}


def _get_or_hydrate_entry(sprint_id: str):
    """Return registry entry for sprint_id, re-hydrating from sprint_index if needed.

    The registry is in-memory and is lost on server restart.  Sprint metadata
    lives permanently in sprint_index.json, so we can reconstruct a minimal
    entry for sprints that pre-date the current process.  This allows the board,
    events, and recovery endpoints to keep working after a restart.
    """
    entry = registry.get(sprint_id)
    if entry is not None:
        return entry

    idx = sprint_index.get_entry(sprint_id)
    if idx is None:
        return None

    # Re-hydrate: register a stub so subsequent calls find it in memory
    entry = registry.register(sprint_id, idx.get("goal", ""))
    # Restore the terminal phase / status from the index
    phase  = idx.get("phase", "complete")
    status = "complete" if phase in ("complete", "reviewed_partial") else phase
    registry.set_status(sprint_id, status, phase)
    logger.info("cockpit_routes: re-hydrated registry entry for sprint=%s phase=%s", sprint_id, phase)
    return entry


def _refresh_project_counts(project_id: str | None) -> None:
    """Recompute project-level sprint counters. No-op if no project_id."""
    if not project_id:
        return
    try:
        from src.services.api import project_index  # local import to avoid cycle
        project_index.recompute_counts(project_id)
    except Exception as exc:
        logger.warning("project_index.recompute_counts failed for %s: %s", project_id, exc)


def _check_sprint_access(user: dict, sprint_id: str) -> None:
    """Raise HTTP 403 if the requesting user cannot access this sprint.

    Rules:
    - Admin role → full access to all sprints.
    - Owner (owner_user_id == user_id) → access to own sprint.
    - Legacy sprint (owner_user_id missing) → Admin only.
    - Everyone else → 403.
    """
    if _is_admin(user):
        return
    owner = _sprint_owner(sprint_id)
    owner_id = owner.get("owner_user_id", "")
    if not owner_id:
        raise HTTPException(
            status_code=403,
            detail="Access denied: this sprint has no owner record and is visible to Admin only.",
        )
    if owner_id != (user.get("user_id") or ""):
        raise HTTPException(
            status_code=403,
            detail="Access denied: you are not the owner of this sprint.",
        )


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
        if po in ("accepted", "agent_accepted", "human_accepted") and (ac_valid or not ac):
            return "pass"
        return "fail"
    return "pending"


_STATUS_MAP = {
    "todo": "todo", "to_do": "todo", "pending": "todo", "planned": "todo",
    "ready": "todo", "draft": "todo", "new": "todo", "open": "todo",
    "reopened": "todo",
    "in_progress": "inprogress", "inprogress": "inprogress",
    "running": "inprogress", "active": "inprogress", "wip": "inprogress",
    "done": "done", "completed": "done", "complete": "done",
    "accepted": "done", "finished": "done",
    "blocked": "blocked", "failed": "blocked",
    "rejected": "blocked", "escalated": "blocked",
    "waiting_review": "waiting_review",
}


def _normalize_status(raw: str) -> str:
    """Normalize sprint_state task status to board API format."""
    key = (raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _STATUS_MAP.get(key, "todo")


def _board_status_for_task(task: dict) -> str:
    """Bucket a sprint task into a Kanban column.

    Adds the "waiting_review" column on top of the raw status: a task whose
    Developer is finished but whose PO acceptance is still pending lives in
    Waiting Review, NOT Done. Mapping rules:

      po_status == "ready_for_review"  → waiting_review (regardless of status)
      po_status == "change_requested"  → todo  (Developer must redo)
      po_status == "human_rejected"    → blocked
      otherwise                         → _normalize_status(status)
    """
    po = str(task.get("po_status") or "").strip().lower()
    status = str(task.get("status") or "").strip().lower()
    if po == "ready_for_review":
        return "waiting_review"
    if po == "change_requested":
        return "todo"
    if po == "human_rejected":
        return "blocked"
    return _normalize_status(status)


def _build_board_response(sprint_id: str, entry) -> dict:
    """Build the board API response from SprintStateStore + registry."""
    from src.pipeline.scrum_team.sprint_state_store import SprintStateStore  # noqa: PLC0415
    from src.loop.sprint_summary import compute_summary  # noqa: PLC0415
    from src.agents.composer_orchestrator import MAX_REROUTE_PER_TASK  # noqa: PLC0415
    state = SprintStateStore.for_sprint(sprint_id).load()

    # Build a story map for estimate + source lookup
    story_map = {s.get("story_id"): s for s in state.get("backlog", [])}

    # Count generated artefacts on disk (used by confidence cap rules).
    try:
        generated_files_count = len(list_generated_files(sprint_id))
    except Exception:
        generated_files_count = 0

    tasks_out = []
    for t in state.get("tasks", []):
        norm_status = _board_status_for_task(t)
        ac = t.get("acceptance_criteria") or []

        # Find estimate + scope source from source story
        source_sid = t.get("source_story_id") or ""
        story = story_map.get(source_sid, {})
        estimate = story.get("estimate") or story.get("story_points") or 3
        story_source = story.get("source") or "inferred"

        tasks_out.append({
            "id":                   t.get("id", "?"),
            "title":                t.get("title", "Untitled"),
            "status":               norm_status,
            "po_status":            t.get("po_status") or "",
            "po_agent_review":      t.get("po_agent_review") or None,
            "priority":             (t.get("priority") or "medium").lower(),
            "estimate":             estimate,
            "acceptance_criteria":  ac,
            "agent":                t.get("assigned_to", "developer"),
            "gate_status":          _task_gate_status(t),
            "gate_detail":          t.get("last_gate_failures") or [],
            "retry_count":          int(t.get("retry_count") or 0),
            "retry_max":            MAX_REROUTE_PER_TASK,
            "source":               story_source,
            "blocker":              t.get("blocker") or "",
        })

    # Use the new summary helper (confidence is no longer hardcoded to 0).
    # We pass the raw state tasks so accepted-count detection can read po_status.
    summary_block = compute_summary(state.get("tasks", []), generated_files_count=generated_files_count)
    summary = {
        # Keep the legacy column counters so the existing JS keeps working.
        "todo":          sum(1 for t in tasks_out if t["status"] == "todo"),
        "inprogress":    sum(1 for t in tasks_out if t["status"] == "inprogress"),
        "waiting_review": sum(1 for t in tasks_out if t["status"] == "waiting_review"),
        "done":          sum(1 for t in tasks_out if t["status"] == "done"),
        "blocked":       sum(1 for t in tasks_out if t["status"] == "blocked"),
        # New: real confidence + increment status the spec requires.
        **summary_block,
    }

    phase = (entry.phase if entry else state.get("phase") or "planning")

    return {
        "sprint_id":    sprint_id,
        "goal":         entry.goal if entry else "",
        "phase":        phase,
        "product_goal": state.get("product_goal") or "",
        "sprint_goal":  state.get("sprint", {}).get("goal", ""),
        "tasks":        tasks_out,
        "summary":      summary,
    }


def _seed_selected_backlog(store, sprint_id: str, project_id: str,
                           selected_item_ids: list[str]) -> int:
    """Pre-populate sprint state backlog with project-level items the user
    explicitly chose at planning time. Stories are tagged with target_sprint
    so run_sprint() prefers them over LLM-generated stories.

    Returns the number of stories successfully seeded.
    """
    if not (project_id and selected_item_ids):
        return 0
    try:
        from src.services.api import product_backlog_store as _backlog  # noqa: PLC0415
    except Exception:
        return 0
    items = _backlog.items_by_ids(project_id, selected_item_ids)
    seeded = 0
    for it in items:
        try:
            store.add_story(
                title              = it.get("title", "Untitled"),
                description        = it.get("description", ""),
                priority           = it.get("priority", "medium"),
                acceptance_criteria= it.get("acceptance_criteria") or [],
                epic               = it.get("epic", ""),
                story_points       = it.get("story_points", 0) or 0,
                target_sprint      = sprint_id,
                source             = "product_backlog",
            )
            seeded += 1
        except Exception as exc:
            logger.warning("seed selected backlog item %s failed: %s", it.get("item_id"), exc)
    if seeded:
        try:
            _backlog.mark_in_sprint(project_id, [it["item_id"] for it in items], sprint_id)
        except Exception as exc:
            logger.warning("mark_in_sprint failed for %s: %s", project_id, exc)
    return seeded


def _run_sprint_background(sprint_id: str, goal: str,
                           project_id: str = "",
                           selected_item_ids: list[str] | None = None) -> None:
    """Execute run_sprint in background thread with event_callback wired to registry."""
    registry.set_status(sprint_id, "running", "planning")

    def _cb(agent: str, etype: str, message: str, task_id=None) -> None:
        registry.append_event(sprint_id, agent, etype, message, task_id)

    try:
        from src.services.api.sprint_bridge import _clean_goal_for_po  # noqa: PLC0415
        from src.loop.sprint_loop import run_sprint  # noqa: PLC0415
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore  # noqa: PLC0415

        clean_goal = _clean_goal_for_po(goal)
        store = SprintStateStore.for_sprint(sprint_id)
        store.reset_for_isolated_sprint()
        seeded = _seed_selected_backlog(store, sprint_id, project_id, selected_item_ids or [])
        if seeded:
            registry.append_event(
                sprint_id, "po", "ceremony",
                f"Sprint Planning — {seeded} item(s) selected from Product Backlog",
            )

        result = run_sprint(clean_goal, sprint_id=sprint_id, event_callback=_cb, state_store=store)

        # Authoritative completion check — count buckets on the live board.
        # Phase decision uses sprint_phase.derive_phase so a partial sprint
        # (some done, some blocked) is correctly flagged as
        # "waiting_sprint_review", NOT "complete".
        from src.loop.sprint_phase import derive_phase  # noqa: PLC0415
        from src.loop.sprint_summary import compute_summary  # noqa: PLC0415

        try:
            tasks = store.load().get("tasks", [])
            summary = compute_summary(tasks, generated_files_count=len(list_generated_files(sprint_id)))
        except Exception:
            summary = {
                "total_tasks":          len(result.completed_stories) + len(result.blocked_stories),
                "done_tasks":           len(result.completed_stories),
                "blocked_tasks":        len(result.blocked_stories),
                "in_progress_tasks":    0,
                "waiting_review_tasks": 0,
            }

        final_phase = derive_phase(summary, run_finished=True)
        # status remains a coarse run-status — the cockpit JS uses it to stop
        # polling. Map non-terminal phases like waiting_sprint_review to
        # status="complete" so polling stops; the richer phase carries the
        # actual review-required signal.
        final_status = "blocked" if final_phase == "blocked" else "complete"

        if summary.get("done_tasks", 0) == 0 and summary.get("blocked_tasks", 0) == 0:
            registry.append_event(
                sprint_id, "system", "status",
                "Sprint ended without accepted tasks — needs refinement.",
            )

        registry.set_status(sprint_id, final_status, final_phase)
        try:
            store.set_phase(final_phase)
        except Exception:
            pass

        # Persist final summary to sprint index for dashboard persistence.
        latest_ev = f"Sprint {final_phase} — {summary.get('done_tasks',0)} done, {summary.get('blocked_tasks',0)} blocked"
        sprint_index.update_entry(sprint_id, {
            "phase":                final_phase,
            "increment_status":     summary.get("increment_status", ""),
            "product_status":       summary.get("product_status", ""),
            "confidence":           summary.get("confidence", 0),
            "done_tasks":           summary.get("done_tasks", 0),
            "waiting_review_tasks": summary.get("waiting_review_tasks", 0),
            "blocked_tasks":        summary.get("blocked_tasks", 0),
            "total_tasks":          summary.get("total_tasks", 0),
            "generated_files_count": summary.get("generated_files_count", 0),
            "latest_event":         latest_ev,
            "agile_compliance":     result.agile_compliance,
        })
        _refresh_project_counts((sprint_index.get_entry(sprint_id) or {}).get("project_id"))

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
async def start_sprint(req: SprintStartRequest, request: Request):
    """Start a sprint run asynchronously and return sprint_id immediately."""
    goal = (req.goal or "").strip()
    selected_ids = [str(i).strip() for i in (req.selected_backlog_item_ids or []) if str(i).strip()]
    project_id   = (req.project_id or "").strip()

    user = _parse_user_header(request)
    if not user["user_name"] or user["user_name"] == "anonymous":
        user["user_name"] = req.user_name or "anonymous"
        user["user_id"]   = req.user_id or ""
    workspace = (req.workspace or "").strip()

    if project_id:
        from src.services.api import project_index  # local import to avoid cycle
        project = project_index.get(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found.")
        if not _is_admin(user):
            owner_id = project.get("owner_user_id") or ""
            if not owner_id or owner_id != (user.get("user_id") or ""):
                raise HTTPException(
                    status_code=403,
                    detail="Access denied: you do not own this project.",
                )

    # Selection-driven planning: derive goal from selected backlog items if absent.
    if selected_ids and project_id:
        from src.services.api import product_backlog_store  # noqa: PLC0415
        items = product_backlog_store.items_by_ids(project_id, selected_ids)
        if not items:
            raise HTTPException(
                status_code=422,
                detail="None of the selected backlog items were found.",
            )
        # Reject items already locked into another sprint.
        already_taken = [it["item_id"] for it in items if it.get("status") in ("in_sprint", "done")]
        if already_taken:
            raise HTTPException(
                status_code=409,
                detail=f"Items already in another sprint: {', '.join(already_taken)}",
            )
        if not goal:
            titles = "; ".join(it.get("title", "") for it in items if it.get("title"))
            goal = f"Deliver: {titles}"[:280]
        # Keep selected_ids in the order present in the backlog file.
        selected_ids = [it["item_id"] for it in items]

    if not goal:
        raise HTTPException(
            status_code=422,
            detail="goal or selected_backlog_item_ids must be provided.",
        )

    sprint_id = f"sprint-cockpit-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Flip selected backlog items to in_sprint synchronously so the project
    # dashboard reflects the commitment immediately, regardless of whether
    # the background runner has started yet.
    if selected_ids and project_id:
        try:
            from src.services.api import product_backlog_store as _backlog  # noqa: PLC0415
            _backlog.mark_in_sprint(project_id, selected_ids, sprint_id)
        except Exception as exc:
            logger.warning("mark_in_sprint failed for %s: %s", project_id, exc)

    registry.register(sprint_id, goal)
    registry.append_event(sprint_id, "system", "status", f"Sprint started — goal: {goal[:120]}")

    # Persist to sprint index so the dashboard survives restarts.
    sprint_index.write_entry({
        "sprint_id":        sprint_id,
        "project_id":       project_id,
        "owner_user_id":    user["user_id"],
        "owner_user_name":  user["user_name"],
        "owner_user_email": user["user_email"],
        "owner_user_role":  user["user_role"],
        "workspace":        workspace,
        "goal":             goal,
        "phase":            "planning",
        "increment_status": "none",
        "product_status":   "not_started",
        "confidence":       0,
        "done_tasks":       0,
        "waiting_review_tasks": 0,
        "blocked_tasks":    0,
        "total_tasks":      0,
        "generated_files_count": 0,
        "archived":         False,
        "created_at":       datetime.now().isoformat(),
        "updated_at":       datetime.now().isoformat(),
        "latest_event":     f"Sprint started — goal: {goal[:120]}",
        "selected_backlog_item_ids": selected_ids,
    })

    # Submit to background thread — returns immediately
    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        _executor, _run_sprint_background, sprint_id, goal, project_id, selected_ids,
    )

    logger.info("cockpit: started sprint=%s goal=%r selected=%d",
                sprint_id, goal[:60], len(selected_ids))
    return {
        "sprint_id":                 sprint_id,
        "goal":                      goal,
        "status":                    "started",
        "selected_backlog_item_ids": selected_ids,
    }


@cockpit_router.get("/api/sprint/{sprint_id}/status")
async def sprint_status(sprint_id: str):
    _validate_sprint_id(sprint_id)
    entry = _get_or_hydrate_entry(sprint_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Sprint not found.")
    return entry.as_dict()


@cockpit_router.get("/api/sprint/{sprint_id}/board")
async def sprint_board(sprint_id: str, request: Request):
    _validate_sprint_id(sprint_id)
    entry = _get_or_hydrate_entry(sprint_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Sprint not found.")
    _check_sprint_access(_parse_user_header(request), sprint_id)
    loop = asyncio.get_running_loop()
    board = await loop.run_in_executor(None, _build_board_response, sprint_id, entry)
    return board


@cockpit_router.get("/api/sprints")
async def list_sprints(
    request: Request,
    archived: bool = False,
    project_id: str = "",
    legacy: bool = False,
):
    """Return sprint index records filtered by ownership.

    Admin sees all sprints.
    Non-admin sees only sprints where owner_user_id matches their user_id.
    Legacy sprints with no owner_user_id are visible to Admin only.

    Optional query filters:
      ?project_id=<id>  — restrict to sprints belonging to one project.
      ?legacy=true      — admin-only: return sprints with no project_id.
    """
    user    = _parse_user_header(request)
    entries = sprint_index.read_all()

    if not archived:
        entries = [e for e in entries if not e.get("archived")]

    if legacy:
        if not _is_admin(user):
            raise HTTPException(status_code=403, detail="Legacy view is admin-only.")
        entries = [e for e in entries if not e.get("project_id")]
    elif project_id:
        entries = [e for e in entries if e.get("project_id") == project_id]

    if not _is_admin(user):
        uid = user.get("user_id") or ""
        entries = [
            e for e in entries
            if uid and e.get("owner_user_id") == uid
        ]

    # Merge live registry phase/status for running sprints.
    for e in entries:
        live = registry.get(e.get("sprint_id", ""))
        if live:
            e["phase"]  = live.phase
            e["status"] = live.status

    return {"sprints": entries, "user_is_admin": _is_admin(user)}


@cockpit_router.post("/api/sprint/{sprint_id}/archive")
async def archive_sprint(sprint_id: str, request: Request):
    _validate_sprint_id(sprint_id)
    user = _parse_user_header(request)
    _check_sprint_access(user, sprint_id)
    entry = sprint_index.get_entry(sprint_id) or {}
    ok = sprint_index.archive_sprint(sprint_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Sprint not found in index.")
    registry.append_event(sprint_id, "system", "status",
                           f"Sprint archived by {user['user_name']}")
    _refresh_project_counts(entry.get("project_id"))
    return {"ok": True, "sprint_id": sprint_id}


@cockpit_router.get("/api/sprint/{sprint_id}/events")
async def sprint_events(sprint_id: str):
    _validate_sprint_id(sprint_id)
    entry = _get_or_hydrate_entry(sprint_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Sprint not found.")
    return {"events": registry.get_events(sprint_id)}


@cockpit_router.get("/api/sprint/{sprint_id}/files")
async def sprint_files(sprint_id: str):
    _validate_sprint_id(sprint_id)
    entry = _get_or_hydrate_entry(sprint_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Sprint not found.")

    loop = asyncio.get_running_loop()
    files = await loop.run_in_executor(None, list_generated_files, sprint_id)
    return {"files": files}


@cockpit_router.get("/api/sprint/{sprint_id}/preview")
async def sprint_preview(sprint_id: str):
    _validate_sprint_id(sprint_id)
    entry = _get_or_hydrate_entry(sprint_id)
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


# ─── decision → apply_human_feedback action mapping ─────────────────────────
_REVIEW_DECISION_MAP = {
    "approve":         "accept",
    "request_changes": "change_request",
    "reject":          "reject",
}


@cockpit_router.post("/api/sprint/{sprint_id}/review")
async def sprint_task_review(sprint_id: str, req: TaskReviewRequest, request: Request):
    """Human task-level review — approve / request_changes / reject a single task.

    Only active when HUMAN_TASK_REVIEW_MODE=true (debug/experiment mode).
    In normal operation PO Agent handles story acceptance; human review is
    limited to the Sprint Review stage at sprint end.
    """
    if not _HUMAN_TASK_REVIEW_MODE:
        raise HTTPException(
            status_code=405,
            detail={
                "error": "human_task_review_disabled",
                "message": (
                    "Task-level human review is disabled. "
                    "PO Agent handles story acceptance automatically. "
                    "Enable with HUMAN_TASK_REVIEW_MODE=true for debug mode."
                ),
            },
        )
    _validate_sprint_id(sprint_id)
    entry = _get_or_hydrate_entry(sprint_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Sprint not found.")

    action = _REVIEW_DECISION_MAP.get((req.decision or "").strip().lower())
    if not action:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown decision {req.decision!r}. Valid: approve | request_changes | reject",
        )

    user = _parse_user_header(request)
    _check_sprint_access(user, sprint_id)
    reviewer = user["user_name"] or (req.reviewer_name or "Human Supervisor").strip() or "Human Supervisor"

    def _apply():
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore  # noqa: PLC0415
        store = SprintStateStore.for_sprint(sprint_id)
        result = store.apply_human_feedback(
            task_id = req.task_id,
            action  = action,
            reason  = (req.feedback or "")[:400],
            actor   = reviewer,
        )
        if result.get("ok") and action == "accept":
            # Promote to Done — accept_story + add_to_increment
            task = result.get("task") or {}
            story_id = task.get("source_story_id") or ""
            if story_id:
                try:
                    store.accept_story(story_id)
                    store.add_to_increment(req.task_id)
                except Exception:
                    pass
        return result

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _apply)

    if not result.get("ok"):
        raise HTTPException(status_code=422, detail=result.get("message", "Review failed."))

    ev_msg = (
        f"Human review [{req.decision}] on {req.task_id} by {reviewer}"
        f" ({user.get('user_role','')}) — {(req.feedback or '')[:80]}"
    )
    registry.append_event(sprint_id, "po", "action", ev_msg, task_id=req.task_id)
    sprint_index.update_entry(sprint_id, {"latest_event": ev_msg})
    _refresh_project_counts((sprint_index.get_entry(sprint_id) or {}).get("project_id"))
    return {"ok": True, "task_id": req.task_id, "decision": req.decision}


@cockpit_router.post("/api/sprint/{sprint_id}/rerun-po-review")
async def rerun_po_review(sprint_id: str, request: Request):
    """Retroactively run PO Agent review on all tasks stuck in waiting_review.

    Useful when a sprint was executed with old code that routed tasks to
    waiting_review instead of running the PO Agent automatically.
    Works even after a server restart (falls back to sprint_index when
    the in-memory registry entry has been lost).
    """
    _validate_sprint_id(sprint_id)
    # Registry is in-memory — lost on restart. Fall back to sprint_index so
    # this endpoint still works for sprints that pre-date the current process.
    if not registry.get(sprint_id) and not sprint_index.get_entry(sprint_id):
        raise HTTPException(status_code=404, detail="Sprint not found.")

    user = _parse_user_header(request)
    _check_sprint_access(user, sprint_id)

    def _run():
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore  # noqa: PLC0415
        from src.agents.po_llm_agent import POLLMAgent  # noqa: PLC0415

        store = SprintStateStore.for_sprint(sprint_id)
        po_agent = POLLMAgent()
        state = store.load()
        sprint_goal = (state.get("sprint") or {}).get("goal", "")

        results = []
        for task in state.get("tasks", []):
            if (task.get("po_status") or "") != "ready_for_review":
                continue
            task_id  = task.get("id", "")
            story_id = task.get("source_story_id") or ""
            story    = next((s for s in state.get("backlog", []) if s.get("story_id") == story_id), {})

            review = po_agent.review_story(
                story               = story,
                task_output         = task.get("result", "") or task.get("summary", "") or task.get("title", ""),
                acceptance_criteria = story.get("acceptance_criteria", []),
                sprint_goal         = sprint_goal,
            )

            if review.get("accepted"):
                store.apply_agent_review(
                    task_id          = task_id,
                    accepted         = True,
                    reason           = review.get("reason", ""),
                    confidence       = review.get("confidence", 1.0),
                    missing_criteria = review.get("missing_criteria", []),
                )
                store.add_to_increment(task_id)
                results.append({"task_id": task_id, "decision": "accepted", "confidence": review.get("confidence")})
                registry.append_event(sprint_id, "po", "action",
                    f"PO Agent (recovery) accepted {task_id} — confidence {round((review.get('confidence') or 1)*100)}%",
                    task_id=task_id)
            else:
                store.block_task(task_id, f"PO Agent (recovery) rejected: {review.get('reason','')[:120]}")
                results.append({"task_id": task_id, "decision": "rejected", "reason": review.get("reason", "")})
                registry.append_event(sprint_id, "po", "action",
                    f"PO Agent (recovery) rejected {task_id}: {review.get('reason','')[:80]}",
                    task_id=task_id)

        return results

    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, _run)

    accepted = sum(1 for r in results if r["decision"] == "accepted")
    rejected = sum(1 for r in results if r["decision"] == "rejected")
    msg = f"PO Agent recovery: {accepted} accepted, {rejected} rejected out of {len(results)} waiting_review tasks."
    registry.append_event(sprint_id, "sm", "ceremony", msg)
    sprint_index.update_entry(sprint_id, {"latest_event": msg})
    _refresh_project_counts((sprint_index.get_entry(sprint_id) or {}).get("project_id"))
    return {"ok": True, "processed": len(results), "accepted": accepted, "rejected": rejected, "results": results}


@cockpit_router.post("/api/sprint/{sprint_id}/sprint-review")
async def sprint_review_decision(sprint_id: str, req: SprintReviewRequest, request: Request):
    """Sprint-level review decision: close_partial | recovery_sprint | reopen_blocked."""
    _validate_sprint_id(sprint_id)
    entry = _get_or_hydrate_entry(sprint_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Sprint not found.")

    decision = (req.decision or "").strip().lower()
    user = _parse_user_header(request)
    _check_sprint_access(user, sprint_id)
    reviewer = user["user_name"] or (req.reviewer_name or "Human Supervisor").strip() or "Human Supervisor"

    current_phase = (entry.phase or "").strip().lower()
    _UNLOCKED_PHASES = {"waiting_sprint_review", "reviewed_partial", "blocked"}
    if current_phase not in _UNLOCKED_PHASES:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "phase_locked",
                "message": "Sprint Review decisions unlock after the run ends.",
                "phase": current_phase,
            },
        )

    if decision == "close_partial":
        registry.set_status(sprint_id, "complete", "reviewed_partial")
        ev_msg = f"Sprint closed as Partial Increment by {reviewer} ({user.get('user_role','')}). {(req.feedback or '')[:120]}"
        registry.append_event(sprint_id, "po", "action", ev_msg)
        sprint_index.update_entry(sprint_id, {"phase": "reviewed_partial", "latest_event": ev_msg})
    elif decision == "recovery_sprint":
        registry.set_status(sprint_id, "complete", "reviewed_partial")
        ev_msg = f"Recovery Sprint recommended by {reviewer} ({user.get('user_role','')})."
        registry.append_event(sprint_id, "sm", "ceremony", ev_msg)
        sprint_index.update_entry(sprint_id, {"phase": "reviewed_partial", "latest_event": ev_msg})
    elif decision == "reopen_blocked":
        registry.set_status(sprint_id, "running", "executing")
        ev_msg = f"Blocked tasks reopened by {reviewer} — sprint continues."
        registry.append_event(sprint_id, "sm", "ceremony", ev_msg)
        sprint_index.update_entry(sprint_id, {"phase": "executing", "latest_event": ev_msg})
    else:
        raise HTTPException(
            status_code=400,
            detail="Unknown decision. Valid: close_partial | recovery_sprint | reopen_blocked",
        )

    _refresh_project_counts((sprint_index.get_entry(sprint_id) or {}).get("project_id"))
    return {"ok": True, "decision": decision, "phase": entry.phase}


# ─────────────────────────────────────────────────────────────────────────────
#  Sprint Review notes (system / human / merged) + Retrospective
# ─────────────────────────────────────────────────────────────────────────────

def _load_sprint_state_dict(sprint_id: str) -> dict:
    """Load the live sprint state for a given sprint_id, or {} if unavailable."""
    try:
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore  # noqa: PLC0415
        return SprintStateStore.for_sprint(sprint_id).load() or {}
    except Exception:
        return {}


@cockpit_router.get("/api/sprint/{sprint_id}/sprint-review")
async def get_sprint_review(sprint_id: str, request: Request):
    _validate_sprint_id(sprint_id)
    user = _parse_user_header(request)
    _check_sprint_access(user, sprint_id)
    from src.services.api import sprint_review_store  # noqa: PLC0415
    entry = sprint_index.get_entry(sprint_id) or {}
    return {
        "sprint_id":          sprint_id,
        "sprint_review":      sprint_review_store.get_review(sprint_id),
        "next_sprint_notes":  sprint_review_store.get_next_sprint_notes(sprint_id),
        "agile_compliance":   entry.get("agile_compliance"),
    }


@cockpit_router.post("/api/sprint/{sprint_id}/sprint-review/system")
async def post_system_review(sprint_id: str, request: Request):
    """(Re)generate system_review from the current sprint outcomes."""
    _validate_sprint_id(sprint_id)
    user = _parse_user_header(request)
    _check_sprint_access(user, sprint_id)

    from src.loop.sprint_review_synth import build_system_review  # noqa: PLC0415
    from src.services.api import sprint_review_store  # noqa: PLC0415

    state = _load_sprint_state_dict(sprint_id)
    generated = len(list_generated_files(sprint_id))
    payload = build_system_review(state, generated_files_count=generated)
    sprint_review_store.update_review(sprint_id, {"system_review": payload})
    return {"ok": True, "system_review": payload}


@cockpit_router.post("/api/sprint/{sprint_id}/sprint-review/human")
async def post_human_review(sprint_id: str, req: HumanReviewRequest, request: Request):
    """Persist the human reviewer's notes onto the sprint."""
    _validate_sprint_id(sprint_id)
    user = _parse_user_header(request)
    _check_sprint_access(user, sprint_id)

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty.")

    reviewer = user["user_name"] or (req.reviewer_name or "").strip() or "anonymous"
    from src.services.api import sprint_review_store  # noqa: PLC0415
    from datetime import timezone  # noqa: PLC0415
    payload = {
        "text":            text,
        "reviewer_id":     user.get("user_id", ""),
        "reviewer_name":   reviewer,
        "reviewer_email":  user.get("user_email", ""),
        "reviewer_role":   user.get("user_role", ""),
        "submitted_at":    datetime.now(timezone.utc).isoformat(),
    }
    sprint_review_store.update_review(sprint_id, {"human_review": payload})
    return {"ok": True, "human_review": payload}


@cockpit_router.post("/api/sprint/{sprint_id}/sprint-review/synthesize")
async def post_synthesize_review(sprint_id: str, request: Request):
    """Merge system + human review into Sprint Notes and propose next sprint."""
    _validate_sprint_id(sprint_id)
    user = _parse_user_header(request)
    _check_sprint_access(user, sprint_id)

    from src.loop.sprint_review_synth import (  # noqa: PLC0415
        build_system_review, synthesize_merged_notes, propose_next_sprint,
    )
    from src.services.api import sprint_review_store  # noqa: PLC0415
    from src.services.api import product_backlog_store  # noqa: PLC0415

    review  = sprint_review_store.get_review(sprint_id)
    system  = review.get("system_review")
    if not system:
        state = _load_sprint_state_dict(sprint_id)
        generated = len(list_generated_files(sprint_id))
        system = build_system_review(state, generated_files_count=generated)
        review["system_review"] = system

    human = review.get("human_review") or {}
    merged_notes = synthesize_merged_notes(system, human)
    review["merged_notes"] = merged_notes
    review["synthesized_at"] = datetime.now().isoformat()
    sprint_review_store.update_review(sprint_id, review)

    entry = sprint_index.get_entry(sprint_id) or {}
    project_id = entry.get("project_id") or ""
    backlog_items = []
    if project_id:
        try:
            backlog_items = product_backlog_store.read(project_id).get("items", [])
        except Exception:
            backlog_items = []
    next_notes = propose_next_sprint(system, backlog_items, last_goal=entry.get("goal", ""))
    sprint_review_store.set_next_sprint_notes(sprint_id, next_notes)

    return {
        "ok":                 True,
        "merged_notes":       merged_notes,
        "next_sprint_notes":  next_notes,
    }


@cockpit_router.post("/api/sprint/{sprint_id}/retro")
async def post_retro(sprint_id: str, req: RetroRequest, request: Request):
    """Persist retro into sprint state and seed action items into the backlog."""
    _validate_sprint_id(sprint_id)
    user = _parse_user_header(request)
    _check_sprint_access(user, sprint_id)

    payload = {
        "went_well": [s for s in (req.went_well or []) if (s or "").strip()],
        "improve":   [s for s in (req.improve   or []) if (s or "").strip()],
        "actions":   [s for s in (req.actions   or []) if (s or "").strip()],
        "reviewer_id":   user.get("user_id", ""),
        "reviewer_name": user["user_name"] or (req.reviewer_name or "").strip() or "anonymous",
        "submitted_at":  datetime.now().isoformat(),
    }

    try:
        from src.pipeline.scrum_team.sprint_state_store import SprintStateStore  # noqa: PLC0415
        store = SprintStateStore.for_sprint(sprint_id)
        with store.state_lock():
            data = store.load()
            data["retro_actions"] = payload
            store.save(data)
    except Exception as exc:
        logger.warning("retro: persist to sprint_state failed: %s", exc)

    # Seed actions into the project backlog as new items.
    seeded_ids: list[str] = []
    entry = sprint_index.get_entry(sprint_id) or {}
    project_id = entry.get("project_id") or ""
    if project_id and payload["actions"]:
        try:
            from src.services.api import product_backlog_store  # noqa: PLC0415
            for title in payload["actions"]:
                item = product_backlog_store.add_item(project_id, {
                    "title":  title,
                    "status": "new",
                    "priority": "medium",
                }, source="sprint_notes")
                if item.get("item_id"):
                    seeded_ids.append(item["item_id"])
        except Exception as exc:
            logger.warning("retro: seeding backlog failed for %s: %s", project_id, exc)

    return {"ok": True, "retro": payload, "seeded_backlog_item_ids": seeded_ids}
