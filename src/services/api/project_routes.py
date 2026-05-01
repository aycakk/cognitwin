"""services/api/project_routes.py — Project Dashboard API.

Routes
------
GET    /api/projects                            — list current user's projects
POST   /api/projects                            — create a new project
GET    /api/projects/{project_id}               — single project (with counts)
POST   /api/projects/{project_id}/archive       — archive (owner / admin)
POST   /api/projects/{project_id}/reopen        — reopen archived project
GET    /api/projects/{project_id}/sprints       — sprints belonging to project
POST   /api/projects/{project_id}/sprints       — start sprint inside project

Access control
--------------
Non-admin users are scoped to projects where owner_user_id == their user_id.
Admins see/edit all projects. All routes require X-Cognitwin-User.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.services.api import project_index
from src.services.api.cockpit_routes import _parse_user_header, _is_admin, start_sprint, SprintStartRequest
from src.loop import sprint_index

logger = logging.getLogger(__name__)

project_router = APIRouter()

_PROJECT_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,80}$")


# ─────────────────────────────────────────────────────────────────────────────
#  Request models
# ─────────────────────────────────────────────────────────────────────────────

class ProjectCreateRequest(BaseModel):
    title: str
    project_goal: str = ""
    description: str = ""
    workspace: str = ""


class ProjectSprintStartRequest(BaseModel):
    goal: str = ""
    workspace: str = ""
    selected_backlog_item_ids: list[str] = []


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_project_id(project_id: str) -> None:
    if not _PROJECT_ID_RE.match(project_id or ""):
        raise HTTPException(status_code=400, detail="Invalid project_id.")


def _check_project_access(user: dict, project: dict | None) -> None:
    """Raise 404 if missing, 403 if user can't access. Admin always passes."""
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    if _is_admin(user):
        return
    owner_id = project.get("owner_user_id") or ""
    if not owner_id or owner_id != (user.get("user_id") or ""):
        raise HTTPException(
            status_code=403,
            detail="Access denied: you do not own this project.",
        )


def _filter_projects_for_user(projects: list[dict], user: dict) -> list[dict]:
    if _is_admin(user):
        return list(projects)
    uid = user.get("user_id") or ""
    if not uid:
        return []
    return [p for p in projects if p.get("owner_user_id") == uid]


# ─────────────────────────────────────────────────────────────────────────────
#  Routes — projects
# ─────────────────────────────────────────────────────────────────────────────

@project_router.get("/api/projects")
async def list_projects(request: Request, archived: bool = False):
    user     = _parse_user_header(request)
    projects = project_index.read_all(archived=archived)
    visible  = _filter_projects_for_user(projects, user)
    return {"projects": visible, "user_is_admin": _is_admin(user)}


@project_router.post("/api/projects")
async def create_project(req: ProjectCreateRequest, request: Request):
    title = (req.title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title must not be empty.")

    user = _parse_user_header(request)
    if not (user.get("user_id") or ""):
        raise HTTPException(
            status_code=400,
            detail="user_id required to create a project (set X-Cognitwin-User header).",
        )

    record = project_index.create(
        title            = title,
        project_goal     = (req.project_goal or "").strip(),
        description      = (req.description  or "").strip(),
        workspace        = (req.workspace    or "").strip(),
        owner_user_id    = user["user_id"],
        owner_user_name  = user.get("user_name", ""),
        owner_user_email = user.get("user_email", ""),
    )
    logger.info("project: created project=%s by user=%s", record["project_id"], user["user_id"])
    return record


@project_router.get("/api/projects/{project_id}")
async def get_project(project_id: str, request: Request):
    _validate_project_id(project_id)
    user    = _parse_user_header(request)
    project = project_index.get(project_id)
    _check_project_access(user, project)
    # Recompute counts on read so stale caches self-heal.
    refreshed = project_index.recompute_counts(project_id) or project
    return refreshed


@project_router.post("/api/projects/{project_id}/archive")
async def archive_project(project_id: str, request: Request):
    _validate_project_id(project_id)
    user    = _parse_user_header(request)
    project = project_index.get(project_id)
    _check_project_access(user, project)
    if not project_index.archive(project_id):
        raise HTTPException(status_code=404, detail="Project not found.")
    return {"ok": True, "project_id": project_id, "status": "archived"}


@project_router.post("/api/projects/{project_id}/reopen")
async def reopen_project(project_id: str, request: Request):
    _validate_project_id(project_id)
    user    = _parse_user_header(request)
    project = project_index.get(project_id)
    _check_project_access(user, project)
    if not project_index.reopen(project_id):
        raise HTTPException(status_code=404, detail="Project not found.")
    return {"ok": True, "project_id": project_id, "status": "active"}


# ─────────────────────────────────────────────────────────────────────────────
#  Routes — project sprints
# ─────────────────────────────────────────────────────────────────────────────

@project_router.get("/api/projects/{project_id}/sprints")
async def list_project_sprints(project_id: str, request: Request, archived: bool = False):
    _validate_project_id(project_id)
    user    = _parse_user_header(request)
    project = project_index.get(project_id)
    _check_project_access(user, project)

    sprints = sprint_index.read_by_project(project_id)
    if not archived:
        sprints = [s for s in sprints if not s.get("archived")]

    # Merge live registry phase/status for running sprints (matches /api/sprints).
    from src.services.api import sprint_run_registry as registry  # local import
    for s in sprints:
        live = registry.get(s.get("sprint_id", ""))
        if live:
            s["phase"]  = live.phase
            s["status"] = live.status

    return {"project_id": project_id, "sprints": sprints}


@project_router.get("/api/projects/{project_id}/next-sprint-suggestion")
async def next_sprint_suggestion(project_id: str, request: Request):
    """Surface the next-sprint plan derived from the project's most recently
    closed sprint (Sprint Notes synthesis output). Returns an empty payload
    when no closed sprint has notes yet.
    """
    _validate_project_id(project_id)
    user    = _parse_user_header(request)
    project = project_index.get(project_id)
    _check_project_access(user, project)

    sprints = sprint_index.read_by_project(project_id)
    # Closed phases that should contribute suggestions.
    _CLOSED = {"reviewed_partial", "complete", "blocked"}
    closed = [s for s in sprints if (s.get("phase") or "").lower() in _CLOSED]
    closed.sort(key=lambda s: s.get("updated_at") or s.get("created_at") or "", reverse=True)

    suggestion: dict = {}
    source_sprint_id: str = ""
    for s in closed:
        notes = s.get("next_sprint_notes") or {}
        if notes:
            suggestion = dict(notes)
            source_sprint_id = s.get("sprint_id", "")
            break

    # Resolve carry-over IDs against the live backlog so deleted items drop out.
    if suggestion:
        try:
            from src.services.api import product_backlog_store  # noqa: PLC0415
            items = product_backlog_store.read(project_id).get("items", [])
            by_id = {it["item_id"]: it for it in items}
            still_in_sprint = {
                it["item_id"] for it in items
                if str(it.get("status", "")).lower() == "in_sprint"
                and it.get("committed_in_sprint") and it["committed_in_sprint"] != source_sprint_id
            }
            suggestion["carry_over_item_ids"] = [
                iid for iid in suggestion.get("carry_over_item_ids", []) or []
                if iid in by_id and iid not in still_in_sprint
            ]
            suggestion["recommended_new_item_ids"] = [
                iid for iid in suggestion.get("recommended_new_item_ids", []) or []
                if iid in by_id and str(by_id[iid].get("status", "")).lower() in ("new", "deferred")
            ]
        except Exception as exc:
            logger.warning("next-sprint: backlog reconcile failed for %s: %s", project_id, exc)

    return {
        "project_id":         project_id,
        "source_sprint_id":   source_sprint_id,
        "next_sprint_notes":  suggestion,
    }


@project_router.post("/api/projects/{project_id}/sprints")
async def start_project_sprint(project_id: str, req: ProjectSprintStartRequest, request: Request):
    """Start a sprint inside a project. Wraps cockpit_routes.start_sprint
    so all sprint-launch logic stays in one place."""
    _validate_project_id(project_id)
    user    = _parse_user_header(request)
    project = project_index.get(project_id)
    _check_project_access(user, project)

    if (project.get("status") or "").lower() == "archived":
        raise HTTPException(
            status_code=409,
            detail="Cannot start a sprint in an archived project. Reopen it first.",
        )

    sprint_req = SprintStartRequest(
        goal       = req.goal,
        workspace  = req.workspace or project.get("workspace", ""),
        user_id    = user.get("user_id", ""),
        user_name  = user.get("user_name", ""),
        project_id = project_id,
        selected_backlog_item_ids = req.selected_backlog_item_ids or [],
    )
    result = await start_sprint(sprint_req, request)
    # start_sprint already wrote project_id into the sprint index; refresh counts.
    project_index.recompute_counts(project_id)
    return result
