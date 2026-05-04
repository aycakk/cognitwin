"""services/api/product_backlog_routes.py — Product Backlog API.

Routes
------
POST   /api/projects/{project_id}/backlog/seed     — generate from project_goal (idempotent)
GET    /api/projects/{project_id}/backlog          — list items
POST   /api/projects/{project_id}/backlog/items    — add one item
PATCH  /api/projects/{project_id}/backlog/items/{item_id}  — edit
DELETE /api/projects/{project_id}/backlog/items/{item_id}  — remove (only if new/deferred)

Access control
--------------
Same scoping as project_routes: owner or admin. All routes require X-Cognitwin-User.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.services.api import project_index, product_backlog_store
from src.services.api.cockpit_routes import _parse_user_header, _is_admin

logger = logging.getLogger(__name__)

product_backlog_router = APIRouter()

_PROJECT_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,80}$")
_ITEM_ID_RE    = re.compile(r"^PB-\d{1,6}$")


# ─────────────────────────────────────────────────────────────────────────────
#  Request models
# ─────────────────────────────────────────────────────────────────────────────

class BacklogItemCreateRequest(BaseModel):
    title: str
    description: str = ""
    acceptance_criteria: list[str] = []
    priority: str = "medium"
    story_points: int = 2
    epic: str = ""


class BacklogItemUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    acceptance_criteria: Optional[list[str]] = None
    priority: Optional[str] = None
    story_points: Optional[int] = None
    epic: Optional[str] = None
    status: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_project_id(project_id: str) -> None:
    if not _PROJECT_ID_RE.match(project_id or ""):
        raise HTTPException(status_code=400, detail="Invalid project_id.")


def _validate_item_id(item_id: str) -> None:
    if not _ITEM_ID_RE.match(item_id or ""):
        raise HTTPException(status_code=400, detail="Invalid item_id.")


def _check_project_access(user: dict, project: dict | None) -> None:
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    if _is_admin(user):
        return
    owner_id = project.get("owner_user_id") or ""
    if not owner_id or owner_id != (user.get("user_id") or ""):
        raise HTTPException(status_code=403, detail="Access denied: you do not own this project.")


def _seed_with_po_llm(project_goal: str, project_title: str) -> list[dict[str, Any]]:
    """Call POLLMAgent to derive epics → stories. Falls back deterministically."""
    if not project_goal.strip():
        return []
    try:
        from src.agents.po_llm_agent import POLLMAgent  # local import — heavy chain
        agent = POLLMAgent()
        epics = agent.decompose_goal(project_goal)
        stories = agent.generate_stories(epics)
    except Exception as exc:
        logger.warning("backlog seed: POLLMAgent failed (%s); using fallback", exc)
        stories = []

    if stories:
        return [
            {
                "title":               s.get("title", "")[:120],
                "description":         s.get("description", "")[:400],
                "acceptance_criteria": s.get("acceptance_criteria") or [],
                "priority":            s.get("priority", "medium"),
                "story_points":        s.get("story_points", 2),
                "epic":                s.get("epic", ""),
                "status":              "new",
                "source":              "po_llm",
            }
            for s in stories
        ]

    # Deterministic fallback so the UI still gets something useful.
    return [{
        "title":               (project_title or project_goal)[:120] or "Initial deliverable",
        "description":         project_goal[:400],
        "acceptance_criteria": ["Deliverable matches the project goal."],
        "priority":            "high",
        "story_points":        3,
        "epic":                "",
        "status":              "new",
        "source":              "po_llm",
    }]


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@product_backlog_router.get("/api/projects/{project_id}/backlog")
async def get_backlog(project_id: str, request: Request):
    _validate_project_id(project_id)
    user = _parse_user_header(request)
    project = project_index.get(project_id)
    _check_project_access(user, project)
    data = product_backlog_store.read(project_id)
    return data


@product_backlog_router.post("/api/projects/{project_id}/backlog/seed")
async def seed_backlog(project_id: str, request: Request, force: bool = False):
    _validate_project_id(project_id)
    user = _parse_user_header(request)
    project = project_index.get(project_id)
    _check_project_access(user, project)

    if product_backlog_store.exists(project_id) and not force:
        existing = product_backlog_store.read(project_id)
        if existing.get("items"):
            return {"ok": True, "skipped": True, "reason": "backlog_exists", **existing}

    project_goal  = project.get("project_goal", "")
    project_title = project.get("title", "")

    loop = asyncio.get_running_loop()
    items = await loop.run_in_executor(None, _seed_with_po_llm, project_goal, project_title)
    data  = product_backlog_store.write_items(project_id, items, source="po_llm")
    logger.info("backlog: seeded project=%s with %d items", project_id, len(items))
    return {"ok": True, "skipped": False, **data}


@product_backlog_router.post("/api/projects/{project_id}/backlog/items")
async def create_item(project_id: str, req: BacklogItemCreateRequest, request: Request):
    _validate_project_id(project_id)
    user = _parse_user_header(request)
    project = project_index.get(project_id)
    _check_project_access(user, project)

    if not (req.title or "").strip():
        raise HTTPException(status_code=400, detail="title must not be empty.")

    try:
        entry = product_backlog_store.add_item(project_id, req.model_dump(), source="user")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return entry


@product_backlog_router.patch("/api/projects/{project_id}/backlog/items/{item_id}")
async def patch_item(project_id: str, item_id: str, req: BacklogItemUpdateRequest, request: Request):
    _validate_project_id(project_id)
    _validate_item_id(item_id)
    user = _parse_user_header(request)
    project = project_index.get(project_id)
    _check_project_access(user, project)

    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update.")

    updated = product_backlog_store.update_item(project_id, item_id, updates)
    if updated is None:
        raise HTTPException(status_code=404, detail="Backlog item not found.")
    return updated


@product_backlog_router.delete("/api/projects/{project_id}/backlog/items/{item_id}")
async def delete_item(project_id: str, item_id: str, request: Request):
    _validate_project_id(project_id)
    _validate_item_id(item_id)
    user = _parse_user_header(request)
    project = project_index.get(project_id)
    _check_project_access(user, project)

    result = product_backlog_store.delete_item(project_id, item_id)
    if result == "not_found":
        raise HTTPException(status_code=404, detail="Backlog item not found.")
    if result == "locked":
        raise HTTPException(
            status_code=409,
            detail="Cannot delete item with status 'in_sprint' or 'done'.",
        )
    return {"ok": True, "item_id": item_id}
