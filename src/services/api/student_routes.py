"""services/api/student_routes.py — Lumi student-dashboard API.

Single-profile endpoints (no auth, no role checks):

    GET    /api/student/health                    -> {degraded: bool}
    GET    /api/student/dashboard                 -> full materialized profile
    POST   /api/student/mood                      -> persist mood
    PATCH  /api/student/assignments/{id}          -> change status
    POST   /api/student/chat                      -> {reply, why, actions, degraded}

All response strings are English. The chat endpoint never surfaces raw Ollama
errors — when the model is unavailable it returns a rule-based fallback with
`degraded: true`.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.services.api import student_profile_store
from src.agents import student_planner

logger = logging.getLogger(__name__)

student_router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
#  Request models
# ─────────────────────────────────────────────────────────────────────────────

class MoodRequest(BaseModel):
    mood: str = Field(..., description="One of: good, ok, meh, tired, stressed.")


class AssignmentStatusRequest(BaseModel):
    status: str = Field(..., description="Pending | Planned | In Progress | Completed | Submitted")


class ChatRequest(BaseModel):
    message: str = ""
    mood: Optional[str] = None
    action: Optional[str] = None  # shorter | postpone | resources | plan | calendar


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@student_router.get("/api/student/health")
async def student_health():
    """1-token probe so the chat panel can pick the right mode badge.

    Returns `degraded: true` if the Ollama call raises for any reason.
    """
    try:
        student_planner._invoke_chat(
            [{"role": "user", "content": "ping"}]
        )
        return {"degraded": False}
    except Exception as exc:
        logger.info("Lumi health: Ollama unavailable (%s)", exc)
        return {"degraded": True}


@student_router.get("/api/student/dashboard")
async def get_dashboard():
    profile = student_profile_store.load_profile()
    return student_profile_store.materialize_dates(profile)


@student_router.post("/api/student/mood")
async def post_mood(req: MoodRequest):
    try:
        return student_profile_store.set_mood(req.mood)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@student_router.patch("/api/student/assignments/{assignment_id}")
async def patch_assignment(assignment_id: str, req: AssignmentStatusRequest):
    try:
        updated = student_profile_store.update_assignment_status(
            assignment_id, req.status
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if updated is None:
        raise HTTPException(status_code=404, detail="Assignment not found.")
    return updated


@student_router.post("/api/student/chat")
async def post_chat(req: ChatRequest):
    profile = student_profile_store.materialize_dates(
        student_profile_store.load_profile()
    )
    result = student_planner.build_plan(
        profile=profile,
        message=req.message or "What should I do today?",
        mood=req.mood,
        action=req.action,
    )
    # Defensive shape: always include all four keys.
    result.setdefault("reply", "")
    result.setdefault("why", "")
    result.setdefault("actions", list(student_planner.ACTION_CHIPS))
    result.setdefault("degraded", True)
    return result
