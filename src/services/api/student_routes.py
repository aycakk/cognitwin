"""services/api/student_routes.py — Lumi student-dashboard API.

Single-profile endpoints (no auth, no role checks):

    GET    /api/student/health                    -> {degraded: bool}
    GET    /api/student/dashboard                 -> full materialized profile
    POST   /api/student/mood                      -> persist mood
    PATCH  /api/student/assignments/{id}          -> change status
    POST   /api/student/chat                      -> {reply, why, actions, degraded}
    POST   /api/student/plan/complete             -> mark current plan complete

All response strings are English. The chat endpoint never surfaces raw Ollama
errors — when the model is unavailable it returns a rule-based fallback with
`degraded: true`.

Personalization signals (block-length learning, postpone/shorten counters,
streak, focus points) ride on the existing `/chat` body and the new
`/plan/complete` endpoint — the API surface stays small.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
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


class PlanCompleteRequest(BaseModel):
    plan_id: Optional[str] = None


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
    # Defensive shape: always include the core keys.
    result.setdefault("reply", "")
    result.setdefault("why", "")
    result.setdefault("actions", list(student_planner.ACTION_CHIPS))
    result.setdefault("degraded", True)
    result.setdefault("blocks", [])

    # Persist the plan as `last_plan` so the dashboard "Why this plan?" card
    # has something to render, and apply chip-driven personalization signals.
    last_plan = {
        "id": f"plan-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trigger_message": req.message or "",
        "mood": req.mood,
        "action": req.action,
        "blocks": result.get("blocks") or [],
        "why": result.get("why") or "",
        "outcome": None,
    }
    try:
        student_profile_store.apply_chat_signal(req.action, last_plan=last_plan)
    except Exception as exc:
        # Personalization is best-effort — never break the chat reply.
        logger.warning("Lumi personalization update failed: %s", exc)

    return result


@student_router.post("/api/student/plan/complete")
async def post_plan_complete(req: PlanCompleteRequest):
    """Mark the current plan complete: bumps focus points, completion count,
    and streak. Returns the updated personalization block."""
    data = student_profile_store.complete_plan(plan_id=req.plan_id)
    return {
        "personalization": data.get("personalization") or {},
        "last_plan": data.get("last_plan"),
    }
