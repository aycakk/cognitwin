"""src/agents/student_planner.py — Lumi daily-plan generator.

Produces an English study plan for the Lumi student dashboard.

Two execution modes, in this order:
  1. _llm_plan  — calls the local Ollama model (via _safe_chat) with a
     compact JSON-output prompt. Returns parsed JSON if the model produced
     well-formed output.
  2. _rule_based_plan — deterministic fallback used whenever the LLM
     path raises (Ollama down, malformed JSON, schema mismatch, etc.).

The frontend treats `degraded=True` as "show the basic-planning-mode badge
and prepend the user-friendly fallback line" — see lumi.html.

All user-facing strings are English.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


ACTION_CHIPS = ["Plan", "Postpone", "Make Shorter", "Suggest Resources", "Add to Calendar"]


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_plan(
    profile: dict,
    message: str,
    mood: Optional[str] = None,
    action: Optional[str] = None,
) -> dict:
    """Return {reply, why, actions, degraded}.

    `profile` is the materialized dashboard dict (with absolute days_left/due_date).
    `action` is one of None | "shorter" | "postpone" | "resources" | "plan" | "calendar".
    """
    try:
        result = _llm_plan(profile, message, mood, action)
        result.setdefault("actions", list(ACTION_CHIPS))
        result["degraded"] = False
        return result
    except Exception as exc:
        logger.warning("Lumi LLM unavailable: %s — using rule-based planner.", exc)
        result = _rule_based_plan(profile, mood, action)
        result["degraded"] = True
        return result


# ─────────────────────────────────────────────────────────────────────────────
#  Rule-based fallback  (deterministic, English)
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based_plan(
    profile: dict,
    mood: Optional[str],
    action: Optional[str],
) -> dict:
    deadlines = sorted(
        profile.get("deadlines", []),
        key=lambda d: d.get("days_left", 999),
    )
    courses = {c["id"]: c for c in profile.get("courses", [])}

    # Pick the top item (most urgent open assignment or exam).
    top = deadlines[0] if deadlines else None

    # Find the weakest course (lowest average) for a review block.
    weakest = min(
        profile.get("courses", []),
        key=lambda c: c.get("average", 100),
        default=None,
    )

    # Resources action — generic but course-aware.
    if action == "resources" and top is not None:
        course = courses.get(top.get("course_id"), {})
        return _resources_reply(top, course)

    # Build the base block plan from the top item.
    blocks = _initial_blocks(top, weakest)

    # Mood adjustment.
    empathy = ""
    if mood in ("tired", "stressed"):
        blocks = [_shrink_block(b, factor=0.6) for b in blocks]
        empathy = (
            "I can see today is heavy — I kept the blocks short so you can "
            "actually finish them.\n\n"
        )

    # Action transforms.
    revised_due = None
    if action == "shorter":
        blocks = [_shrink_block(b, factor=0.5) for b in blocks]
        # Drop the lowest-priority (last) block when there are 3+.
        if len(blocks) >= 3:
            blocks = blocks[:-1]
    elif action == "postpone" and top is not None:
        revised_due = max(int(top.get("days_left", 0)) + 1, 1)

    reply_text = empathy + _format_plan(blocks, top, revised=revised_due)
    why_text = _format_why(top, weakest, blocks, mood, action)

    return {
        "reply": reply_text,
        "why": why_text,
        "actions": list(ACTION_CHIPS),
    }


def _initial_blocks(top: Optional[dict], weakest: Optional[dict]) -> list[dict]:
    """Return a list of {title, minutes} blocks tied to the top deadline."""
    blocks: list[dict] = []

    if top is not None:
        days_left = int(top.get("days_left", 0))
        title = top["title"]
        if days_left <= 1:
            # Crunch — 3 sequenced blocks.
            blocks.append({"title": f"Research for {title}",  "minutes": 45})
            blocks.append({"title": f"Draft slides for {title}", "minutes": 30})
            blocks.append({"title": f"Rehearse {title}",       "minutes": 20})
        elif days_left <= 3:
            blocks.append({"title": f"Outline {title}",       "minutes": 40})
            blocks.append({"title": f"Draft for {title}",     "minutes": 30})
        else:
            blocks.append({"title": f"Start {title}",         "minutes": 30})

    if weakest is not None:
        blocks.append({
            "title": f"{weakest['name']} review",
            "minutes": 25,
        })

    if not blocks:
        blocks.append({"title": "Light review of weakest course", "minutes": 25})

    return blocks


def _shrink_block(block: dict, factor: float) -> dict:
    minutes = max(10, int(round(block["minutes"] * factor / 5.0) * 5))
    return {**block, "minutes": minutes}


def _format_plan(blocks: list[dict], top: Optional[dict], revised: Optional[int]) -> str:
    if not blocks:
        return "Nothing urgent today — use the time for review or rest."

    lines: list[str] = []
    if top is not None:
        if revised is not None:
            lines.append(
                f"Postponing {top['title']} by one day — the new target is in "
                f"{revised} day(s). Today's lighter plan:"
            )
        else:
            days_left = int(top.get("days_left", 0))
            when = "tomorrow" if days_left == 1 else f"in {days_left} day(s)"
            lines.append(
                f"Your {top['title']} is due {when}, so let's prioritize it today:"
            )
    for i, b in enumerate(blocks, 1):
        lines.append(f"{i}. {b['minutes']} minutes — {b['title']}")

    return "\n".join(lines)


def _format_why(
    top: Optional[dict],
    weakest: Optional[dict],
    blocks: list[dict],
    mood: Optional[str],
    action: Optional[str],
) -> str:
    parts: list[str] = []
    if action == "shorter":
        parts.append("Made the plan shorter — total time roughly halved.")
    elif action == "postpone" and top is not None:
        parts.append(f"Pushed {top['title']} back by one day per your request.")
    elif top is not None:
        parts.append(
            f"I suggested this because {top['title']} is due in "
            f"{int(top.get('days_left', 0))} day(s)"
        )
        if weakest is not None:
            parts[-1] += (
                f", and your {weakest['name']} average ({weakest.get('average', '?')}) "
                f"is the lowest — worth a short review block."
            )
        else:
            parts[-1] += "."
    if mood in ("tired", "stressed"):
        parts.append("I shortened the blocks so the plan fits your energy today.")
    return " ".join(parts) if parts else "No deadlines are pressing — keep your usual routine."


def _resources_reply(top: dict, course: dict) -> dict:
    name = (course.get("name") or "").lower()
    title = top.get("title", "this assignment")

    if "math" in name:
        bullets = [
            "Khan Academy: Derivatives intro",
            "Paul's Online Math Notes — calculus section",
            "Your textbook §3.2 with worked examples",
        ]
    elif "physics" in name:
        bullets = [
            "Khan Academy: Newton's laws & friction",
            "HyperPhysics — friction concept map",
            "Your lab manual — friction experiment write-up template",
        ]
    elif "english" in name:
        bullets = [
            "Purdue OWL — academic essay structure",
            "Grammarly: tone and clarity tips",
            "Your reading list — pull two supporting quotes",
        ]
    elif "history" in name:
        bullets = [
            "Britannica: Republic Era overview",
            "A primary-source compilation from your school library",
            "Public-television documentaries — pick 10–15 minutes of footage",
        ]
    elif "computer" in name or "cs" in name:
        bullets = [
            "MDN Web Docs — JavaScript basics",
            "freeCodeCamp's todo-app walkthrough",
            "Your project README — define scope first, then code",
        ]
    else:
        bullets = [
            "Course syllabus — confirm scope first",
            "Your class notes from the most recent lesson",
            "Ask your teacher for one starter reference",
        ]

    reply = (
        f"Here are some starting points for {title}:\n"
        + "\n".join(f"- {b}" for b in bullets)
        + "\n\nThese are starting suggestions, not a curated list."
    )
    why = "Pulled generic, well-known resources matched to the course of the top deadline."
    return {"reply": reply, "why": why, "actions": list(ACTION_CHIPS)}


# ─────────────────────────────────────────────────────────────────────────────
#  LLM path
# ─────────────────────────────────────────────────────────────────────────────

_LLM_SYSTEM_PROMPT = (
    "You are Lumi, a calm and concise study coach for a single high-school student. "
    "You speak ENGLISH ONLY. You produce concrete daily plans, not motivational "
    "speeches. You always reply with a single JSON object containing exactly these "
    "keys: \"reply\" (string, the plan), \"why\" (string, one short paragraph). "
    "No extra prose, no markdown fences. Keep total length under 600 characters."
)


def _invoke_chat(messages: list[dict]) -> dict:
    """Default LLM transport.

    Wrapped in a module-level function so tests can monkeypatch it without
    having to import the heavy `src.pipeline.shared` module (which pulls in
    chromadb / ollama). Imports are lazy so this module stays importable in
    environments where the LLM stack is not installed.
    """
    from src.pipeline.shared import _safe_chat
    from src.core.llm_config import DEFAULT_MODEL
    return _safe_chat(DEFAULT_MODEL, messages)


def _llm_plan(
    profile: dict,
    message: str,
    mood: Optional[str],
    action: Optional[str],
) -> dict:
    summary = _summarize_profile(profile, mood, action)
    user_msg = (
        f"STUDENT CONTEXT:\n{summary}\n\n"
        f"STUDENT MESSAGE:\n{message.strip() or 'What should I do today?'}\n\n"
        "Respond with the JSON object only."
    )
    messages = [
        {"role": "system", "content": _LLM_SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    resp = _invoke_chat(messages)
    content = (resp.get("message") or {}).get("content", "").strip()
    parsed = _extract_json(content)
    if not isinstance(parsed, dict):
        raise ValueError(f"LLM did not return a JSON object: {content[:200]!r}")
    reply = str(parsed.get("reply") or "").strip()
    why = str(parsed.get("why") or "").strip()
    if not reply:
        raise ValueError("LLM JSON missing 'reply'")
    if not why:
        why = "Generated from the student's calendar and current deadlines."
    return {"reply": reply, "why": why}


def _summarize_profile(profile: dict, mood: Optional[str], action: Optional[str]) -> str:
    courses = profile.get("courses", [])
    deadlines = profile.get("deadlines", [])[:5]
    schedule = profile.get("today_schedule", [])
    weakest = min(courses, key=lambda c: c.get("average", 100), default=None)

    lines: list[str] = []
    lines.append(f"Today's schedule (HH:MM): " + ", ".join(
        f"{s['start']} {s['title']}" for s in schedule
    ))
    lines.append("Upcoming deadlines: " + (
        ", ".join(
            f"{d['title']} ({d['kind']}, in {d['days_left']}d)"
            for d in deadlines
        ) or "none"
    ))
    if weakest is not None:
        lines.append(
            f"Weakest course: {weakest['name']} avg {weakest.get('average', '?')}/100."
        )
    if mood:
        lines.append(f"Mood: {mood}.")
    if action:
        lines.append(
            "Action requested: "
            + {
                "shorter":   "make today's plan shorter (halve total time)",
                "postpone":  "postpone the top deadline by one day",
                "resources": "list 2-3 starting resources for the top item",
                "plan":      "save the plan to today (no special change)",
                "calendar":  "add the plan to the calendar (no special change)",
            }.get(action, action)
            + "."
        )
    return "\n".join(lines)


def _extract_json(text: str) -> Any:
    """Try to extract a single JSON object from a model response."""
    if not text:
        return None
    # Strip code-fence wrappers if the model added them despite instructions.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    # Direct parse.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # First {...} block.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None
