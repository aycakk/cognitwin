"""agents/po_llm_agent.py — LLM-backed Product Owner for autonomous sprint planning.

Provides goal decomposition and story generation via Ollama (llama3.2).
This is a SEPARATE class from the existing rule-based ProductOwnerAgent,
which handles interactive backlog mutations and remains unchanged.

POLLMAgent is only called by sprint_loop.py during the PLAN phase.
It does NOT write to sprint state directly — it returns plain dicts
that the sprint loop persists via SprintStateStore.add_story().

JSON reliability strategy:
  1. Strict system prompt with a clear template + example
  2. Regex-based JSON extraction from LLM output
  3. Hard fallback: if extraction fails, synthesise minimal stories from goal text
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_MODEL = "llama3.2"

# ─────────────────────────────────────────────────────────────────────────────
#  System prompts
# ─────────────────────────────────────────────────────────────────────────────

_EPIC_SYSTEM_PROMPT = """You are a Product Owner. Break down a software goal into 1–4 epics.

STRICT RULES:
- Output ONLY a valid JSON array. No explanation, no markdown, no commentary.
- Each item: {"title": "...", "description": "..."}
- title: max 80 characters
- description: max 200 characters
- 1 to 4 items total

EXAMPLE:
[
  {"title": "User Authentication", "description": "Allow users to register and log in securely."},
  {"title": "Dashboard", "description": "Show key metrics after login."}
]"""

_STORY_SYSTEM_PROMPT = """You are a Product Owner. Convert epics into user stories.

STRICT RULES:
- Output ONLY a valid JSON array. No explanation, no markdown, no commentary.
- Each item must have ALL these fields:
  "epic": parent epic title (string)
  "title": story title, max 80 chars (string)
  "description": "As a user, I want to [action] so that [benefit]." (string)
  "acceptance_criteria": list of 2–3 strings, each max 80 chars
  "priority": exactly one of "high", "medium", "low"
- 1 to 6 items total across all epics.

EXAMPLE:
[
  {
    "epic": "User Authentication",
    "title": "User can register with email",
    "description": "As a user, I want to register with my email so that I can access the platform.",
    "acceptance_criteria": ["Registration form validates email format", "Password must be at least 8 chars", "Success confirmation shown"],
    "priority": "high"
  }
]"""


# ─────────────────────────────────────────────────────────────────────────────
#  JSON extraction helper
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> Any:
    """Extract the first valid JSON array or object from raw LLM output.

    Tries three strategies:
      1. Direct parse (clean output)
      2. Regex: find the outermost [...] block
      3. Regex: find the outermost {...} block (for single-item output)
    """
    text = (text or "").strip()

    # Strategy 1: clean parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: outermost JSON array
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: outermost JSON object (wrap in list)
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group())
            return [obj] if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
#  POLLMAgent
# ─────────────────────────────────────────────────────────────────────────────

class POLLMAgent:
    """
    LLM-backed Product Owner for autonomous sprint planning.

    Only two methods are used by the sprint loop:
      decompose_goal(goal, context) → list of epic dicts
      generate_stories(epics)       → list of story dicts

    The existing rule-based ProductOwnerAgent handles interactive commands
    (create_story, prioritize, accept, reject) and is not replaced.
    """

    def __init__(self, model: str = _MODEL) -> None:
        self._model = model

    # ─────────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────────

    def decompose_goal(
        self,
        goal: str,
        context: str = "",
    ) -> list[dict[str, str]]:
        """Break a high-level goal string into a list of epic dicts.

        Returns: list of {"title": str, "description": str}
        Falls back to a single epic equal to the goal if the LLM fails.
        """
        context_section = f"\nContext from previous sprints: {context}" if context else ""
        user_prompt = f"Goal: {goal}{context_section}\n\nDecompose this goal into epics:"

        messages = [
            {"role": "system", "content": _EPIC_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        try:
            from src.pipeline.shared import _safe_chat  # lazy — avoids ollama import at module load
            resp   = _safe_chat(self._model, messages)
            raw    = resp.get("message", {}).get("content", "")
            parsed = _extract_json(raw)

            if isinstance(parsed, list) and parsed:
                epics = [
                    {
                        "title":       str(e.get("title", "Unnamed Epic"))[:80],
                        "description": str(e.get("description", ""))[:200],
                    }
                    for e in parsed
                    if isinstance(e, dict) and e.get("title")
                ]
                if epics:
                    logger.debug("po_llm: decomposed goal into %d epics", len(epics))
                    return epics[:4]  # cap at 4 per system prompt constraint

        except Exception as exc:
            logger.warning("po_llm: decompose_goal LLM call failed: %s", exc)

        # Fallback: treat the entire goal as one epic
        logger.warning("po_llm: using fallback single-epic for goal=%r", goal[:40])
        return [{"title": goal[:80], "description": goal[:200]}]

    def generate_stories(
        self,
        epics: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Generate user stories for a list of epics.

        Returns: list of {epic, title, description, acceptance_criteria, priority}
        Falls back to one minimal story per epic if the LLM fails.
        """
        if not epics:
            return []

        epics_text  = json.dumps(epics, ensure_ascii=False)
        user_prompt = (
            f"Epics:\n{epics_text}\n\n"
            "Generate user stories for these epics:"
        )

        messages = [
            {"role": "system", "content": _STORY_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        try:
            from src.pipeline.shared import _safe_chat  # lazy
            resp   = _safe_chat(self._model, messages)
            raw    = resp.get("message", {}).get("content", "")
            parsed = _extract_json(raw)

            if isinstance(parsed, list) and parsed:
                stories = []
                for s in parsed:
                    if not isinstance(s, dict) or not s.get("title"):
                        continue
                    priority = s.get("priority", "medium")
                    if priority not in ("high", "medium", "low"):
                        priority = "medium"
                    stories.append({
                        "epic":                 str(s.get("epic", ""))[:80],
                        "title":                str(s.get("title", ""))[:80],
                        "description":          str(s.get("description", ""))[:300],
                        "acceptance_criteria":  [
                            str(c)[:80]
                            for c in s.get("acceptance_criteria", [])
                            if isinstance(c, str) and c.strip()
                        ][:5],
                        "priority":             priority,
                    })
                if stories:
                    logger.debug("po_llm: generated %d stories", len(stories))
                    return stories[:6]  # cap at 6 per system prompt constraint

        except Exception as exc:
            logger.warning("po_llm: generate_stories LLM call failed: %s", exc)

        # Fallback: one minimal story per epic
        logger.warning("po_llm: using fallback story generation (%d epics)", len(epics))
        return [
            {
                "epic":                e["title"],
                "title":               e["title"],
                "description":         (
                    f"As a user, I want to {e['title'].lower()} "
                    "so that the sprint goal is achieved."
                ),
                "acceptance_criteria": ["Feature works as described by the epic."],
                "priority":            "medium",
            }
            for e in epics
        ]
