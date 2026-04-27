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

from src.core.llm_config import DEFAULT_MODEL

logger = logging.getLogger(__name__)

_MODEL = DEFAULT_MODEL

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
  "title": story title in English, max 80 chars (string)
  "description": "As a user, I want to [action] so that [benefit]." in English (string)
  "acceptance_criteria": list of 2–4 strings, each max 80 chars — REQUIRED, never empty
  "priority": exactly one of "high", "medium", "low"
  "story_points": integer effort estimate 1–8 (Fibonacci: 1, 2, 3, 5, 8)
  "deployment_package": short release package name, max 40 chars (string, can be empty "")
- 1 to 6 items total across all epics.
- acceptance_criteria MUST NOT be an empty list. Every story needs at least 2 criteria.

EXAMPLE:
[
  {
    "epic": "User Authentication",
    "title": "User can register with email",
    "description": "As a user, I want to register with my email so that I can access the platform.",
    "acceptance_criteria": ["Registration form validates email format", "Password must be at least 8 chars", "Success confirmation shown after registration"],
    "priority": "high",
    "story_points": 3,
    "deployment_package": "MVP Auth"
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
                    ac = [
                        str(c)[:80]
                        for c in s.get("acceptance_criteria", [])
                        if isinstance(c, str) and c.strip()
                    ][:5]
                    title_en = str(s.get("title", ""))[:80]
                    user_story_en = str(s.get("description", ""))[:300]
                    # story_points: clamp to 1–8, default 2
                    sp_raw = s.get("story_points", 2)
                    try:
                        story_points = max(1, min(8, int(sp_raw)))
                    except (TypeError, ValueError):
                        story_points = 2
                    stories.append({
                        "epic":                 str(s.get("epic", ""))[:80],
                        "title":                title_en,
                        "title_en":             title_en,
                        "title_tr":             "",
                        "description":          user_story_en,
                        "user_story_en":        user_story_en,
                        "user_story_tr":        "",
                        "acceptance_criteria":  ac,
                        "priority":             priority,
                        "story_points":         story_points,
                        "deployment_package":   str(s.get("deployment_package", ""))[:40],
                    })
                if stories:
                    logger.debug("po_llm: generated %d stories", len(stories))
                    return stories[:6]  # cap at 6 per system prompt constraint

        except Exception as exc:
            logger.warning("po_llm: generate_stories LLM call failed: %s", exc)

        # Fallback: one minimal story per epic (always includes AC)
        logger.warning("po_llm: using fallback story generation (%d epics)", len(epics))
        return self._fallback_stories(epics)

    def review_story(
        self,
        story:               dict[str, Any],
        task_output:         str,
        acceptance_criteria: list[str] | None = None,
        sprint_goal:         str = "",
    ) -> dict[str, Any]:
        """Decide whether a completed task satisfies the Definition of Done.

        Returns
        -------
        {
          "accepted":          bool,
          "reason":            str,
          "missing_criteria":  list[str],
        }

        Deterministic implementation (no LLM):
          - If there are no acceptance_criteria, the story is auto-accepted
            (legacy behaviour — matches complete_task() empty-AC bypass).
          - Otherwise, each criterion must share at least one significant
            keyword (≥4 chars, alphanumeric) with the task output, case-
            insensitive. Criteria that fail are listed in missing_criteria.
          - Rejection reason is a short, human-readable summary.

        A deterministic check keeps the acceptance step auditable and
        reproducible; an LLM-backed variant can subclass and override.
        """
        ac = [c for c in (acceptance_criteria or story.get("acceptance_criteria") or []) if isinstance(c, str) and c.strip()]
        if not ac:
            return {
                "accepted":         True,
                "reason":           "No acceptance criteria defined (legacy auto-accept).",
                "missing_criteria": [],
            }

        output_lc = (task_output or "").lower()
        if not output_lc.strip():
            return {
                "accepted":         False,
                "reason":           "Task output is empty.",
                "missing_criteria": list(ac),
            }

        missing: list[str] = []
        for criterion in ac:
            words = [w for w in re.findall(r"[A-Za-z0-9]+", criterion.lower()) if len(w) >= 4]
            if not words:
                # No significant words to match — skip like C8 does
                continue
            if not any(w in output_lc for w in words):
                missing.append(criterion)

        if missing:
            return {
                "accepted":         False,
                "reason":           f"{len(missing)} acceptance criterion(a) not evidenced in task output.",
                "missing_criteria": missing,
            }

        return {
            "accepted":         True,
            "reason":           "All acceptance criteria evidenced in task output.",
            "missing_criteria": [],
        }

    def _fallback_stories(
        self, epics: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """Deterministic fallback: one minimal story per epic with guaranteed AC."""
        return [
            {
                "epic":                e["title"],
                "title":               e["title"],
                "title_en":            e["title"],
                "title_tr":            "",
                "description":         (
                    f"As a user, I want to {e['title'].lower()} "
                    "so that the sprint goal is achieved."
                ),
                "user_story_en":       (
                    f"As a user, I want to {e['title'].lower()} "
                    "so that the sprint goal is achieved."
                ),
                "user_story_tr":       "",
                "acceptance_criteria": [
                    f"Feature described by '{e['title']}' is implemented and functional.",
                    "No regression in existing functionality.",
                ],
                "priority":            "medium",
                "story_points":        2,
                "deployment_package":  "",
            }
            for e in epics
        ]
