"""pipeline/product_owner_runner.py — Product Owner pipeline path.

Routing
───────
Two modes, decided by _is_planning_request():

  PLANNING mode  (natural-language goal description)
    Detected when query contains planning/generation keywords.
    Delegated to POLLMAgent.decompose_goal() + generate_stories().
    Results are persisted to backlog via SprintStateStore and returned
    as a human-readable summary.

  COMMAND mode  (legacy mutation commands)
    Detected by the existing ProductOwnerAgent intent patterns.
    Handles: hikaye oluştur, backlog listele, S-001 öncelik high,
             kabul kriterleri, accept, reject, review_completed, etc.
    Fully deterministic — no LLM, no ChromaDB.

Gate coverage (both modes):
  C1 — PII leak detection
  C4 — Hallucination marker sweep
"""

from __future__ import annotations

import re

from src.agents.product_owner_agent import ProductOwnerAgent
from src.core.schemas import AgentTask, AgentResponse, AgentRole, TaskStatus
from src.pipeline.scrum_team.sprint_state_store import SprintStateStore

# Shared sprint state — Product Owner writes to backlog only.
_SPRINT_STATE = SprintStateStore()
_agent = ProductOwnerAgent(state_store=_SPRINT_STATE)

# ───────────────────────────────────────────────────────────────────────��─────
#  Gate patterns  (kept self-contained — no vector-memory imports)
# ─────────────────────────────────────────────────────────────────────────────

_PII_RE = re.compile(
    r"\b\d{9,12}\b"
    r"|[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}"
    r"|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
)

_HALLUCINATION_RE = re.compile(
    r"sanırım|galiba|muhtemelen|tahmin\s+etmek|belki|herhalde",
    re.I,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Planning-intent detection
#  Matches natural-language product-planning prompts that should go to the
#  LLM path, NOT the rule-based command handler.
#
#  A query matches when it contains ANY of:
#    • an explicit planning verb + artifact noun ("create epics", "generate backlog")
#    • Turkish equivalents ("epic oluştur", "backlog yarat")
#    • "for:" pattern with a goal description after it
#    • multiple artifact nouns in one sentence (epics + stories + criteria)
#    • "decompose" / "break down" / "plan for"
# ─────────────────────────────────────────────────────────────────────────────

_PLANNING_RE = re.compile(
    # English planning verbs + artifact nouns (allow up to 2 filler words between them)
    r"create\s+(?:\w+\s+){0,3}(?:epics?|backlog|user\s+stories?|stories?|acceptance)"
    r"|generate\s+(?:\w+\s+){0,3}(?:epics?|backlog|stories?|requirements?|plan)"
    r"|write\s+(?:\w+\s+){0,3}(?:epics?|stories?|user\s+stories?|acceptance\s+criteria)"
    r"|plan\s+(?:for|the|a|out|this)"
    r"|break\s+(?:down|this|it)\s+into"
    r"|decompose\s+(?:this|the|into|goal)?"
    # Standalone canonical terms — "user stories" is never a mutation command
    r"|user\s+stories?\s+(?:for|about|covering|describing|to|and)"
    r"|user\s+stories?\s+for\b"
    # "epics, stories, and acceptance criteria" — multiple artifact nouns in one query
    r"|(?:epics?.*stories?|stories?.*epics?|backlog.*stories?|stories?.*backlog)"
    r"|(?:epics?.*acceptance|acceptance.*epics?)"
    # "for: <goal>" pattern
    r"|(?:epics?|backlog|stories?|criteria)\s+for\s*:"
    # ── Turkish: <artifact-noun stem + any suffix> + planning verb ────────────
    # Using stem + \w* catches every Turkish morphological variant in one rule:
    #   singular:   epic, hikaye, kriter, gereksinim
    #   plural:     epikler, hikayeler, kriterler, gereksinimler
    #   accusative: epikleri, hikayeleri, hikayelerini, kriterleri, kriterlerini
    #
    # _COMMAND_RE runs FIRST in _is_planning_request(), so
    # "hikaye oluştur: <title>" (colon present) stays on the rule path
    # even though "hikaye oluştur" (no colon) correctly matches here.
    #
    # Extended verb list — added çıkar, türet, belirle, tanımla, listele, analiz
    # so that "kabul kriterlerini çıkar" and "hikayeleri belirle" route to
    # the LLM planning path instead of the command path that emits "hikaye ID gerekli".
    r"|(?:epic|epik)\w*\s+(?:oluştur|yaz|yarat|üret|hazırla|çıkar|türet|belirle|tanımla)\b"
    r"|backlog\s+(?:oluştur|yaz|yarat|üret|hazırla|çıkar|türet|oluşturalım)\b"
    r"|hikaye\w*\s+(?:oluştur|yaz|yarat|üret|hazırla|çıkar|türet|belirle|tanımla|listele)\b"
    r"|kullanıcı\s+hikaye\w*\s+(?:oluştur|yaz|yarat|üret|hazırla|çıkar|türet|belirle|tanımla)\b"
    r"|kabul\s+kriter\w*\s+(?:oluştur|yaz|yarat|üret|hazırla|çıkar|türet|belirle|tanımla)\b"
    r"|kriter\w*\s+(?:oluştur|yaz|yarat|üret|hazırla|çıkar|türet|belirle|tanımla)\b"
    r"|gereksinim\w*\s+(?:oluştur|yaz|yarat|üret|hazırla|çıkar|türet|belirle|tanımla)\b"
    # "X için:" pattern with a goal description after the colon
    r"|(?:için|for)\s*:\s*\w"
    # New project trigger — "yeni proje" always routes to planning, never command
    r"|yeni\s+proje\b"
    r"|proje\s+başlat\b"
    r"|proje\s+senaryosu\b",
    re.I,
)

# Legacy mutation commands that must NOT be redirected to LLM.
# These match the existing ProductOwnerAgent._INTENT_PATTERNS.
_COMMAND_RE = re.compile(
    r"hikaye\s+oluştur\s*:"           # hikaye oluştur: <title>  (has colon = specific command)
    r"|story\s+ekle\s*:"
    r"|create\s+story\s*:"
    r"|\bS-\d+\b"                      # any explicit story ID reference
    r"|backlog\s+listele"
    r"|list\s+backlog"
    r"|show\s+backlog"
    r"|backlog\s+durum"
    r"|backlog\s+status"
    r"|review\s+completed"
    r"|tamamlanan.*incele"
    r"|inceleme\s+bekleyen",
    re.I,
)


def _is_planning_request(query: str) -> bool:
    """
    Return True when the query is a natural-language planning/generation request
    that should be routed to POLLMAgent.

    Decision order:
      1. If the query matches an explicit legacy command pattern → False (COMMAND mode)
      2. If the query matches a planning pattern                 → True  (PLANNING mode)
      3. Otherwise                                               → False (COMMAND mode)

    Explicit command patterns take priority so that "hikaye oluştur: login ekranı"
    (which has both "oluştur" and a colon-delimited title) stays on the rule path.
    """
    if _COMMAND_RE.search(query):
        return False
    return bool(_PLANNING_RE.search(query))


# ─────────────────────────────────────────────────────────────────────────────
#  LLM planning path
# ─────────────────────────────────────────────────────────────────────────────

def _extract_planning_goal(query: str) -> str:
    """
    Pull the goal description from a planning prompt.

    Handles patterns like:
      "Create epics … for: Add login screen …"   → "Add login screen …"
      "Create epics for Add login screen"         → "Add login screen"
      "Plan the login feature"                    → "login feature"
      plain sentence with no 'for:'               → full query (trimmed)
    """
    # "… for: <goal>" or "… için: <goal>"
    for_colon = re.search(r"(?:for|için)\s*:\s*(.+)", query, re.I | re.DOTALL)
    if for_colon:
        return for_colon.group(1).strip()

    # "… for <goal>" — strip leading planning phrase and return the rest
    for_plain = re.search(
        r"(?:create|generate|write|plan|decompose|break\s+down)"
        r"[\w\s,]+?\s+for\s+(.+)",
        query, re.I | re.DOTALL,
    )
    if for_plain:
        return for_plain.group(1).strip()

    # Remove leading planning verb+artifacts and return remainder
    stripped = re.sub(
        r"^(?:create|generate|write|plan|decompose|break\s+down)[\w\s,]*?[,:]?\s*",
        "",
        query.strip(), count=1, flags=re.I,
    )
    return stripped.strip() or query.strip()


def _run_llm_planning(query: str, task: AgentTask) -> str:
    """
    Decompose a natural-language goal into epics + stories via POLLMAgent,
    persist every story to the backlog, and return a formatted summary.
    """
    from src.agents.po_llm_agent import POLLMAgent  # lazy — avoids ollama at import time

    goal    = _extract_planning_goal(query)
    po_llm  = POLLMAgent()

    epics   = po_llm.decompose_goal(goal)
    stories = po_llm.generate_stories(epics)

    if not stories:
        return (
            "LLM planning failed to generate stories. "
            "Try: /sprint \"" + goal[:60] + "\" for autonomous mode, "
            "or use 'hikaye oluştur: <başlık>' for manual story creation."
        )

    # Persist each story to the backlog
    saved: list[dict] = []
    for s in stories:
        sid = _SPRINT_STATE.add_story(
            title=s.get("title", "Untitled"),
            description=s.get("description", ""),
            priority=s.get("priority", "medium"),
            acceptance_criteria=s.get("acceptance_criteria", []),
        )
        saved.append({**s, "story_id": sid})

    return _format_planning_response(goal, epics, saved)


def _format_planning_response(
    goal:    str,
    epics:   list[dict],
    stories: list[dict],
) -> str:
    """Render a readable planning summary for the UI."""
    lines: list[str] = [
        f"Product Owner — Planning Result",
        f"Goal: {goal}",
        f"{'─' * 60}",
    ]

    # Group stories by epic for display
    epic_map: dict[str, list[dict]] = {}
    for s in stories:
        key = s.get("epic", "General")
        epic_map.setdefault(key, []).append(s)

    for epic in epics:
        epic_title = epic.get("title", "Epic")
        epic_desc  = epic.get("description", "")
        lines.append(f"\n📌 Epic: {epic_title}")
        if epic_desc:
            lines.append(f"   {epic_desc}")

        for s in epic_map.get(epic_title, []):
            sid      = s.get("story_id", "?")
            title    = s.get("title", "-")
            priority = s.get("priority", "medium")
            criteria = s.get("acceptance_criteria", [])

            lines.append(f"\n  📖 [{sid}] {title}  (priority: {priority})")

            desc = s.get("description", "")
            if desc:
                lines.append(f"     {desc}")

            if criteria:
                lines.append("     Acceptance criteria:")
                for c in criteria:
                    lines.append(f"       ✓ {c}")

    lines.append(f"\n{'─' * 60}")
    lines.append(
        f"✅ {len(stories)} {'story' if len(stories) == 1 else 'stories'} added to backlog "
        f"across {len(epics)} {'epic' if len(epics) == 1 else 'epics'}."
    )
    lines.append(
        "   Use 'backlog listele' to review  |  'S-NNN öncelik high' to reprioritize  "
        "|  'S-NNN kabul et' to accept."
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_product_owner_pipeline(task: AgentTask) -> AgentResponse:
    """
    Execute the Product Owner pipeline.

    ROUTING (decided by _is_planning_request):
      Planning prompt  → POLLMAgent (decompose_goal + generate_stories + persist)
      Command prompt   → ProductOwnerAgent (rule-based, deterministic)

    Both paths then apply:
      C1 gate — PII leak detection
      C4 gate — hallucination marker sweep
    """
    query = task.masked_input

    # ── Route decision ────────────────────────────────────────────────────────
    if _is_planning_request(query):
        response = _run_llm_planning(query, task)
    else:
        response = _agent.handle_query(query)

    # ── C1: PII leak ──────────────────────────────────────────────────────────
    if _PII_RE.search(response):
        return AgentResponse(
            task_id=task.task_id,
            agent_role=AgentRole.PRODUCT_OWNER,
            draft=(
                "⚠ Product Owner çıktısında kişisel veri tespit edildi. "
                "Bu bilgi paylaşılamaz."
            ),
            status=TaskStatus.FAILED,
        )

    # ── C4: Hallucination markers ─────────────────────────────────────────────
    if _HALLUCINATION_RE.search(response):
        response = _HALLUCINATION_RE.sub("[doğrulanmamış]", response)

    # ── Emit ──────────────────────────────────────────────────────────────────
    return AgentResponse(
        task_id=task.task_id,
        agent_role=AgentRole.PRODUCT_OWNER,
        draft=response,
        status=TaskStatus.COMPLETED,
    )
