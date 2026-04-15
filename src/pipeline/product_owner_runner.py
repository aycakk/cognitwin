"""pipeline/product_owner_runner.py — Product Owner pipeline path.

Rule-based pipeline.  Does NOT use ChromaDB or vector memory.
Grounds responses in local sprint state (data/sprint_state.json)
via SprintStateStore — specifically the ``backlog`` array.

Gate coverage (same as ScrumMaster path):
  C1  — PII leak detection
  C4  — Hallucination marker sweep

The ProductOwnerAgent produces a fully deterministic response from
local backlog state before any gate check is applied.
ProductOwnerAgent is the WRITE owner of the backlog array.
"""

from __future__ import annotations

import re

from src.agents.product_owner_agent import ProductOwnerAgent
from src.core.schemas import AgentTask, AgentResponse, AgentRole, TaskStatus
from src.pipeline.scrum_team.sprint_state_store import SprintStateStore

# Shared sprint state — Product Owner writes to backlog only.
_SPRINT_STATE = SprintStateStore()
_agent = ProductOwnerAgent(state_store=_SPRINT_STATE)

# ─────────────────────────────────────────────────────────────────────────────
#  Gate patterns (local copies — same as scrum_master_runner, kept
#  self-contained to avoid importing vector-memory singletons)
# ─────────────────────────────────────────────────────────────────────────────

_PII_RE = re.compile(
    r"\b\d{9,12}\b"                                        # TC / student ID
    r"|[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}"     # e-mail
    r"|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",                # phone
)

_HALLUCINATION_RE = re.compile(
    r"sanırım|galiba|muhtemelen|tahmin\s+etmek|belki|herhalde",
    re.I,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_product_owner_pipeline(task: AgentTask) -> AgentResponse:
    """
    Execute the Product Owner rule pipeline.

    Stage 1 — Rule-based processing (ProductOwnerAgent)
              Reads/writes local backlog state; no LLM, no ChromaDB.

    Stage 2 — C1 gate: PII leak check
              Blocks response if unmasked PII is present in the output.

    Stage 3 — C4 gate: hallucination marker sweep
              Replaces hedging phrases with a neutral marker.

    Stage 4 — Emission
              Returns the verified response.
    """
    query = task.masked_input

    # ── Stage 1 — Rule engine ─────────────────────────────────────────────
    response = _agent.handle_query(query)

    # ── Stage 2 — C1: PII leak ────────────────────────────────────────────
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

    # ── Stage 3 — C4: Hallucination markers ──────────────────────────────
    if _HALLUCINATION_RE.search(response):
        response = _HALLUCINATION_RE.sub("[doğrulanmamış]", response)

    # ── Stage 4 — Emission ────────────────────────────────────────────────
    return AgentResponse(
        task_id=task.task_id,
        agent_role=AgentRole.PRODUCT_OWNER,
        draft=response,
        status=TaskStatus.COMPLETED,
    )
