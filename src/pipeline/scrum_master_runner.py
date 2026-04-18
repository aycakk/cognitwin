"""pipeline/scrum_master_runner.py — Hybrid Scrum Master pipeline.

Architecture
────────────
The pipeline has two paths that share the same entry point:

  DETERMINISTIC PATH (state-mutating or structured-lookup intents)
  ────────────────────────────────────────────────────────────────
  Intents: assign, add_task, update_task, promote_story, set_goal,
           sprint_status, standup, blockers, delegate,
           retrospective, review

  Flow: ScrumMasterAgent rule engine → C1 (PII) + C4 (hallucination) → emit

  These intents modify sprint state or produce precise structured output
  that must not be altered by an LLM.  Rule-based behaviour is preserved
  exactly as before.

  LLM-AUGMENTED PATH (analytical / open-ended intents)
  ─────────────────────────────────────────────────────
  Intents: sprint_analysis, general

  Flow:
    1. Sprint state context  — SprintStateStore.read_context_block()
    2. Ontology context      — build_scrum_master_ontology_context()
                               (agile.ttl + scrum_master.ttl via SPARQL)
    3. Rule engine output    — ScrumMasterAgent (for sprint_analysis only;
                               provides structured risk signals as grounding)
    4. LLM synthesis         — llama3.2 with SM system prompt
    5. C1 + C4 safety gates  — same as deterministic path

  The LLM receives:
    • SPRINT STATE   : real task/blocker/assignment data
    • ONTOLOGY       : SprintHealthStatus, RiskSignal, ImpedimentCategory
                       and RemediationAction vocabulary from scrum_master.ttl
    • RULE OUTPUT    : deterministic risk signals (sprint_analysis only)

  This grounds the LLM in actual sprint data and Scrum framework semantics
  while preserving all state-mutating operations as deterministic rules.

Gate coverage
─────────────
  C1 — PII leak detection    (all paths, same regex as original)
  C4 — Hallucination markers (all paths, same regex as original)
  C2/C3/C5/C6/C7/C8 are NOT applied here:
    C2/C7 require retrieval-based context (ChromaDB), not used in SM path
    C3    is student-ontology compliance (not applicable)
    C5    role-permission boundary check (SM has no restricted boundaries)
    C6    anti-sycophancy ASP (not wired into SM prompts)
    C8    jargon/length (not applicable to sprint management output)

ChromaDB isolation
──────────────────
This module imports ONLY from:
  • src.agents.scrum_master_agent
  • src.core.schemas
  • src.pipeline.scrum_team.sprint_state_store
  • src.ontology.loader  (no ChromaDB dependency)
  • ollama

This keeps the runner independent of the vector-memory singletons
instantiated in src/pipeline/shared.py.
"""

from __future__ import annotations

import logging
import re

from ollama import chat as _ollama_chat

from src.agents.scrum_master_agent import ScrumMasterAgent
from src.core.schemas import AgentTask, AgentResponse, AgentRole, TaskStatus
from src.ontology.loader import build_scrum_master_ontology_context
from src.pipeline.scrum_team.sprint_state_store import SprintStateStore

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Singletons
# ─────────────────────────────────────────────────────────────────────────────

_SPRINT_STATE = SprintStateStore()
_agent        = ScrumMasterAgent(state_store=_SPRINT_STATE)

# ─────────────────────────────────────────────────────────────────────────────
#  Gate patterns (local copies — no import from shared to keep the runner
#  self-contained and independent of the vector-memory singletons)
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
#  LLM-augmented intent set
#  All other intents stay on the deterministic rule path.
# ─────────────────────────────────────────────────────────────────────────────

_LLM_INTENTS: frozenset[str] = frozenset({"sprint_analysis", "general"})

# ─────────────────────────────────────────────────────────────────────────────
#  Scrum Master LLM system prompt
# ─────────────────────────────────────────────────────────────────────────────

_SM_LLM_SYSTEM_PROMPT = """\
████████████████████████████████████████████████████████████████████████████
          SCRUM MASTER AGENT — ANALYTIC ENGINE v1.0
          Ground sources : Sprint State JSON + scrum_master.ttl + agile.ttl
          Gate coverage  : C1 (PII) · C4 (Hallucination)
████████████████████████████████████████████████████████████████████████████

SECTION 0 ▸ AGENT IDENTITY
══════════════════════════════════════════════════════════
Sen deneyimli bir Scrum Master AI ajanısın.
İki yetkili bilgi kaynağın var:

  • SPRINT STATE    → sprint_state.json'dan alınan gerçek zamanlı veri:
                      görevler, atamalar, engeller, sprint hedefi.
  • ONTOLOGY       → scrum_master.ttl + agile.ttl:
                      SprintHealthStatus, RiskSignal, ImpedimentCategory
                      ve RemediationAction tanım ve kural tabanı.

TEMEL KURAL: Yanıtların YALNIZCA bu iki kaynaktan türetilmeli.

SECTION 1 ▸ RESPONSE PROTOCOL
══════════════════════════════════════════════════════════
• Risk analizi     → Ontoloji'deki RiskSignal bireylerini ve severityLevel
                     değerlerini kullanarak önceliklendir.
• Sprint sağlığı   → SprintHealthStatus (healthy / at_risk / critical)
                     ile mevcut sprint durumunu karşılaştır.
• Engel analizi    → ImpedimentCategory tiplerinden en uygununu öner.
• Öneri            → RemediationAction'dan somut aksiyon seç ve açıkla.
• Tüm öneriler spesifik ve uygulanabilir olmalı
  (örn. "T-003'ü developer-02'ye devret", "bugün standup'ta T-001 öncelikli").
• Türkçe yanıt ver.
• Kanıtsız tahmin YAPMA.
  Sprint state'te bilgi yoksa "mevcut veride bu bilgi yok" de.
████████████████████████████████████████████████████████████████████████████
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sm_chat(messages: list[dict]) -> str:
    """
    Lightweight Ollama wrapper for the Scrum Master LLM path.

    Returns the stripped content string.  Handles both the ChatResponse
    object (ollama ≥ 0.2) and the legacy dict shape (older versions).
    """
    resp = _ollama_chat(
        model="llama3.2",
        messages=messages,
        options={"temperature": 0.15, "top_p": 0.9},
    )
    if hasattr(resp, "message"):
        return resp.message.content.strip()
    if isinstance(resp, dict):
        return resp.get("message", {}).get("content", "").strip()
    return str(resp).strip()


def _apply_safety_gates(task: AgentTask, response: str) -> AgentResponse:
    """
    Apply C1 (PII) and C4 (hallucination) gates to any response string.

    Shared by both the deterministic and LLM paths so gate behaviour is
    identical regardless of which path produced the response.
    """
    # C1 — PII leak
    if _PII_RE.search(response):
        return AgentResponse(
            task_id=task.task_id,
            agent_role=AgentRole.SCRUM_MASTER,
            draft=(
                "⚠ Scrum Master çıktısında kişisel veri tespit edildi. "
                "Bu bilgi paylaşılamaz."
            ),
            status=TaskStatus.FAILED,
        )

    # C4 — Hallucination marker sweep
    if _HALLUCINATION_RE.search(response):
        response = _HALLUCINATION_RE.sub("[doğrulanmamış]", response)

    return AgentResponse(
        task_id=task.task_id,
        agent_role=AgentRole.SCRUM_MASTER,
        draft=response,
        status=TaskStatus.COMPLETED,
    )


def _run_llm_augmented(task: AgentTask, query: str, intent: str) -> AgentResponse:
    """
    LLM-augmented path for analytical and open-ended SM queries.

    Context assembly:
      • sprint_context   — real sprint state (tasks, blockers, assignments)
      • ontology_context — SM ontology vocabulary from scrum_master.ttl
      • rule_block       — structured risk signals from rule engine
                           (only for sprint_analysis; general uses state only)

    The LLM synthesises all three into a grounded, actionable response.
    """
    # 1. Sprint state context (read-only, no lock needed)
    sprint_context = _SPRINT_STATE.read_context_block()

    # 2. Ontology context (SPARQL over agile.ttl + scrum_master.ttl)
    ontology_context = build_scrum_master_ontology_context()

    # 3. Rule engine base output for sprint_analysis (provides deterministic
    #    risk signals as structured grounding input to the LLM)
    rule_block = ""
    if intent == "sprint_analysis":
        rule_output = _agent.handle_query(query)
        rule_block  = f"KURAL MOTORU RİSK ANALİZİ:\n{rule_output}\n\n"

    # 4. Build LLM message
    user_message = (
        f"{sprint_context}\n\n"
        f"{ontology_context}\n\n"
        f"{rule_block}"
        f"SORU: {query}\n\n"
        "INSTRUCTION: Yukarıdaki SPRINT STATE ve SCRUM MASTER ONTOLOGY CONTEXT "
        "bilgilerini kullanarak Scrum Master perspektifinden yanıt ver. "
        "Risk sinyallerini ontoloji tanımlarıyla eşleştir. "
        "SprintHealthStatus'a göre sprint sağlığını değerlendir. "
        "Somut ve uygulanabilir RemediationAction öner. "
        "Kural motoru çıktısı varsa onu temel al ve zenginleştir. "
        "Türkçe yanıt ver. Kanıtsız tahmin yapma."
    )

    base_messages = [
        {"role": "system", "content": _SM_LLM_SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    logger.debug("scrum-llm: intent=%r query=%r", intent, query[:80])

    # 5. LLM call
    draft = _sm_chat(base_messages)

    # 6. Safety gates (C1 + C4) — same as deterministic path
    return _apply_safety_gates(task, draft)


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_scrum_master_pipeline(task: AgentTask) -> AgentResponse:
    """
    Execute the hybrid Scrum Master pipeline.

    Routing decision
    ────────────────
    intent in _LLM_INTENTS  →  LLM-augmented path (_run_llm_augmented)
    all other intents        →  Deterministic rule path (ScrumMasterAgent)

    Both paths apply C1 + C4 safety gates before emitting.
    """
    query  = task.masked_input
    intent = _agent.detect_intent(query)

    # ── Deterministic path ────────────────────────────────────────────────────
    if intent not in _LLM_INTENTS:
        response = _agent.handle_query(query)
        return _apply_safety_gates(task, response)

    # ── LLM-augmented path ────────────────────────────────────────────────────
    return _run_llm_augmented(task, query, intent)
