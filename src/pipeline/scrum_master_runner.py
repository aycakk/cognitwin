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
Model loading
─────────────
  LOCAL MODEL PATH (SCRUM_MODEL_PATH env var points to a directory)
  ──────────────────────────────────────────────────────────────────
  When SCRUM_MODEL_PATH is set and the directory exists, the runner loads
  the locally trained Scrum Master model using the HuggingFace Transformers
  pipeline.  The model is expected to be in AutoModelForCausalLM format
  (saved with model.save_pretrained / tokenizer.save_pretrained).

  OLLAMA FALLBACK (SCRUM_MODEL_PATH not set or directory missing)
  ───────────────────────────────────────────────────────────────
  When no local model is found, the runner falls back to llama3.2 via
  Ollama.  This keeps the pipeline functional during development even if
  the trained model has not been placed in models/scrum_model yet.

This module imports ONLY from:
  • src.agents.scrum_master_agent
  • src.core.schemas
  • src.pipeline.scrum_team.sprint_state_store
  • src.ontology.loader  (no ChromaDB dependency)
  • ollama         (Ollama fallback)
  • transformers   (local model — optional, only used when model dir exists)

This keeps the runner independent of the vector-memory singletons
instantiated in src/pipeline/shared.py.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from ollama import chat as _ollama_chat

# Path to the locally trained Scrum Master model, mounted via Docker volume.
# When the directory exists → local HuggingFace model is loaded.
# When missing or empty    → Ollama llama3.2 fallback is used.
_SCRUM_MODEL_PATH: str = os.environ.get("SCRUM_MODEL_PATH", "")

# Lazy-loaded transformers pipeline singleton.
# Populated on first LLM call when the local model directory is present.
_local_pipe: Any = None

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

_LLM_INTENTS: frozenset[str] = frozenset({"sprint_analysis", "sprint_planning", "general"})

# ─────────────────────────────────────────────────────────────────────────────
#  Scrum Master LLM system prompt
# ─────────────────────────────────────────────────────────────────────────────

_SM_LLM_SYSTEM_PROMPT = """\
████████████████████████████████████████████████████████████████████████████
          SCRUM MASTER AGENT — ANALYTIC ENGINE v1.1
          Ground sources : Sprint State JSON + scrum_master.ttl + agile.ttl
                           + PO Stories (workflow mode)
          Gate coverage  : C1 (PII) · C4 (Hallucination)
████████████████████████████████████████████████████████████████████████████

SECTION 0 ▸ AGENT IDENTITY
══════════════════════════════════════════════════════════
Sen deneyimli bir Scrum Master AI ajanısın.
Üç yetkili bilgi kaynağın var:

  • SPRINT STATE    → sprint_state.json'dan alınan gerçek zamanlı veri:
                      görevler, atamalar, engeller, sprint hedefi.
  • ONTOLOGY        → scrum_master.ttl + agile.ttl:
                      SprintHealthStatus, RiskSignal, ImpedimentCategory
                      ve RemediationAction tanım ve kural tabanı.
  • PO HİKAYELERİ  → Yeni proje senaryolarında Ürün Sahibi'nin ürettiği
                      kullanıcı hikayeleri, kabul kriterleri ve öncelik
                      sıralaması. Bu kaynak sprint planlama görevi için
                      birincil girdi olarak kullanılmalıdır.

TEMEL KURAL: Birden fazla kaynak mevcutsa hepsini entegre et.
PO hikayeleri varsa sprint planı ONLARA GÖRE yapılmalı — eski state'e
takılıp kalmadan.

SECTION 1 ▸ RESPONSE PROTOCOL
══════════════════════════════════════════════════════════
• Sprint planlama  → PO hikayelerindeki her story için sprint görevi oluştur.
                     Öncelik sırasını PO'nun belirlediği önceliğe göre yap.
                     Her göreve developer-default atama yap ve story point tahmin et.
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
  Mevcut veride bilgi yoksa "bu bilgi mevcut veride yok" de.
████████████████████████████████████████████████████████████████████████████
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_local_pipe():
    """
    Lazy-load the local Scrum Master model via transformers.pipeline.

    Called at most once.  Returns the pipeline on success, None on failure.
    The None return causes _sm_chat to fall back to Ollama automatically.
    """
    global _local_pipe
    if _local_pipe is not None:
        return _local_pipe
    try:
        from transformers import pipeline as hf_pipeline
        _local_pipe = hf_pipeline(
            "text-generation",
            model=_SCRUM_MODEL_PATH,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.15,
            top_p=0.9,
        )
        logger.info("scrum: local model loaded from %s", _SCRUM_MODEL_PATH)
    except Exception as exc:
        logger.error("scrum: local model load failed (%s) — falling back to Ollama", exc)
        _local_pipe = None
    return _local_pipe


def _sm_chat(messages: list[dict]) -> str:
    """
    Dual-mode LLM call for the Scrum Master hybrid path.

    Priority 1 — Local trained model (SCRUM_MODEL_PATH directory exists)
                 ONLY when the input fits within the model's context window.
      Calls the HuggingFace transformers pipeline loaded from the Docker
      volume mount at /app/models/scrum_model.

    Priority 2 — Ollama llama3.2 fallback in all other cases:
      • No local model directory.
      • Local model failed to load.
      • Input is too long for the local model (estimated token count >
        _LOCAL_MODEL_MAX_INPUT_TOKENS).  Agile-workflow prompts include PO
        stories + backlog + sprint state + ontology, easily exceeding 2048
        tokens.  Sending an oversized input causes "indexing errors" and
        produces garbage output — we skip the local model in that case.

    Returns the stripped response string in both cases.
    """
    # ── Priority 1: local HuggingFace model ──────────────────────────────────
    _LOCAL_MODEL_MAX_INPUT_TOKENS = 1400   # local model max_length=2048, reserve 512 for output + margin
    _CHARS_PER_TOKEN               = 4     # rough but reliable estimate for Turkish/English mixed text

    if _SCRUM_MODEL_PATH and os.path.isdir(_SCRUM_MODEL_PATH):
        pipe = _load_local_pipe()
        if pipe is not None:
            # Estimate total input token count from raw character length.
            # If the estimate exceeds the local model's safe input budget,
            # fall through to Ollama instead of causing indexing errors.
            total_chars = sum(len(str(m.get("content", ""))) for m in messages)
            estimated_tokens = total_chars // _CHARS_PER_TOKEN
            if estimated_tokens > _LOCAL_MODEL_MAX_INPUT_TOKENS:
                logger.info(
                    "scrum: input too long for local model (~%d tokens > %d limit) "
                    "— falling back to Ollama",
                    estimated_tokens, _LOCAL_MODEL_MAX_INPUT_TOKENS,
                )
            else:
                try:
                    result    = pipe(messages, max_new_tokens=512)
                    generated = result[0]["generated_text"]
                    if isinstance(generated, list):
                        # Chat-template output: list of messages, last is assistant
                        return generated[-1].get("content", "").strip()
                    return str(generated).strip()
                except Exception as exc:
                    logger.warning(
                        "scrum: local model inference failed (%s) — falling back to Ollama", exc,
                    )

    # ── Priority 2: Ollama fallback ───────────────────────────────────────────
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
    LLM-augmented path for analytical, planning, and open-ended SM queries.

    Context assembly:
      • po_section       — PO stories from task.context (workflow mode only)
      • backlog_context  — backlog items from sprint_state.json
      • sprint_context   — real sprint state (tasks, blockers, assignments)
      • ontology_context — SM ontology vocabulary from scrum_master.ttl
      • rule_block       — structured risk signals from rule engine
                           (only for sprint_analysis; planning/general skip this)

    For workflow sprint_planning intent, PO stories are placed FIRST so the
    LLM treats them as the primary planning input, not as a question to answer.
    """
    # 1. Extract PO output injected by agile_workflow._make_child_task
    task_ctx   = task.context or {}
    po_output  = task_ctx.get("po_output", "")
    wf_step    = task_ctx.get("workflow_step", "")

    po_section = ""
    if po_output and len(po_output.strip()) >= 15:
        po_section = (
            "=== ÜRÜN SAHİBİ HİKAYELERİ VE PROJE BRIEFİ ===\n"
            f"{po_output}\n"
            "=== END PO HİKAYELERİ ===\n\n"
        )

    # 2. Backlog context — shows stories PO just added (not yet in tasks array)
    backlog_raw = _SPRINT_STATE.read_backlog_context_block()
    backlog_context = backlog_raw if "(none)" not in backlog_raw else ""

    # 3. Sprint state context (tasks, assignments, blockers)
    sprint_context = _SPRINT_STATE.read_context_block()

    # 4. Ontology context (SPARQL over agile.ttl + scrum_master.ttl)
    ontology_context = build_scrum_master_ontology_context()

    # 5. Rule engine base output — only for sprint_analysis, not planning
    rule_block = ""
    if intent == "sprint_analysis":
        rule_output = _agent.handle_query(query)
        rule_block  = f"KURAL MOTORU RİSK ANALİZİ:\n{rule_output}\n\n"

    # 6. Build instruction — sprint_planning gets a concrete task-generation goal
    if intent == "sprint_planning" or wf_step == "scrum_master":
        instruction = (
            "GÖREV: Yukarıdaki PO hikayelerini ve backlog içeriğini kullanarak "
            "somut bir sprint planı üret. Her hikaye için:\n"
            "  1. Görev başlığı ve kısa açıklama yaz\n"
            "  2. Story point tahmini yap (1-8 arası)\n"
            "  3. developer-default'a atama yap\n"
            "  4. Kabul kriterlerini listele\n"
            "  5. Potansiyel risk veya engelleri belirt\n"
            "Backlog boşsa veya PO hikayeleri yoksa kullanıcı isteğinden "
            "çıkarım yaparak temsili bir sprint planı oluştur. "
            "Türkçe yanıt ver. 'Görev ID gerekli' deme — bu yeni bir proje planlamasıdır."
        )
    else:
        instruction = (
            "INSTRUCTION: Yukarıdaki SPRINT STATE, BACKLOG ve SCRUM MASTER ONTOLOGY "
            "bilgilerini kullanarak Scrum Master perspektifinden yanıt ver. "
            "Risk sinyallerini ontoloji tanımlarıyla eşleştir. "
            "SprintHealthStatus'a göre sprint sağlığını değerlendir. "
            "Somut ve uygulanabilir RemediationAction öner. "
            "Kural motoru çıktısı varsa onu temel al ve zenginleştir. "
            "Türkçe yanıt ver. Kanıtsız tahmin yapma."
        )

    user_message = "".join([
        po_section,
        f"{backlog_context}\n\n" if backlog_context else "",
        f"{sprint_context}\n\n",
        f"{ontology_context}\n\n",
        rule_block,
        f"KULLANICI İSTEĞİ: {query}\n\n",
        instruction,
    ])

    base_messages = [
        {"role": "system", "content": _SM_LLM_SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    logger.debug("scrum-llm: intent=%r wf_step=%r query=%r", intent, wf_step, query[:80])

    # 7. LLM call
    draft = _sm_chat(base_messages)

    # 8. Safety gates (C1 + C4) — same as deterministic path
    return _apply_safety_gates(task, draft)


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_scrum_master_pipeline(task: AgentTask) -> AgentResponse:
    """
    Execute the hybrid Scrum Master pipeline.

    Routing decision
    ────────────────
    workflow_step == "scrum_master"  →  LLM sprint_planning path (always)
                                        Bypasses keyword detection — agile_workflow
                                        already built an enriched prompt with PO output;
                                        the SM's only job here is sprint planning.
    intent in _LLM_INTENTS           →  LLM-augmented path (_run_llm_augmented)
    all other intents                →  Deterministic rule path (ScrumMasterAgent)

    Both paths apply C1 + C4 safety gates before emitting.
    """
    query       = task.masked_input
    task_ctx    = task.context or {}
    workflow_step = task_ctx.get("workflow_step", "")

    # ── Workflow fast-path: always plan the sprint when called from agile_workflow
    if workflow_step == "scrum_master":
        return _run_llm_augmented(task, query, "sprint_planning")

    intent = _agent.detect_intent(query)

    # ── Deterministic path ────────────────────────────────────────────────────
    if intent not in _LLM_INTENTS:
        response = _agent.handle_query(query)
        return _apply_safety_gates(task, response)

    # ── LLM-augmented path ────────────────────────────────────────────────────
    return _run_llm_augmented(task, query, intent)
