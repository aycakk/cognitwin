"""pipeline/developer_runner.py — developer pipeline path (Scrum team).

Part of the Scrum team domain.  The developer role is collaborative
and role-based — it reads shared sprint/task state from SprintStateStore
and uses injected codebase context, but does NOT use:
  - Personal footprint signals
  - Personal developer profile / DeveloperProfileStore
  - developer_id as a person identity key

Sprint state writes are the exclusive responsibility of
ScrumMasterAgent via the scrum_master_runner path.

Path note: this file lives at src/pipeline/, so Path(__file__).resolve()
.parents[2] resolves to the project root.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ollama import chat

from src.agents.developer_orchestrator import DeveloperOrchestrator
from src.core.schemas import AgentTask, AgentResponse, AgentRole, TaskStatus
from src.gates.evaluator import evaluate_all_gates
from src.pipeline.redo import run_redo_loop
from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
from src.pipeline.shared import (
    VECTOR_MEM,
    VECTOR_TOP_K,
    BLINDSPOT_TRIGGERS,
    build_blindspot_block,
    _safe_chat,
    _sanitize_output,
)

# Shared sprint state.
# Task lifecycle writes (start / complete / block) originate here.
# All other sprint state writes (add task, assign, update status) remain the
# exclusive responsibility of ScrumMasterAgent via scrum_master_runner.
_SPRINT_STATE = SprintStateStore()

# ─────────────────────────────────────────────────────────────────────────────
#  TASK LIFECYCLE — intent detection + rule-based handlers
#  These are resolved before the DeveloperOrchestrator so that task
#  management commands return deterministic responses without an LLM round-trip.
# ─────────────────────────────────────────────────────────────────────────────

_TASK_ID_RE = re.compile(r"\b(T-\d+)\b", re.I)

_TASK_LIFECYCLE_INTENTS: list[tuple[str, re.Pattern]] = [
    ("show_tasks", re.compile(
        r"görevlerim|görev(?:leri?)?\s*(?:listele|göster|neler)"
        r"|my\s*tasks?|show\s+(?:my\s+)?tasks?|list\s+(?:my\s+)?tasks?"
        r"|assigned\s+tasks?|atanan\s+görev",
        re.I,
    )),
    # "T-001 başlat" / "start T-001"
    ("start_task", re.compile(
        r"(?:\bT-\d+\b.*\b(?:başlat|start)\b|\b(?:başlat|start)\b.*\bT-\d+\b)",
        re.I,
    )),
    # "T-001 tamamlandı: özet" / "complete T-001: summary"
    ("complete_task", re.compile(
        r"(?:\bT-\d+\b.*\b(?:tamamla(?:ndı)?|complet(?:e|ed)?|bitti?|done)\b"
        r"|\b(?:tamamla(?:ndı)?|complet(?:e|ed)?)\b.*\bT-\d+\b)",
        re.I,
    )),
    # "T-001 engellendi: sebep" / "block T-001: reason"
    ("block_task", re.compile(
        r"(?:\bT-\d+\b.*\b(?:engelle?(?:ndi)?|block(?:ed)?|tıkandı)\b"
        r"|\b(?:engelle?|block)\b.*\bT-\d+\b)",
        re.I,
    )),
]


def _detect_task_lifecycle_intent(query: str) -> str | None:
    """Return the first matching task-lifecycle intent, or None."""
    for intent_name, pattern in _TASK_LIFECYCLE_INTENTS:
        if pattern.search(query):
            return intent_name
    return None


def _handle_task_lifecycle(intent: str, query: str) -> str:
    """
    Execute a task lifecycle command and return a plain-text response.

    Mirrors the rule-based pattern used by ScrumMasterAgent and
    ProductOwnerAgent — deterministic, no LLM, no REDO loop.
    Assignee is always 'developer-default' because the developer path
    is role-based (no personal identity key).
    """
    if intent == "show_tasks":
        tasks = _SPRINT_STATE.get_tasks_for_assignee("developer-default")
        if not tasks:
            return (
                "Şu an atanmış aktif göreviniz yok.\n"
                "Scrum Master'dan görev atanmasını isteyin."
            )
        lines = [f"Atanmış Görevler ({len(tasks)}):"]
        for t in tasks:
            lines.append(
                f"  [{t['id']}] {t.get('title', '-')}"
                f"  | durum: {t.get('status', '?')}"
                f"  | öncelik: {t.get('priority', '?')}"
            )
            if t.get("blocker"):
                lines.append(f"    ⚠ Engel: {t['blocker']}")
        lines.append(
            "\nKomutlar:\n"
            "  T-NNN başlat                    — görevi başlat\n"
            "  T-NNN tamamlandı: <özet>        — görevi tamamla\n"
            "  T-NNN engellendi: <sebep>       — engeli kaydet"
        )
        return "\n".join(lines)

    task_match = _TASK_ID_RE.search(query)
    if not task_match:
        return "Görev ID'si gerekli (örn. T-001)."
    task_id = task_match.group(1).upper()

    if intent == "start_task":
        if _SPRINT_STATE.start_task(task_id):
            return (
                f"Görev başlatıldı\n"
                f"  ID    : {task_id}\n"
                f"  Durum : in_progress\n"
                f"  Not   : Tamamlandığında -> '{task_id} tamamlandı: <özet>'"
            )
        return f"Görev bulunamadı: {task_id}"

    if intent == "complete_task":
        summary_match = re.search(
            r"(?:tamamla(?:ndı)?|complet(?:e|ed)?|bitti?|done)[:\s]+(.+)",
            query, re.I | re.DOTALL,
        )
        summary = summary_match.group(1).strip() if summary_match else ""
        if _SPRINT_STATE.complete_task(task_id, result_summary=summary):
            msg = (
                f"Görev tamamlandı\n"
                f"  ID    : {task_id}\n"
                f"  Durum : done\n"
            )
            if summary:
                msg += f"  Özet  : {summary[:120]}\n"
            msg += "  Not   : PO inceleme için bilgilendirildi (ready_for_review)."
            return msg
        return f"Görev bulunamadı: {task_id}"

    if intent == "block_task":
        reason_match = re.search(
            r"(?:engelle?(?:ndi)?|block(?:ed)?|tıkandı)[:\s]+(.+)",
            query, re.I | re.DOTALL,
        )
        reason = reason_match.group(1).strip() if reason_match else "sebep belirtilmedi"
        if _SPRINT_STATE.block_task(task_id, reason=reason):
            return (
                f"Görev engellendi\n"
                f"  ID    : {task_id}\n"
                f"  Durum : blocked\n"
                f"  Sebep : {reason[:120]}\n"
                "  Not   : Scrum Master engeli çözmeli."
            )
        return f"Görev bulunamadı: {task_id}"

    return "Bilinmeyen komut."


# ─────────────────────────────────────────────────────────────────────────────
#  DEVELOPER WORKFLOW SYSTEM PROMPT
#  Used ONLY when called from agile_workflow (_step_developer).
#  Bypasses DeveloperOrchestrator (codebase-oriented) and produces
#  scenario-specific technical sprint tasks from the SM plan.
# ─────────────────────────────────────────────────────────────────────────────

_DEVELOPER_WORKFLOW_SYSTEM = """\
Sen bir Scrum takımında çalışan Developer Agent'sın.
Scrum Master sprint planındaki Sprint-1 hikayelerini teknik olarak gerçekleştirirsin.

ÖNEMLİ: Aşağıdaki örnek farklı bir senaryo (e-ticaret platformu) için hazırlanmıştır.
Sen KULLANICININ SM SPRINT PLANINI kullanarak tamamen ÖZGÜN içerik üretmelisin.
Örnek içeriği kopyalama — sadece FORMAT'ı kullan.

--- FORMAT ÖRNEĞİ (e-ticaret senaryosu) ---

## 💻 Developer Teknik Değerlendirme — Sprint-1

### [S-001] Ürün Listeleme için Teknik Görevler
**Teknik Açıklama:** PostgreSQL'den ürün listesi çeken paginated REST endpoint
**Uygulama Adımları:**
1. `Product` SQLAlchemy modeli + migration oluştur (`id`, `name`, `price`, `stock`, `created_at`)
2. `GET /api/v1/products` endpoint yaz — `?page=` + `?limit=` query param desteği
3. Sayfalama mantığını `offset/limit` ile implemente et, boş sayfa için 200+[] döndür
**Bağımlılıklar:** SQLAlchemy 2.x, Alembic migration, FastAPI QueryParam
**Tahmini Efor:** 6 saat
**Teknoloji Yığını:** Python 3.11, FastAPI, SQLAlchemy, PostgreSQL

### [S-002] Sepete Ekleme için Teknik Görevler
**Teknik Açıklama:** Session bazlı sepet yönetimi + stok kontrolü
**Uygulama Adımları:**
1. `Cart` + `CartItem` modelleri oluştur, `session_id` ile sepet ilişkilendir
2. `POST /api/v1/cart/items` endpoint — body: `{product_id, quantity}`
3. Stok kontrolünü `SELECT FOR UPDATE` ile transaction içinde yap (race condition önleme)
**Bağımlılıklar:** S-001 tamamlanmış olmalı (Product modeli gerekli)
**Tahmini Efor:** 8 saat
**Teknoloji Yığını:** Python 3.11, FastAPI, SQLAlchemy, PostgreSQL

## ⏭️ Kapsam Dışı — Sonraki Sprint
- Ödeme sistemi entegrasyonu (Stripe/iyzico) — açıkça ertelendi, bu sprint dışında
- Kargo takip modülü — sonraki sprint

## ⚠️ Teknik Risk ve Bağımlılıklar
- S-001 bitmeden S-002 başlayamaz (Product modeli bağımlılığı)
- Stok race condition: pessimistic lock zorunlu, optimistic lock yeterli değil

--- FORMAT ÖRNEĞİ SONU ---

KURALLAR:
1. Yukarıdaki örnek E-TİCARET senaryosuna ait. Sen SM planındaki GERÇEK Sprint-1
   hikayelerini kullan — örnek hikaye adlarını kopyalama.
2. SADECE "Sprint: Sprint-1" veya SM planındaki ana hikaye listesindeki S-NNN
   hikayelerini işle. "Sprint Scope Dışı" / "Kapsam Dışı" bölümündeki özellikler
   için teknik görev KESİNLİKLE ÜRETME.
3. Ödeme / payment sistemi Sprint-1 teknik görevlerine KESİNLİKLE dahil edilemez.
   Ödeme varsa sadece "⏭️ Kapsam Dışı" bölümünde bir satır yaz, teknik adım üretme.
4. T-NNN görev ID'si KULLANMA — bu yeni bir proje, stale ID'ler geçersizdir.
5. Bu yeni bir proje — mevcut kod dosyalarına ve eski task listesine referans verme.
6. Teknoloji varsayımı: Python 3.11, FastAPI, SQLAlchemy/PostgreSQL, JWT auth.
7. Türkçe yanıt ver.
8. Her teknik adım spesifik, uygulanabilir ve ölçülebilir olsun.
9. "Bunu hafızamda bulamadım" DEME — SM sprint planı ve PO hikayeleri sana verildi.
10. "Sprint Akış Yönetimi" bir kullanıcı hikayesi değildir — teknik görev olarak ekleme.
"""

# Developer-specific system prompt — the developer path must NOT use
# the Student Agent system prompt (SYSTEM_PROMPT in shared.py).
_DEVELOPER_SYSTEM_PROMPT = """\
Sen bir Scrum takımında görev yapan CogniTwin Developer Agent'sın.
Rol tabanlı çalışırsın — kişisel profil veya dijital ayak izi kullanmazsın.

Bağlam kaynakların:
  • Proje kaynak kodu       (CODEBASE CONTEXT)
  • Sprint durumu           (SPRINT CONTEXT)
  • Ontoloji kısıtları     (developer_ontology.ttl)
  • İş akışı bağlamı       (WORKFLOW CONTEXT) — Scrum Master sprint planı ve
                            Ürün Sahibi hikayeleri. Agile workflow modunda mevcuttur.

TEMEL KURALLAR (İhlal Edilemez):
  1. CODEBASE CONTEXT, SPRINT CONTEXT ve WORKFLOW CONTEXT içinde sağlanan bilgiden yanıt ver.
     WORKFLOW CONTEXT varsa (SM sprint planı / PO hikayeleri), o bağlamı birincil girdi olarak kullan.
  2. Bir dosya yolundan (path) bahsediyorsan, o dosya MUTLAKA CODEBASE CONTEXT içinde olmalı.
  3. Bir fonksiyon veya sınıf adından bahsediyorsan, o isim MUTLAKA CODEBASE CONTEXT içinde olmalı.
  4. CODEBASE CONTEXT ve WORKFLOW CONTEXT ikisi de boşsa, YALNIZCA şu yanıtı ver: "Bunu hafızamda bulamadım."
  5. Dosya yolu, sınıf adı, fonksiyon adı veya framework TAHMİN ETME — sadece bağlamda gördüklerini kullan.
  6. Bağlamda olmayan teknik detayı UYDURMA — halüsinasyon BLOKLANDI.
  7. PII ifşa etme.
  8. JavaScript, React, Node.js gibi projede OLMAYAN teknolojilere referans verme.
     Bu proje Python 3.11 + FastAPI + Ollama + ChromaDB + rdflib kullanır.

Yanıt verirken:
  - WORKFLOW CONTEXT (SM sprint planı) mevcutsa, o plana göre teknik uygulama adımları üret.
  - Bahsettiğin her dosya yolunu CODEBASE CONTEXT'ten doğrula.
  - Emin olmadığın teknik detayı "tahmin" olarak işaretle veya yanıt verme.
"""

# ─────────────────────────────────────────────────────────────────────────────
#  SPECIALIST BEHAVIOR PROFILES
#  Task-aware prompt blocks injected into Developer system content at runtime.
#  select_developer_specialist() is pure/deterministic — no LLM, no side effects.
#  Profiles are additive: _DEVELOPER_SYSTEM_PROMPT rules always apply underneath.
#  _REDO_SPECIALIST_BLOCK is appended to redo_rules only (not to system prompt).
# ─────────────────────────────────────────────────────────────────────────────

_SPECIALIST_PROFILES: dict[str, str] = {
    "default_developer": "",
    "minimal_change_engineer": (
        "SPECIALIST MODE — Minimal Change:\n"
        "- Implement ONLY what the task explicitly states. Do not expand scope.\n"
        "- If you notice something outside the task, note it as a follow-up — do not implement it.\n"
        "- Every claim must be directly justifiable by the task requirements.\n"
    ),
    "senior_developer_lite": (
        "SPECIALIST MODE — Implementation:\n"
        "- Analyze task requirements fully before producing output.\n"
        "- Reference the specific context source for each technical claim: "
        "\"(from codebase context)\", \"(from sprint context)\".\n"
        "- Produce complete, actionable steps — no deferred explanations.\n"
    ),
    "architect_lite": (
        "SPECIALIST MODE — Architecture:\n"
        "- Present at least two design options with explicit trade-offs.\n"
        "- Name what you are giving up, not only what you are gaining.\n"
        "- Every abstraction must justify its complexity.\n"
    ),
    "reality_checker_grounding": (
        "SPECIALIST MODE — Grounding:\n"
        "- Every file path, class name, and function name must appear in CODEBASE CONTEXT above.\n"
        "- If not found in provided context: emit BlindSpot — do not infer from general knowledge.\n"
        "- Default posture: uncertainty. Certainty requires explicit evidence in context.\n"
    ),
}

_REDO_SPECIALIST_BLOCK = (
    "Focus revision on the failing gate only:\n"
    "🔴 C4/C2_DEV: Remove any file path, function name, or claim not found in provided context.\n"
    "🔴 C1: Remove all unmasked PII.\n"
    "🟡 C3/C5: Remove out-of-scope or out-of-role content.\n"
    "Do not rewrite sections that passed gates.\n"
)

_SPECIALIST_TASK_RE: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(fix|bug|düzelt|hata|patch|repair|broken|bozuk|çöz)\b", re.I),
     "minimal_change_engineer"),
    (re.compile(r"\b(architect|design|tasarla|mimari|structure|nasıl\s+tasarlan)\b", re.I),
     "architect_lite"),
    (re.compile(r"\b(implement|add|ekle|geliştir|yeni\s+özellik|feature|create|oluştur)\b", re.I),
     "senior_developer_lite"),
]


def select_developer_specialist(
    query: str,
    codebase_context: str,
    workflow_context: str,  # noqa: ARG001 — reserved for future routing signals
) -> str:
    """Return a specialist behavior block for the given task.

    Pure and deterministic — no LLM call, no state writes.
    Returns an empty string for default_developer (no addition to system prompt).
    Codebase analysis queries only activate reality_checker_grounding when
    codebase_context is non-empty, preventing it from firing in workflow mode.
    """
    if codebase_context and re.search(
        r"\b(analiz|analyze|incele|how\s+does|where\s+is|ne\s+yapar)"
        r"|nasıl\s+çalış|nerede\b",
        query, re.I,
    ):
        return _SPECIALIST_PROFILES["reality_checker_grounding"]

    for pattern, profile_key in _SPECIALIST_TASK_RE:
        if pattern.search(query):
            return _SPECIALIST_PROFILES[profile_key]

    return _SPECIALIST_PROFILES["default_developer"]


# ─────────────────────────────────────────────────────────────────────────────
#  CODEBASE CONTEXT INJECTION
# ─────────────────────────────────────────────────────────────────────────────

# keyword → source file(s) to include when that keyword appears in the query
_DEV_CONTEXT_MAP: list[tuple[frozenset, str]] = [
    (frozenset({"routing", "route", "api", "endpoint", "librechat", "openai",
                "completions", "models", "v1"}),
     "src/services/api/openai_routes.py"),
    (frozenset({"pipeline", "gate", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8",
                "redo", "blindspot", "grounding", "architecture", "repository",
                "codebase", "this project", "this repo",
                "student path", "developer path", "both paths",
                "student and developer", "compare path",
                "branch", "branches", "worktree", "merge",
                "debug", "bug", "fix issue", "traceback", "exception",
                "stack trace", "error in"}),
     "src/services/api/pipeline.py"),
    (frozenset({"orchestrator", "developer orchestrator", "8-stage", "8 stage",
                "understand", "footprint", "profile"}),
     "src/agents/developer_orchestrator.py"),
    (frozenset({"developer agent", "role packet", "role library", "blueprint"}),
     "src/agents/developer_agent.py"),
    (frozenset({"student agent", "student path", "student pipeline", "studentagent"}),
     "src/agents/student_agent.py"),
    (frozenset({"chroma", "chromadb", "vector memory", "vector store", "memory backend"}),
     "src/database/chroma_manager.py"),
    (frozenset({"masker", "pii", "mask", "anonymi"}),
     "src/utils/masker.py"),
]

# chars read per file — enough for full small files, first section of large ones
_CODE_SNIPPET_CHARS = 4000


def _extract_out_of_scope(sm_output: str, po_output: str) -> str:
    """
    Scan SM and PO outputs for explicitly deferred / out-of-scope features.

    Returns a comma-separated string of out-of-scope items so the Developer
    LLM receives an unambiguous "do NOT work on these" list.  If nothing is
    found in the structured sections, falls back to keyword detection for
    common deferred features (payment, subscription, admin panel).

    This is a belt-and-suspenders complement to the system-prompt rules —
    it converts the natural-language "Sprint Scope Dışı" section into an
    explicit list that is embedded directly in the user prompt.
    """
    out_items: list[str] = []

    # Pattern: any out-of-scope section header followed by bullet lines.
    # Handles both SM format ("Sprint Scope Dışı") and PO format ("Kapsam Dışı").
    _OUT_SECTION = re.compile(
        r"(?:Sprint\s+Scope\s+Dışı|Kapsam\s+Dışı|Sonraki\s+Sprint|Out\s+of\s+Scope)"
        r"[^\n]*\n((?:[ \t]*[-•*]\s*.+\n?)*)",
        re.I,
    )
    for text in (sm_output, po_output):
        for m in _OUT_SECTION.finditer(text or ""):
            for line in m.group(1).strip().splitlines():
                item = re.sub(r"^\s*[-•*]\s*", "", line).strip()
                # strip parenthetical notes like "(açıkça ertelendi — ...)"
                item = re.sub(r"\s*\(.*?\)", "", item).strip()
                if item and len(item) > 3:
                    out_items.append(item)

    # Fallback: keyword detection for common deferred features not yet in
    # a structured section (covers cases where SM output is less structured).
    combined = (sm_output or "") + (po_output or "")
    _FALLBACK = [
        (re.compile(r"ödeme|payment|billing|fatural", re.I), "Ödeme / payment sistemi"),
        (re.compile(r"abonelik|subscription",          re.I), "Abonelik yönetimi"),
        (re.compile(r"admin\s*panel",                  re.I), "Admin paneli"),
        (re.compile(r"kargo|shipping",                 re.I), "Kargo / shipping"),
        (re.compile(r"analitik|analytics",             re.I), "Analitik modülü"),
    ]
    for pat, label in _FALLBACK:
        if pat.search(combined):
            # Only add via fallback if not already captured from the section scan
            already = any(pat.search(x) for x in out_items)
            if not already:
                out_items.append(label)

    # Deduplicate while preserving order; cap at 8 items
    seen: set[str] = set()
    unique: list[str] = []
    for item in out_items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return "; ".join(unique[:8])


def _build_workflow_context(task_ctx: dict) -> str:
    """Build a labeled context block from SM/PO outputs injected by agile_workflow.

    Returns an empty string when neither output is present so that the caller
    can use filter(None, ...) without any change to the join logic.
    """
    parts: list[str] = []
    po_output = (task_ctx.get("po_output") or "").strip()
    sm_output = (task_ctx.get("sm_output") or "").strip()
    if po_output and len(po_output) >= 15:
        parts.append(
            "=== ÜRÜN SAHİBİ HİKAYELERİ ===\n"
            f"{po_output}\n"
            "=== END PO HİKAYELERİ ==="
        )
    if sm_output and len(sm_output) >= 15:
        parts.append(
            "=== SCRUM MASTER SPRINT PLANI ===\n"
            f"{sm_output}\n"
            "=== END SPRINT PLANI ==="
        )
    return "\n\n".join(parts)


def _run_workflow_developer(task: AgentTask, workflow_context: str) -> AgentResponse:
    """Direct LLM path for Developer in agile workflow mode.

    Bypasses DeveloperOrchestrator (designed for codebase analysis / debug) and
    uses a workflow-specific system prompt to turn the SM sprint plan + PO stories
    into concrete technical tasks for the current scenario.

    Skips _validate_prose_result (which adds [NOT FOUND IN REPO] noise for new
    project paths that don't yet exist in the repo) and the REDO gate loop
    (which is calibrated for codebase queries, not sprint planning output).

    Scope boundary enforcement
    ──────────────────────────
    _extract_out_of_scope() scans both SM and PO outputs for explicitly deferred
    features and builds a concise list.  This list is embedded in the user prompt
    as an unambiguous "do NOT work on these" block so that the LLM cannot
    accidentally expand into out-of-scope items even when they appear in the
    context (e.g. in the SM "Sprint Scope Dışı" section or the original request).
    """
    task_ctx   = task.context or {}
    original   = task_ctx.get("original_request", task.masked_input)
    sm_output  = task_ctx.get("sm_output", "")
    po_output  = task_ctx.get("po_output", "")

    # ── Build explicit out-of-scope list from SM + PO outputs ────────────────
    out_of_scope = _extract_out_of_scope(sm_output, po_output)
    if out_of_scope:
        scope_block = (
            "\n\n⛔ KESİN KAPSAM SINIRI (MUTLAKA UYGULA):\n"
            "  Aşağıdaki özellikler Sprint-1 KAPSAMINDA DEĞİLDİR.\n"
            "  Bu özellikler için teknik uygulama görevi KESİNLİKLE ÜRETME.\n"
            "  Sadece '⏭️ Kapsam Dışı' bölümünde bir satırla belirt:\n"
            f"  → {out_of_scope}\n"
        )
    else:
        scope_block = ""

    user_prompt = (
        f"{workflow_context}"
        f"{scope_block}\n\n"
        f"PROJE İSTEĞİ: {original}\n\n"
        "GÖREV: Yukarıdaki Scrum Master sprint planındaki Sprint-1 hikayelerini "
        "(SM planının ana hikaye listesindeki S-NNN ID'leri) kullanarak her biri için "
        "somut teknik uygulama adımları üret.\n"
        "Sprint Scope Dışı / Kapsam Dışı olarak işaretlenmiş özellikler için "
        "teknik görev ÜRETME — sadece '⏭️ Kapsam Dışı' bölümünde listele.\n"
        "Ödeme / payment için hiçbir teknik adım yazma."
    )

    _wf_specialist_block = select_developer_specialist(
        original, codebase_context="", workflow_context=workflow_context
    )
    _wf_system_content = (
        _DEVELOPER_WORKFLOW_SYSTEM + "\n\n" + _wf_specialist_block
        if _wf_specialist_block else _DEVELOPER_WORKFLOW_SYSTEM
    )

    draft = ""
    try:
        resp  = _safe_chat(
            "llama3.2",
            [
                {"role": "system", "content": _wf_system_content},
                {"role": "user",   "content": user_prompt},
            ],
        )
        draft = (resp.get("message", {}).get("content", "") or "").strip()
    except Exception as exc:
        import logging as _log
        _log.getLogger(__name__).error("developer-workflow: LLM failed: %s", exc, exc_info=True)

    if not draft or len(draft.strip()) < 15:
        draft = "Developer iş akışı çıktısı üretilemedi."

    return AgentResponse(
        task_id=task.task_id,
        agent_role=AgentRole.DEVELOPER,
        draft=draft,
        status=TaskStatus.COMPLETED if len(draft.strip()) >= 15 else TaskStatus.FAILED,
    )


def _build_codebase_context(query: str) -> str:
    """
    Build a source-code context block for developer queries about this repo.

    Matches keywords in the query against a file map, reads the relevant
    source files (truncated), and returns a formatted context string.
    Returns an empty string when no codebase keywords are found so that
    the orchestrator's footprint logic is not bypassed unnecessarily.
    """
    text = query.lower()
    # developer_runner.py lives at src/pipeline/ → parents[2] is project root
    project_root = Path(__file__).resolve().parents[2]

    collected: dict[str, str] = {}   # rel_path → content, insertion-ordered dedup
    for keywords, rel_path in _DEV_CONTEXT_MAP:
        if any(kw in text for kw in keywords):
            if rel_path not in collected:
                abs_path = project_root / rel_path
                if abs_path.exists():
                    raw = abs_path.read_text(encoding="utf-8", errors="replace")
                    collected[rel_path] = raw[:_CODE_SNIPPET_CHARS]

    if not collected:
        return ""

    parts = ["=== CODEBASE CONTEXT (live source files) ==="]
    for rel_path, content in collected.items():
        parts.append(f"\n--- {rel_path} ---")
        parts.append(content)
        if len(content) >= _CODE_SNIPPET_CHARS:
            parts.append("... [truncated]")
    parts.append("\n=== END CODEBASE CONTEXT ===")
    return "\n".join(parts)




# ─────────────────────────────────────────────────────────────────────────────
#  DEBUG RESULT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def _validate_debug_result(parsed: dict, project_root: Path) -> dict:
    """
    Machine-verify the files and functions reported by the LLM.

    For every path in files_used: check the file exists under project_root.
    For every name in functions_used: check it appears as an identifier in
    the source of the cited files (simple text search — not AST).

    Returns the input dict extended with a '_validation' key and, if any
    files were invented, sets speculative=True.
    """
    files_used     = [str(f) for f in (parsed.get("files_used") or []) if f]
    functions_used = [str(f) for f in (parsed.get("functions_used") or []) if f]

    invalid_files     : list[str] = []
    unverified_funcs  : list[str] = []

    for rel_path in files_used:
        if not (project_root / rel_path).exists():
            invalid_files.append(rel_path)

    valid_files = [f for f in files_used if (project_root / f).exists()]
    for func_name in functions_used:
        found_in_any = False
        pattern = re.compile(rf"\b{re.escape(func_name)}\b")
        for rel_path in valid_files:
            try:
                content = (project_root / rel_path).read_text(
                    encoding="utf-8", errors="replace"
                )
                if pattern.search(content):
                    found_in_any = True
                    break
            except Exception:
                pass
        if not found_in_any:
            unverified_funcs.append(func_name)

    all_verified = not invalid_files and not unverified_funcs
    result = dict(parsed)
    result["_validation"] = {
        "invalid_files":    invalid_files,
        "unverified_funcs": unverified_funcs,
        "files_ok":         len(invalid_files) == 0,
        "functions_ok":     len(unverified_funcs) == 0,
        "all_verified":     all_verified,
    }
    if invalid_files:
        result["speculative"] = True   # invented paths → force speculative flag
    return result


def _validate_prose_result(draft: str, project_root: Path) -> str:
    """
    Validate file paths and function names mentioned in prose responses.

    Extracts file-path-like patterns and backtick-fenced identifiers from
    the LLM draft, checks them against the actual repository, and appends
    [NOT FOUND IN REPO] tags to any invented references.

    Returns the draft with validation tags injected.
    """
    # Extract file paths that look like project-relative paths
    file_pattern = re.compile(
        r'(?:src|tests|scripts|ontologies|data|infra)/[\w/]+\.(?:py|ttl|json|yaml|yml|toml|md)'
    )
    cited_files = file_pattern.findall(draft)

    # Extract function/class names from backtick-fenced code or def/class keywords
    ident_pattern = re.compile(
        r'(?:`(\w{3,})`)'                       # backtick-fenced identifiers
        r'|(?:(?:def|class)\s+(\w{3,}))'         # def/class declarations
    )
    cited_idents = [
        m.group(1) or m.group(2)
        for m in ident_pattern.finditer(draft)
        if m.group(1) or m.group(2)
    ]

    invalid_files: list[str] = []
    for rel_path in set(cited_files):
        if not (project_root / rel_path).exists():
            invalid_files.append(rel_path)

    # Check identifiers against valid cited files
    valid_files = [f for f in set(cited_files) if (project_root / f).exists()]
    unverified_idents: list[str] = []
    for ident in set(cited_idents):
        # Skip common Python builtins / keywords
        if ident in {"self", "None", "True", "False", "str", "int", "dict", "list",
                      "set", "tuple", "bool", "float", "print", "return", "import",
                      "from", "class", "def", "async", "await", "for", "while", "if"}:
            continue
        found = False
        for rel_path in valid_files:
            try:
                content = (project_root / rel_path).read_text(
                    encoding="utf-8", errors="replace"
                )
                if re.search(rf"\b{re.escape(ident)}\b", content):
                    found = True
                    break
            except Exception:
                pass
        if not found and valid_files:
            # Only flag if we have files to check against
            unverified_idents.append(ident)

    # Tag invalid file paths in the draft
    result = draft
    for inv_file in invalid_files:
        result = result.replace(inv_file, f"{inv_file} [NOT FOUND IN REPO]")

    # Append summary if issues found
    if invalid_files or unverified_idents:
        warnings = []
        if invalid_files:
            warnings.append(
                f"Doğrulanamayan dosya yolları: {', '.join(invalid_files)}"
            )
        if unverified_idents:
            warnings.append(
                f"Doğrulanamayan tanımlayıcılar: {', '.join(unverified_idents)}"
            )
        result += (
            "\n\n⚠ [Doğrulama Notu] Bazı referanslar repoda doğrulanamadı:\n  - "
            + "\n  - ".join(warnings)
        )

    return result


def _format_debug_result(v: dict) -> str:
    """
    Render a validated structured debug dict as clean, readable text.
    Marks any invented/unverified items explicitly so the user sees them.
    """
    val   = v.get("_validation", {})
    lines = []

    if v.get("entry_point"):
        lines.append(f"Entry Point: {v['entry_point']}")

    if v.get("files_used"):
        lines.append("\nFiles:")
        bad = set(val.get("invalid_files") or [])
        for f in v["files_used"]:
            tag = "  [NOT FOUND IN REPO]" if f in bad else ""
            lines.append(f"  - {f}{tag}")

    if v.get("functions_used"):
        lines.append("\nFunctions:")
        unv = set(val.get("unverified_funcs") or [])
        for fn in v["functions_used"]:
            tag = "  [NOT VERIFIED IN FILES]" if fn in unv else ""
            lines.append(f"  - {fn}{tag}")

    if v.get("execution_path"):
        lines.append("\nExecution Path:")
        for i, step in enumerate(v["execution_path"], 1):
            lines.append(f"  {i}. {step}")

    if v.get("suspected_root_cause"):
        lines.append(f"\nRoot Cause: {v['suspected_root_cause']}")

    if v.get("evidence"):
        lines.append("\nEvidence:")
        for e in v["evidence"]:
            lines.append(f"  - {e}")

    if v.get("fix"):
        lines.append(f"\nFix: {v['fix']}")

    conf = float(v.get("confidence") or 0.0)
    spec = bool(v.get("speculative", False))
    spec_tag = " (SPECULATIVE)" if spec else ""
    lines.append(f"\nConfidence: {conf:.2f}{spec_tag}")

    if not val.get("all_verified", True):
        lines.append(
            "\n[Validation] Some reported files or functions could not be "
            "verified in the repository. Treat this analysis with caution."
        )
    else:
        lines.append("\n[Validation] All reported files and functions verified in repository.")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  DEVELOPER PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def _process_developer_message(task: AgentTask) -> AgentResponse:
    """
    Developer path (Scrum team): DeveloperOrchestrator → gates → REDO.

    Role-based pipeline — no personal footprint, profile, or identity.
    Context sources:
      - Codebase snippets (keyword-matched from _DEV_CONTEXT_MAP)
      - Sprint state (read-only from SprintStateStore)
      - Ontology constraints (developer_ontology.ttl)

    Sprint state writes are the exclusive responsibility of
    ScrumMasterAgent via scrum_master_runner.  This path is read-only.
    """
    user_text     = task.masked_input
    strategy      = task.metadata.get("strategy", "auto")
    session_id    = task.session_id
    task_ctx      = task.context or {}
    workflow_step = task_ctx.get("workflow_step", "")
    # Acceptance criteria for C8: callers (e.g. sprint_loop) pass AC via
    # task.context so the developer's internal REDO loop evaluates C8 with
    # the real criteria. Interactive callers (main_cli) pass nothing → C8
    # trivially PASSes via evaluator.py early-return on empty list.
    task_ac: list[str] = list(task_ctx.get("acceptance_criteria") or [])

    # ── Pre-dispatch: task lifecycle commands ─────────────────────────────────
    # Lifecycle intent detection is SKIPPED in workflow mode.
    # The enriched dev_input created by agile_workflow._step_developer contains
    # phrases like "atanan görevleri teknik açıdan değerlendir" which match the
    # show_tasks pattern ("atanan görev") and would return stale sprint_state.json
    # data instead of analyzing the SM sprint plan.
    if not workflow_step:
        lifecycle_intent = _detect_task_lifecycle_intent(user_text)
        if lifecycle_intent:
            return AgentResponse(
                task_id=task.task_id,
                agent_role=AgentRole.DEVELOPER,
                draft=_handle_task_lifecycle(lifecycle_intent, user_text),
                status=TaskStatus.COMPLETED,
            )

    redo_log: list[dict] = []

    # Stage 1 — Retrieve vector context (used only by gate evaluators).
    # namespace="developer" keeps developer codebase snippets out of the
    # student academic_memory collection.
    vector_context, is_empty = VECTOR_MEM.retrieve(user_text, k=VECTOR_TOP_K, namespace="developer")

    # Stage 2 — DeveloperOrchestrator pipeline (role-based, no personal data)
    # Inject actual source file snippets so the LLM can answer codebase
    # questions rather than falling back to generic training knowledge.
    codebase_context = _build_codebase_context(user_text)

    # Stage 2a — Sprint context injection (read-only from SprintStateStore)
    # In workflow mode, sprint_state.json contains stale data from previous
    # sessions that is unrelated to the current project scenario.  Skip it
    # so the LLM focuses on workflow_context (SM plan + PO stories) instead.
    # NOTE: codebase_context is kept as a separate variable — Stage 2b below
    # gates JSON debug validation on it; that gate must not fire for sprint-only
    # queries where codebase_context is empty.
    if workflow_step == "developer":
        sprint_context = ""
    else:
        sprint_context = _SPRINT_STATE.read_context_block()

    # Stage 2b — Workflow context injection (SM sprint plan + PO stories).
    # Populated when this step is called from agile_workflow._step_developer.
    # Added LAST so it takes precedence over the generic sprint state when
    # both are present (the SM plan is project-specific; sprint_state is generic).
    workflow_context = _build_workflow_context(task.context or {})

    # ── Workflow fast-path: bypass DeveloperOrchestrator for agile workflow ──
    # DeveloperOrchestrator is calibrated for codebase analysis / debug queries.
    # In workflow mode, the developer's job is sprint-task generation from the
    # SM plan — a fundamentally different task.  The orchestrator's
    # _validate_prose_result would also add [NOT FOUND IN REPO] tags for new
    # project paths that don't yet exist in the repo, corrupting the output.
    if workflow_step == "developer" and workflow_context:
        return _run_workflow_developer(task, workflow_context)

    combined_context = "\n\n".join(filter(None, [codebase_context, sprint_context, workflow_context]))

    orchestrator = DeveloperOrchestrator(
        chat_fn=_safe_chat,
        default_model="llama3.2",
    )
    result = orchestrator.run(
        request=user_text,
        strategy=strategy,
        memory_context=combined_context,   # codebase + sprint context (not personal)
    )
    draft = _sanitize_output(str(result.get("solution", "")))
    if not draft:
        draft = "Bunu hafızamda bulamadım."

    # Stage 2b — Structured debug validation + prose grounding check
    # Validates cited files/functions against the actual repository.
    # Runs for ALL responses (not just when codebase_context is present)
    # to catch hallucinated paths from parametric memory.
    project_root = Path(__file__).resolve().parents[2]
    if codebase_context:
        try:
            parsed = json.loads(draft)
            if isinstance(parsed, dict) and "files_used" in parsed:
                validated = _validate_debug_result(parsed, project_root)
                draft     = _format_debug_result(validated)
            else:
                draft = _validate_prose_result(draft, project_root)
        except (json.JSONDecodeError, ValueError):
            # Prose response — validate cited file paths and identifiers
            draft = _validate_prose_result(draft, project_root)
    else:
        # No codebase context was injected — still validate any file
        # paths the LLM may have hallucinated from parametric memory.
        draft = _validate_prose_result(draft, project_root)

    # Stage 3 — Gate array + REDO loop
    # Uses the developer-specific system prompt (NOT the Student Agent prompt).
    # Specialist block is appended when task type matches a known profile.
    _specialist_block = select_developer_specialist(
        user_text, codebase_context, workflow_context
    )
    _system_content = (
        _DEVELOPER_SYSTEM_PROMPT + "\n\n" + _specialist_block
        if _specialist_block else _DEVELOPER_SYSTEM_PROMPT
    )
    base_messages = [
        {"role": "system", "content": _system_content},
        {"role": "user",   "content": user_text},
    ]

    draft, limit_hit = run_redo_loop(
        draft, base_messages, vector_context, is_empty, redo_log,
        agent_role=AgentRole.DEVELOPER,
        query=user_text,
        redo_rules=(
            "Answer ONLY from verified developer context. "
            "If not found: \"Bunu hafizamda bulamadim.\"\n"
            + _REDO_SPECIALIST_BLOCK
        ),
        limit_message_template=(
            "Dogrulama basarisiz (Gate {gate}). "
            "Yanit guvenli bicimde teslim edilemiyor.\n"
            "Bunu hafizamda bulamadim."
        ),
        post_process=_sanitize_output,
        gate_fn=evaluate_all_gates,
        chat_fn=chat,
        blindspot_fn=build_blindspot_block,
        session_id=session_id,
        gate_kwargs={
            "codebase_context":    combined_context,
            "acceptance_criteria": task_ac,
        },
    )
    if limit_hit:
        return AgentResponse(
            task_id=task.task_id,
            agent_role=AgentRole.DEVELOPER,
            draft=draft,
            status=TaskStatus.FAILED,
            redo_log=redo_log,
        )

    # Stage 4 — Emission
    if BLINDSPOT_TRIGGERS.search(draft):
        draft = build_blindspot_block(user_text) + draft

    return AgentResponse(
        task_id=task.task_id,
        agent_role=AgentRole.DEVELOPER,
        draft=draft,
        status=TaskStatus.COMPLETED,
        redo_log=redo_log,
    )
