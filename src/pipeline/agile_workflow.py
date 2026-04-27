"""pipeline/agile_workflow.py — Sequential Agile team orchestration.

Architecture
────────────
Implements the full PO-first project delivery workflow:

  User → PO → [Composer gate] → SM → [Composer gate] → Dev → [Composer gate] → PO review → User

  STEP 1  Product Owner      — Receives raw project request.
                               Converts it into structured user stories with
                               acceptance criteria and priority.

  GATE 1  Composer           — Validates PO output before passing to SM.
                               No LLM. Checks usability, logs transition.

  STEP 2  Scrum Master       — Receives PO output as enriched context.
                               Plans the sprint, assigns tasks, evaluates
                               sprint health using the SM ontology and real
                               sprint state.

  GATE 2  Composer           — Validates SM output before passing to Developer.
                               No LLM. Checks usability, logs transition.

  STEP 3  Developer          — Receives SM sprint plan as enriched context.
                               Executes assigned tasks, provides technical
                               assessment and implementation steps.

  GATE 3  Composer           — Validates Developer output before PO review.
                               No LLM. Merges all collected outputs into a
                               team summary for the PO to review.

  STEP 4  Product Owner      — Final review: receives the merged team output
  (final review)               and presents a user-friendly delivery summary
                               in the PO's voice via LLM. Falls back to the
                               raw Composer merge if the LLM is unavailable.

Context passing
───────────────
Each step builds a fresh AgentTask whose masked_input is an enriched
prompt containing the previous agent's output.  The original query is
always preserved.  SprintStateStore is shared across all agents — no
extra wiring needed.

Entry points
────────────
  run_agile_workflow(task)        — called by composer_runner.py (Composer model)
                                    and by pipeline.py (PO model + workflow query)
  is_workflow_request(query)      — used by both callers to detect full-chain intent

SM intent routing
─────────────────
The SM prompt triggers the `sprint_analysis` intent (keywords: analiz,
değerlendir, öneri) which routes to the LLM-augmented path — giving SM
access to ontology context + sprint state.

PO intent routing
─────────────────
The PO prompt includes "kullanıcı hikayeleri oluştur" to trigger the
`create_story` deterministic intent regardless of original query phrasing.

Failure handling
────────────────
Each step is isolated in try/except.  If a step fails or returns unusable
output the workflow continues with what it has.  Gates fall back to the
original request context when the upstream agent produced nothing usable.
The final PO review falls back to the raw Composer merge on LLM failure.
"""

from __future__ import annotations

import logging
import re

from src.agents.composer_agent import ComposerAgent, HandoffResult
from src.core.schemas import AgentTask, AgentResponse, AgentRole, TaskStatus
from src.core.session_store import SESSION_STORE
from src.pipeline.developer_runner import _process_developer_message
from src.pipeline.product_owner_runner import run_product_owner_pipeline
from src.pipeline.scrum_master_runner import run_scrum_master_pipeline
from src.pipeline.shared import DEFAULT_MODEL

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Workflow trigger detection
# ─────────────────────────────────────────────────────────────────────────────

_WORKFLOW_RE = re.compile(
    r"proje\s+(?:başlat|planla|oluştur|yarat|kur|geliştir|teslim)"
    r"|yeni\s+proje"
    r"|sprint\s+(?:planla|başlat|organize\s+et)"
    r"|(?:tam|bütün|komple|eksiksiz)\s+(?:süreç|workflow|akış|sprint)"
    r"|agile\s+workflow"
    r"|(?:baştan\s+)?planla\s+ve\s+(?:geliştir|uygula|kodla|teslim\s+et)"
    r"|feature\s+request"
    r"|geliştirme\s+isteği"
    r"|proje\s+isteği"
    r"|end.to.end|u[çc]tan\s+uca"
    r"|tam\s+sprint"
    r"|sıfırdan\s+(?:planla|başlat|geliştir)"
    r"|proje\s+(?:teslim|delivery|deliverable)"
    r"|ekibi\s+(?:çalıştır|organize\s+et|koordine\s+et)"
    r"|tüm\s+ajanlar[ıi]"
    r"|sırayla\s+çalıştır"
    r"|takım[ıi]\s+(?:çalıştır|organize)",
    re.I,
)


def is_workflow_request(query: str) -> bool:
    """Return True when the query calls for a full PO→SM→Dev chain."""
    return bool(_WORKFLOW_RE.search(query or ""))


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _effective_session_id(task: AgentTask) -> str:
    return task.session_id or task.task_id


def _agent_session_id(parent_session: str, role_slug: str) -> str:
    return f"{parent_session}/{role_slug}"


def _make_child_task(
    parent: AgentTask,
    role: AgentRole,
    enriched_input: str,
    agent_session_id: str,
    extra_context: dict | None = None,
) -> AgentTask:
    ctx = {**parent.context}
    if extra_context:
        ctx.update(extra_context)
    return AgentTask(
        session_id=agent_session_id,
        role=role,
        masked_input=enriched_input,
        parent_task_id=parent.task_id,
        context=ctx,
        metadata={**parent.metadata},
    )


def _extract_draft(response: AgentResponse) -> str:
    draft = getattr(response, "draft", None)
    return str(draft).strip() if draft else ""


def _usable(text: str) -> bool:
    return bool(text) and len(text.strip()) >= 15


# ─────────────────────────────────────────────────────────────────────────────
#  Composer gate — lightweight hop validator (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _composer_gate(
    from_agent: str,
    text: str,
    to_agent: str,
    parent_session: str,
) -> HandoffResult:
    """Explicit Composer checkpoint between agent handoffs.

    No LLM call — just validates the outgoing text and logs the transition
    so that Composer is visibly present at every hop in the orchestration.
    """
    composer = ComposerAgent()
    result = composer.validate_handoff(from_agent, text, to_agent)
    logger.info(
        "agile-workflow: composer-gate  %s → %s  ok=%s  reason=%s  session=%s",
        from_agent, to_agent, result.ok, result.reason, parent_session,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  PO workflow system prompt
#  Used ONLY in agile_workflow mode — bypasses generic POLLMAgent prompts so
#  the PO produces scenario-specific, structured stories with sprint scope.
# ─────────────────────────────────────────────────────────────────────────────

_PO_WORKFLOW_SYSTEM = """\
Sen deneyimli bir Ürün Sahibi (Product Owner) AI ajanısın.
Kullanıcının proje senaryosunu okuyarak yapılandırılmış kullanıcı hikayeleri üretirsin.

ÖNEMLİ: Aşağıdaki örnek farklı bir senaryo (blog platformu) için hazırlanmıştır.
Sen kullanıcının GERÇEK senaryosunu kullanarak tamamen ÖZGÜN içerik üretmelisin.
Örnek içeriği kopyalama — sadece FORMAT'ı kullan.

--- FORMAT ÖRNEĞİ (blog platformu senaryosu) ---

## Proje Hedefi
Yazarların makale yayınlayabildiği, okuyucuların yorum yapabildiği minimal bir blog platformu geliştirmek.

## Kapsam
**İlk Sprint (Sprint-1):**
- Yazar kaydı ve girişi
- Makale oluşturma ve listeleme

**Kapsam Dışı / Sonraki Sprint:**
- Reklam ve para kazanma sistemi (açıkça ertelendi — sonraki sprint)
- E-posta bildirim servisi

## Epikler ve Kullanıcı Hikayeleri

### Epic 1: Yazar Yönetimi
**[S-001] Yazar Kaydı** | Öncelik: HIGH | Sprint: Sprint-1
> Kullanıcı olarak e-posta ve şifre ile yazar hesabı açmak istiyorum, çünkü makale yayınlayabilmek istiyorum.
Kabul Kriterleri:
- [ ] E-posta ve şifre ile kayıt formu çalışır
- [ ] Aynı e-posta ile ikinci kayıt reddedilir
- [ ] Kayıt başarılıysa JWT token döner

**[S-002] Yazar Girişi** | Öncelik: HIGH | Sprint: Sprint-1
> Kullanıcı olarak hesabıma giriş yapmak istiyorum, çünkü makalelerimi yönetmek istiyorum.
Kabul Kriterleri:
- [ ] Doğru kimlik bilgileriyle giriş yapılır
- [ ] Hatalı şifrede 401 hatası döner
- [ ] Başarılı girişte JWT access token döner

### Epic 2: İçerik Yönetimi
**[S-003] Makale Oluşturma** | Öncelik: HIGH | Sprint: Sprint-1
> Kullanıcı olarak başlık ve içerikle yeni makale yayınlamak istiyorum, çünkü fikirlerimi paylaşmak istiyorum.
Kabul Kriterleri:
- [ ] Başlık zorunlu (maks. 120 karakter)
- [ ] Makale kaydedildiğinde zaman damgası oluşturulur
- [ ] Sadece giriş yapmış yazarlar makale ekleyebilir

**[S-004] Makale Listeleme** | Öncelik: MEDIUM | Sprint: Sprint-1
> Kullanıcı olarak tüm makaleleri listelemek istiyorum, çünkü son içeriklere göz atmak istiyorum.
Kabul Kriterleri:
- [ ] Makaleler tarihe göre azalan sırada listelenir
- [ ] Her makalede başlık, yazar ve tarih gösterilir
- [ ] Sayfalama (10 makale/sayfa) desteklenir

--- FORMAT ÖRNEĞİ SONU ---

KURALLAR:
1. Yukarıdaki örnek BLOG PLATFORMU içindir. Sen kullanıcının senaryosuna göre tamamen yeni içerik yaz.
2. Kapsam ipuçlarındaki "Sprint-1'de olacaklar" listesine kesinlikle uy — sadece bu özellikler için hikaye üret.
3. "Kapsam Dışı" listesindeki özellikler için Sprint: İleride yaz ve Kapsam Dışı bölümüne ekle.
4. Her hikayeye S-001, S-002, ... şeklinde benzersiz ID ver — kullanıcıdan ID isteme.
5. 4-8 arası hikaye üret; her biri 3-5 kabul kriteri içersin.
6. Türkçe yanıt ver.
7. Sadece Proje Hedefi, Kapsam ve Epikler/Hikayeler bölümlerini yaz — ek açıklama ekleme.
8. KESİNLİKLE YASAK: "hikaye ID gerekli", "hangi hikayenin kabul kriterlerini", "ID girin" gibi
   ifadeler KULLANMA — ID'leri sen üretirsin, kullanıcıdan ID isteme.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Scope hint extractor — pre-parses the user query so the PO LLM gets explicit
#  in/out-of-scope signals instead of having to infer them from natural language.
# ─────────────────────────────────────────────────────────────────────────────

_SCOPE_IN_PATTERNS = [
    (re.compile(r"kay[ıi]t|register|sign.?up",        re.I), "Kullanıcı kaydı"),
    (re.compile(r"giri[sş]|login|sign.?in|oturum",    re.I), "Kullanıcı girişi / oturum açma"),
    (re.compile(r"görev|task(?!.*payment)|to.?do",     re.I), "Görev yönetimi"),
    # NOTE: "sprint akışı" is an agile process concern, NOT a user story.
    # Removed from in-scope to prevent it from appearing as Sprint-1 item.
    (re.compile(r"profil|profile",                     re.I), "Kullanıcı profili"),
    (re.compile(r"bildirim|notif",                     re.I), "Bildirim sistemi"),
    (re.compile(r"dashboard|kontrol\s*panel",          re.I), "Dashboard / kontrol paneli"),
    (re.compile(r"arama|search",                       re.I), "Arama işlevi"),
]

_SCOPE_OUT_PATTERNS = [
    (re.compile(r"öde(me|n)|payment|billing|fatural", re.I), "Ödeme sistemi"),
    (re.compile(r"abonelik|subscription",             re.I), "Abonelik yönetimi"),
    (re.compile(r"raporla|report(?!.*task)",          re.I), "Raporlama modülü"),
    (re.compile(r"analitik|analytics",                re.I), "Analitik / istatistik"),
    (re.compile(r"mobil\s*uygulama|mobile\s*app",     re.I), "Mobil uygulama"),
    (re.compile(r"admin\s*panel",                     re.I), "Admin paneli"),
]

_DEFER_SIGNAL_RE = re.compile(
    r"sonraki\s*sprint|ileride|kals[ıi]n|ertelensin|next\s*sprint|later|defer",
    re.I,
)


def _extract_scope_hints(query: str) -> str:
    """Return an explicit scope block to prepend to po_input.

    Scans the original query for feature keywords and deferral signals,
    then builds a concise Turkish hint block so the PO LLM doesn't have
    to infer sprint scope from ambiguous natural language.
    """
    in_scope: list[str]  = []
    out_scope: list[str] = []

    for _pat, label in _SCOPE_OUT_PATTERNS:
        if _pat.search(query):
            out_scope.append(label)

    for _pat, label in _SCOPE_IN_PATTERNS:
        if _pat.search(query):
            in_scope.append(label)

    if not in_scope and not out_scope:
        return ""

    lines = ["[KAPSAM İPUÇLARI — LLM tarafından çıkarıldı, kesinlikle uy]"]
    if in_scope:
        lines.append("Sprint-1 kapsamında OLMASI GEREKEN özellikler:")
        for f in in_scope:
            lines.append(f"  - {f}")
    if out_scope:
        lines.append("Kapsam DIŞI / Sonraki Sprint'e ertelenen özellikler:")
        for f in out_scope:
            lines.append(f"  - {f} (Sprint: İleride — Kapsam Dışı bölümüne yaz)")
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  PO template fallback — used only when ALL LLM attempts fail.
#  Builds a minimal but SM-usable structured output directly from scope hints
#  so the Scrum Master never receives blank/English/command-mode output.
# ─────────────────────────────────────────────────────────────────────────────

def _build_po_template_fallback(original: str, scope_hints: str) -> str:
    """Return a minimal Türkçe structured PO output when the LLM is unavailable.

    Derives in-scope features from _SCOPE_IN_PATTERNS and out-of-scope
    features from _SCOPE_OUT_PATTERNS so the content is always tied to the
    actual user query rather than being a generic template.
    """
    in_feats  = [label for pat, label in _SCOPE_IN_PATTERNS  if pat.search(original)]
    out_feats = [label for pat, label in _SCOPE_OUT_PATTERNS if pat.search(original)]

    # If no features detected at all, use a minimal single-epic output
    if not in_feats:
        in_feats = ["Temel uygulama işlevleri"]

    # Build sprint-1 scope block
    scope_lines = "\n".join(f"- {f}" for f in in_feats)
    out_lines   = (
        "\n".join(f"- {f} (sonraki sprint'e ertelendi)" for f in out_feats)
        if out_feats else "- (Henüz belirlenmedi)"
    )

    # Build one story per detected in-scope feature
    stories: list[str] = []
    for i, feat in enumerate(in_feats, start=1):
        sid = f"S-{i:03d}"
        stories.append(
            f"**[{sid}] {feat}** | Öncelik: HIGH | Sprint: Sprint-1\n"
            f"> Kullanıcı olarak {feat.lower()} özelliğini kullanmak istiyorum, "
            f"çünkü projenin temel işlevine ihtiyacım var.\n"
            f"Kabul Kriterleri:\n"
            f"- [ ] Özellik başarıyla çalışır\n"
            f"- [ ] Hata durumunda anlaşılır mesaj gösterilir\n"
            f"- [ ] Kimlik doğrulama gerektiren işlemler korunur\n"
        )

    return (
        f"## Proje Hedefi\n"
        f"{original[:200].strip()}\n\n"
        f"## Kapsam\n"
        f"**İlk Sprint (Sprint-1):**\n{scope_lines}\n\n"
        f"**Kapsam Dışı / Sonraki Sprint:**\n{out_lines}\n\n"
        f"## Epikler ve Kullanıcı Hikayeleri\n\n"
        f"### Epic 1: Temel Özellikler\n"
        + "\n".join(stories)
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1 — Product Owner
# ─────────────────────────────────────────────────────────────────────────────

def _step_product_owner(
    parent: AgentTask,
    parent_session: str,
) -> tuple[AgentResponse, str, str]:
    """Convert the raw project request into structured user stories.

    Workflow mode uses a dedicated PO system prompt (_PO_WORKFLOW_SYSTEM) via
    a direct LLM call instead of run_product_owner_pipeline.  The standard
    pipeline's generic POLLMAgent prompts do not carry sprint scope constraints
    (in_scope / out_of_scope / sprint assignment), so they produce generic output
    that causes the SM and Developer to drift into template-like behavior.

    Falls back to run_product_owner_pipeline if the direct LLM call fails.
    """
    original    = parent.masked_input
    po_sid      = _agent_session_id(parent_session, "po")
    scope_hints = _extract_scope_hints(original)

    po_input = (
        f"Proje Senaryosu:\n\n{original}\n\n"
        + (f"{scope_hints}\n" if scope_hints else "")
        + "Yukarıdaki senaryoyu ve kapsam ipuçlarını dikkate alarak "
        + "yapılandırılmış kullanıcı hikayeleri ve kabul kriterleri oluştur."
    )

    SESSION_STORE.create_session(
        session_id=po_sid,
        agent_role=AgentRole.PRODUCT_OWNER.value,
        query=po_input,
        parent_session_id=parent_session,
        metadata={"workflow_step": "product_owner"},
    )

    task = _make_child_task(
        parent, AgentRole.PRODUCT_OWNER, po_input, po_sid,
        extra_context={"workflow_step": "product_owner", "original_request": original},
    )

    # Workflow PO: direct LLM call with scenario-specific system prompt.
    # This bypasses run_product_owner_pipeline's generic POLLMAgent so the
    # output explicitly includes sprint scope, story IDs, and out_of_scope marks.
    draft = ""
    try:
        from src.pipeline.shared import _safe_chat  # lazy — avoids ollama at module load
        resp  = _safe_chat(
            DEFAULT_MODEL,
            [
                {"role": "system", "content": _PO_WORKFLOW_SYSTEM},
                {"role": "user",   "content": po_input},
            ],
        )
        draft = (resp.get("message", {}).get("content", "") or "").strip()
        logger.info(
            "agile-workflow: PO workflow LLM OK  session=%s  (%d chars)",
            po_sid, len(draft),
        )
    except Exception as exc:
        logger.error(
            "agile-workflow: PO workflow LLM failed: %s — retrying with minimal prompt", exc,
            exc_info=True,
        )

    # Attempt 2: simpler prompt if primary call failed or returned empty.
    # Avoids falling to the generic English POLLMAgent pipeline.
    if not _usable(draft):
        try:
            from src.pipeline.shared import _safe_chat  # noqa: re-import inside guard is fine
            in_feats_str = ", ".join(
                label for pat, label in _SCOPE_IN_PATTERNS if pat.search(original)
            )
            simple_user = (
                f"Proje: {original}\n\n"
                + (f"Sprint-1 kapsamı: {in_feats_str}\n" if in_feats_str else "")
                + "Bu proje için Proje Hedefi, Kapsam ve Epikler/Hikayeler bölümlerini üret."
            )
            resp2 = _safe_chat(
                DEFAULT_MODEL,
                [
                    {"role": "system", "content": _PO_WORKFLOW_SYSTEM},
                    {"role": "user",   "content": simple_user},
                ],
            )
            draft = (resp2.get("message", {}).get("content", "") or "").strip()
            if _usable(draft):
                logger.info(
                    "agile-workflow: PO retry LLM OK  session=%s  (%d chars)",
                    po_sid, len(draft),
                )
        except Exception as exc2:
            logger.error(
                "agile-workflow: PO retry LLM also failed: %s — using template fallback", exc2,
            )

    # Last resort: structured template built from scope hints so SM still gets
    # usable input instead of generic English POLLMAgent output.
    if not _usable(draft):
        logger.warning("agile-workflow: PO using template fallback  session=%s", po_sid)
        draft = _build_po_template_fallback(original, scope_hints)

    response = AgentResponse(
        task_id=task.task_id,
        agent_role=AgentRole.PRODUCT_OWNER,
        draft=draft,
        status=TaskStatus.COMPLETED if _usable(draft) else TaskStatus.FAILED,
    )

    SESSION_STORE.record_output(
        session_id=po_sid,
        output=draft,
        status=response.status.value if hasattr(response.status, "value") else str(response.status),
        metadata={"task_id": task.task_id},
    )
    return response, draft, po_sid


# ─────────────────────────────────────────────────────────────────────────────
#  Step 2 — Scrum Master
# ─────────────────────────────────────────────────────────────────────────────

def _step_scrum_master(
    parent: AgentTask,
    po_output: str,
    parent_session: str,
) -> tuple[AgentResponse, str, str]:
    """Plan the sprint and assign tasks based on PO output.

    PO stories are passed via task.context["po_output"] → po_section (first in
    the SM user message).  sm_input is intentionally lean — it carries only the
    original request and the sprint-planning directive so the PO output is not
    duplicated inside the user message, which would waste context window on
    small LLMs like llama3.2.
    """
    original = parent.masked_input
    sm_sid   = _agent_session_id(parent_session, "sm")
    sm_input = (
        f"Proje isteği: {original}\n\n"
        f"Yukarıdaki PO hikayeleri doğrultusunda Sprint-1 için sprint planı oluştur: "
        f"görev atamaları, story point tahminleri, risk ve engelleri belirt."
    )

    SESSION_STORE.create_session(
        session_id=sm_sid,
        agent_role=AgentRole.SCRUM_MASTER.value,
        query=sm_input,
        parent_session_id=parent_session,
        metadata={"workflow_step": "scrum_master"},
    )

    task = _make_child_task(
        parent, AgentRole.SCRUM_MASTER, sm_input, sm_sid,
        extra_context={
            "workflow_step": "scrum_master",
            "po_output": po_output,
            "original_request": original,
        },
    )
    response = run_scrum_master_pipeline(task)
    draft    = _extract_draft(response)

    SESSION_STORE.record_output(
        session_id=sm_sid,
        output=draft,
        status=response.status.value if hasattr(response.status, "value") else str(response.status),
        metadata={"task_id": task.task_id},
    )
    return response, draft, sm_sid


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3 — Developer
# ─────────────────────────────────────────────────────────────────────────────

def _step_developer(
    parent: AgentTask,
    sm_output: str,
    po_output: str,
    parent_session: str,
) -> tuple[AgentResponse, str, str]:
    """Execute assigned tasks based on the SM sprint plan.

    The dev_input carries an explicit sprint-scope boundary so the Developer
    LLM cannot drift into out-of-scope features (e.g. payment) that appear
    in the SM "Sprint Scope Dışı" or PO "Kapsam Dışı" sections.
    The detailed scope enforcement is also repeated inside developer_runner.py
    (_extract_out_of_scope + _DEVELOPER_WORKFLOW_SYSTEM rules) for redundancy.
    """
    original  = parent.masked_input
    dev_sid   = _agent_session_id(parent_session, "dev")
    dev_input = (
        f"Proje isteği: {original}\n\n"
        f"Scrum Master sprint planı ve görev atamaları:\n{sm_output}\n\n"
        f"Referans — Ürün Sahibi hikayeleri:\n{po_output}\n\n"
        "GÖREV: Yukarıdaki sprint planındaki SADECE Sprint-1 kapsamındaki "
        "S-NNN hikayelerini işle. Her hikaye için somut teknik uygulama adımları, "
        "bağımlılıklar ve efor tahmini üret.\n"
        "KESİNLİKLE YASAK: 'Sprint Scope Dışı' veya 'Kapsam Dışı / Sonraki Sprint' "
        "olarak işaretlenmiş özellikler (ödeme/payment dahil) için teknik görev üretme. "
        "Bu özellikler sadece 'Kapsam Dışı' bölümünde listelenebilir.\n"
        "T-NNN eski görev ID'si kullanma — bu tamamen yeni bir proje sprint'idir."
    )

    SESSION_STORE.create_session(
        session_id=dev_sid,
        agent_role=AgentRole.DEVELOPER.value,
        query=dev_input,
        parent_session_id=parent_session,
        metadata={"workflow_step": "developer"},
    )

    task = _make_child_task(
        parent, AgentRole.DEVELOPER, dev_input, dev_sid,
        extra_context={
            "workflow_step": "developer",
            "sm_output": sm_output,
            "po_output": po_output,
            "original_request": original,
        },
    )
    task.metadata["strategy"] = task.metadata.get("strategy", "auto")

    response = _process_developer_message(task)
    draft    = _extract_draft(response)

    SESSION_STORE.record_output(
        session_id=dev_sid,
        output=draft,
        status=response.status.value if hasattr(response.status, "value") else str(response.status),
        metadata={"task_id": task.task_id},
    )
    return response, draft, dev_sid


# ─────────────────────────────────────────────────────────────────────────────
#  Team-output builder for PO final review
#  Preserves newlines so the PO review LLM receives properly formatted Markdown.
#  ComposerAgent.compose() collapses whitespace (_clean_text) which destroys
#  Markdown structure — this function is used instead for the PO review input.
# ─────────────────────────────────────────────────────────────────────────────

def _build_team_raw_context(po_text: str, sm_text: str, dev_text: str) -> str:
    """Concatenate agent outputs with section labels, preserving newlines.

    Intended ONLY for the PO final review LLM prompt — gives the model a
    readable, structure-preserving view of all three agent outputs rather
    than the whitespace-collapsed text produced by ComposerAgent.compose().
    """
    parts: list[str] = []
    if _usable(po_text):
        parts.append(
            f"=== ÜRÜN SAHİBİ HİKAYELERİ ===\n{po_text}\n=== END PO ==="
        )
    if _usable(sm_text):
        parts.append(
            f"=== SCRUM MASTER SPRINT PLANI ===\n{sm_text}\n=== END SM ==="
        )
    if _usable(dev_text):
        parts.append(
            f"=== DEVELOPER TEKNİK DEĞERLENDİRME ===\n{dev_text}\n=== END DEV ==="
        )
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
#  Step 4 — PO Final Review
# ─────────────────────────────────────────────────────────────────────────────

_PO_REVIEW_SYSTEM = (
    "Sen bir Ürün Sahibi (Product Owner) olarak çalışıyorsun.\n"
    "Agile geliştirme ekibinin ürettiği teknik çıktıyı alıp kullanıcıya\n"
    "anlaşılır, iş odaklı bir proje teslim özeti sunuyorsun.\n\n"
    "KURALLAR:\n"
    "- Teknik jargonu kullanıcı diline çevir.\n"
    "- Tamamlanan iş kalemlerini (epic, story, görev) net biçimde özetle.\n"
    "- Sprint planı ve geliştirici değerlendirmesini entegre et.\n"
    "- Varsa açık maddeler ve önerilen sonraki adımları belirt.\n"
    "- Kısa, profesyonel ve kullanıcı odaklı bir dil kullan.\n"
    "- Markdown kullanabilirsin (başlıklar, liste).\n"
    "- Türkçe yanıt ver.\n"
)


def _format_po_fallback(team_output: str, original: str) -> str:
    """Plain-text PO delivery summary when the LLM is unavailable."""
    return (
        "## Proje Teslim Özeti\n\n"
        f"**Kullanıcı İsteği:** {original}\n\n"
        f"**Agile Ekip Çıktısı:**\n\n{team_output}\n\n"
        "---\n"
        "_Not: Ürün Sahibi LLM özeti üretilemedi. Ekip çıktısı doğrudan sunulmaktadır._"
    )


def _build_visible_workflow_output(
    po_text: str,
    sm_text: str,
    dev_text: str,
    po_review_text: str,
    warnings: list[str],
) -> str:
    """Build the final LibreChat-visible response with labeled agent sections.

    Each agent's contribution appears under its own Markdown heading so the
    user can see every step of the workflow in a single response message.
    The PO final review is always placed last as the delivery summary.
    """
    divider = "\n\n---\n\n"
    parts: list[str] = []

    if _usable(po_text):
        parts.append(f"## 📋 Ürün Sahibi — Proje Hikayeleri\n\n{po_text}")

    if _usable(sm_text):
        parts.append(f"## 🏃 Scrum Master — Sprint Planı\n\n{sm_text}")

    if _usable(dev_text):
        parts.append(f"## 💻 Developer — Teknik Değerlendirme\n\n{dev_text}")

    if _usable(po_review_text):
        parts.append(f"## ✅ Ürün Sahibi — Teslim Özeti\n\n{po_review_text}")

    if warnings:
        warn_block = "\n".join(f"- {w}" for w in warnings)
        parts.append(f"## ⚠️ Uyarılar\n\n{warn_block}")

    return divider.join(parts) if parts else "Workflow tamamlanamadı."


def _step_po_final_review(
    parent: AgentTask,
    team_output: str,
    parent_session: str,
) -> tuple[AgentResponse, str, str]:
    """PO reviews the merged team output and presents a user-friendly summary.

    Uses the LLM directly (_safe_chat) with a PO-persona system prompt.
    Falls back to a formatted plain-text summary if the LLM is unavailable.
    This is the last step before the response reaches the user — closing the
    loop: User → PO → ... → PO → User.
    """
    po_review_sid = _agent_session_id(parent_session, "po_review")
    original      = parent.masked_input

    user_prompt = (
        f"Kullanıcı isteği: {original}\n\n"
        f"Agile ekibinin ürettiği sonuç:\n{team_output}\n\n"
        "Yukarıdaki ekip çıktısını, kullanıcıya sunacağın bir Ürün Sahibi "
        "teslim özeti olarak yeniden yaz."
    )

    SESSION_STORE.create_session(
        session_id=po_review_sid,
        agent_role=AgentRole.PRODUCT_OWNER.value,
        query=user_prompt,
        parent_session_id=parent_session,
        metadata={"workflow_step": "po_final_review"},
    )

    draft = ""
    try:
        from src.pipeline.shared import _safe_chat  # lazy import — avoids ollama at module load
        resp  = _safe_chat(
            DEFAULT_MODEL,
            [
                {"role": "system", "content": _PO_REVIEW_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
        )
        draft = (resp.get("message", {}).get("content", "") or "").strip()
        logger.info(
            "agile-workflow: PO final review OK  session=%s  (%d chars)",
            po_review_sid, len(draft),
        )
    except Exception as exc:
        logger.error("agile-workflow: PO final review LLM failed: %s", exc, exc_info=True)
        draft = _format_po_fallback(team_output, original)

    SESSION_STORE.record_output(
        session_id=po_review_sid,
        output=draft,
        status="completed" if _usable(draft) else "failed",
        metadata={},
    )

    response = AgentResponse(
        task_id=parent.task_id,
        agent_role=AgentRole.PRODUCT_OWNER,
        draft=draft,
        status=TaskStatus.COMPLETED if _usable(draft) else TaskStatus.FAILED,
    )
    return response, draft, po_review_sid


# ─────────────────────────────────────────────────────────────────────────────
#  Main orchestration entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_agile_workflow(task: AgentTask) -> AgentResponse:
    """Execute the full PO-first Agile workflow.

    Flow: PO → [Composer gate] → SM → [Composer gate] → Dev → [Composer gate]
          → Composer merge → PO final review → User

    Composer is explicitly present at every agent handoff as a lightweight
    gate (no LLM per hop).  The final response comes from the PO, not the
    Composer, matching the real-world product-owner-led delivery model.

    Returns
    ───────
    AgentResponse with:
      • agent_role = PRODUCT_OWNER  (PO is the final voice to the user)
      • draft      = PO final review text (fallback: Composer merge)
      • status     = COMPLETED
      • metadata["workflow"] = step counts, agent list, child sessions, warnings
    """
    parent_session = _effective_session_id(task)

    logger.info(
        "agile-workflow: starting PO→[C]→SM→[C]→Dev→[C]→PO  task=%s  session=%s",
        task.task_id, parent_session,
    )

    SESSION_STORE.create_session(
        session_id=parent_session,
        agent_role=AgentRole.COMPOSER.value,
        query=task.masked_input,
        parent_session_id=None,
        metadata={"workflow": "agile_po_first"},
    )

    collected:      list[dict[str, str]] = []
    warnings:       list[str]            = []
    child_sessions: list[str]            = []

    # ── Step 1: Product Owner ─────────────────────────────────────────────────
    po_text = ""
    po_sid  = ""
    try:
        _, po_text, po_sid = _step_product_owner(task, parent_session)
        child_sessions.append(po_sid)
        if _usable(po_text):
            collected.append({"agent": AgentRole.PRODUCT_OWNER.value, "draft": po_text})
            logger.info("agile-workflow: PO OK  session=%s  (%d chars)", po_sid, len(po_text))
        else:
            warnings.append("Ürün Sahibi: kullanılabilir çıktı üretilemedi.")
            po_text = ""
    except Exception as exc:
        warnings.append(f"Ürün Sahibi adımı başarısız oldu: {exc}")
        po_text = ""
        logger.error("agile-workflow: PO step failed: %s", exc, exc_info=True)

    # ── Composer Gate 1: PO → SM ──────────────────────────────────────────────
    gate1 = _composer_gate(
        "ProductOwnerAgent", po_text, "ScrumMasterAgent", parent_session,
    )
    if not gate1.ok:
        warnings.append(gate1.reason)
    # Use po_text directly — gate1.payload runs _clean_text which collapses newlines,
    # destroying the Markdown structure the SM LLM needs to parse S-NNN story IDs.
    sm_context = po_text if gate1.ok else f"Proje isteği: {task.masked_input}"

    # ── Step 2: Scrum Master ──────────────────────────────────────────────────
    sm_text = ""
    sm_sid  = ""
    try:
        _, sm_text, sm_sid = _step_scrum_master(task, sm_context, parent_session)
        child_sessions.append(sm_sid)
        if _usable(sm_text):
            collected.append({"agent": AgentRole.SCRUM_MASTER.value, "draft": sm_text})
            logger.info("agile-workflow: SM OK  session=%s  (%d chars)", sm_sid, len(sm_text))
        else:
            warnings.append("Scrum Master: kullanılabilir çıktı üretilemedi.")
            sm_text = ""
    except Exception as exc:
        warnings.append(f"Scrum Master adımı başarısız oldu: {exc}")
        sm_text = ""
        logger.error("agile-workflow: SM step failed: %s", exc, exc_info=True)

    # ── Composer Gate 2: SM → Developer ──────────────────────────────────────
    gate2 = _composer_gate(
        "ScrumMasterAgent", sm_text, "DeveloperAgent", parent_session,
    )
    if not gate2.ok:
        warnings.append(gate2.reason)
    # Use sm_text directly — same whitespace-preservation reason as gate1 above.
    dev_context = sm_text if gate2.ok else sm_context

    # ── Step 3: Developer ─────────────────────────────────────────────────────
    dev_text = ""
    dev_sid  = ""
    try:
        _, dev_text, dev_sid = _step_developer(task, dev_context, po_text, parent_session)
        child_sessions.append(dev_sid)
        if _usable(dev_text):
            collected.append({"agent": AgentRole.DEVELOPER.value, "draft": dev_text})
            logger.info("agile-workflow: Dev OK  session=%s  (%d chars)", dev_sid, len(dev_text))
        else:
            warnings.append("Developer: kullanılabilir çıktı üretilemedi.")
            dev_text = ""
    except Exception as exc:
        warnings.append(f"Developer adımı başarısız oldu: {exc}")
        dev_text = ""
        logger.error("agile-workflow: Developer step failed: %s", exc, exc_info=True)

    # ── Composer Gate 3: Developer → PO final review ─────────────────────────
    gate3 = _composer_gate(
        "DeveloperAgent", dev_text, "ProductOwnerAgent (final review)", parent_session,
    )
    if not gate3.ok:
        warnings.append(gate3.reason)

    # ── Composer merge — intermediate team summary for PO review ─────────────
    composer = ComposerAgent()

    if not collected:
        error_text = composer.format_final_response(
            summary="Agile workflow tüm adımlarda başarısız oldu.",
            key_points=["Hiçbir ajan kullanılabilir çıktı üretemedi."],
            warnings=warnings or ["Bilinmeyen pipeline hatası."],
            final_answer=(
                "Agile workflow tamamlanamadı. "
                "Lütfen isteği yeniden belirtin veya sistem yöneticisiyle iletişime geçin."
            ),
        )
        SESSION_STORE.record_output(
            session_id=parent_session,
            output=error_text,
            status="failed",
            metadata={"child_sessions": child_sessions, "warnings": warnings},
        )
        return AgentResponse(
            task_id=task.task_id,
            agent_role=AgentRole.COMPOSER,
            draft=error_text,
            status=TaskStatus.FAILED,
            metadata={
                "workflow": {
                    "session_id":      parent_session,
                    "child_sessions":  child_sessions,
                    "steps_completed": 0,
                    "steps_attempted": 4,
                    "agents_used":     [],
                    "warnings":        warnings,
                }
            },
        )

    composed = composer.compose(collected)  # used for metadata (useful_count etc.) only

    # Build the PO review input as a structured, newline-preserving concatenation
    # of raw agent outputs instead of using composed["response_text"].
    # ComposerAgent._clean_text collapses all whitespace (including newlines) which
    # destroys the Markdown structure and makes the PO review LLM unable to
    # produce a coherent summary.  Raw text preserves headers, bullets, story IDs.
    team_summary = _build_team_raw_context(po_text, sm_text, dev_text)
    if not _usable(team_summary):
        # Last-resort fallback when all three agent outputs are empty
        team_summary = composed["response_text"]

    # ── Step 4: PO Final Review ───────────────────────────────────────────────
    po_review_text = ""
    po_review_sid  = ""
    try:
        _, po_review_text, po_review_sid = _step_po_final_review(
            task, team_summary, parent_session,
        )
        child_sessions.append(po_review_sid)
        if not _usable(po_review_text):
            warnings.append("PO final review: kullanılabilir çıktı üretilemedi, Composer çıktısı kullanılıyor.")
            po_review_text = team_summary
    except Exception as exc:
        warnings.append(f"PO final review adımı başarısız oldu: {exc} — Composer çıktısı kullanılıyor.")
        po_review_text = team_summary
        logger.error("agile-workflow: PO review step failed: %s", exc, exc_info=True)

    final_reviewer = "product_owner" if _usable(po_review_text) else "composer"

    # Build the visible response — each agent gets a labeled section so the
    # user can see PO stories, SM plan, Developer output, and PO review in
    # a single LibreChat message instead of just the final merged text.
    final_output = _build_visible_workflow_output(
        po_text=po_text,
        sm_text=sm_text,
        dev_text=dev_text,
        po_review_text=po_review_text if _usable(po_review_text) else "",
        warnings=warnings,
    )

    workflow_meta: dict = {
        "session_id":      parent_session,
        "child_sessions":  child_sessions,   # [po, sm, dev, po_review]
        "steps_completed": len(collected),
        "steps_attempted": 4,
        "agents_used":     [s["agent"] for s in collected],
        "final_reviewer":  final_reviewer,
        "warnings":        warnings,
        "useful_count":    composed.get("useful_count", 0),
        "merged_count":    composed.get("merged_count", 0),
        "conflict_count":  len(composed.get("conflicts", [])),
    }

    SESSION_STORE.record_output(
        session_id=parent_session,
        output=final_output,
        status="completed",
        metadata=workflow_meta,
    )

    logger.info(
        "agile-workflow: complete  session=%s  %d/3 agent steps  "
        "final_reviewer=%s  children=%s",
        parent_session, len(collected), final_reviewer, child_sessions,
    )

    return AgentResponse(
        task_id=task.task_id,
        agent_role=AgentRole.PRODUCT_OWNER,   # PO is the final voice to the user
        draft=final_output,
        status=TaskStatus.COMPLETED,
        metadata={"workflow": workflow_meta},
    )
