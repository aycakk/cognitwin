"""pipeline/hr_runner.py — HR / Recruiter agent pipeline runner.

Called by src/services/api/pipeline.py when model name contains 'hr'.

Pipeline stages:
  1. Intent detection (rule-based regex)
  2. Recruiter profile load
  3. Token budget check
  4. Prompt construction (HRAgent prompt builders)
  5. LLM call via Ollama
  6. Gate evaluation (C1 PII + C4 hallucination)
  7. Audit logging
  8. Response emission

No vector memory is used (HR works with data provided in the message).
No fine-tuned model — prompt engineering + structured outputs.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from ollama import chat

from src.core.schemas import AgentTask, AgentResponse, AgentRole, TaskStatus
from src.agents.hr_agent import HRAgent
from src.pipeline.hr.hr_schemas import (
    HRSessionContext,
    HRStructuredResponse,
    INTENT_AUTOMATION_MAP,
    TOKEN_ACTION_COSTS,
)
from src.pipeline.hr.recruiter_profile_store import (
    load_profile,
    save_profile,
    build_profile_summary,
    update_preferences,
    update_from_feedback,
)
from src.pipeline.hr.token_ledger import (
    load_ledger,
    check_and_deduct,
    budget_status_block,
    action_cost_table,
)
from src.pipeline.hr.audit_logger import log_action
from src.pipeline.hr.n8n_client import build_payloads, trigger_all, _is_enabled as _n8n_enabled

logger = logging.getLogger(__name__)

_agent = HRAgent()

# ── Session store (in-memory; sessions are short-lived) ──────────────────────
_hr_sessions: dict[str, HRSessionContext] = {}

_AUTOMATION_SIGNAL_RE = re.compile(
    r"\botomasyon\b|\bn8n\b|\bwebhook\b|tetikle|g[öo]nder|aktar|kaydet|bildir",
    re.I,
)

_ACTION_HINT_RE: dict[str, re.Pattern] = {
    "shortlist_to_sheets": re.compile(r"kısa liste|sheet|tablo|airtable|ats", re.I),
    "notify_slack": re.compile(r"slack|bildir|kanal", re.I),
    "send_outreach_email": re.compile(r"e.?posta|mail|işe davet|mesaj", re.I),
    "create_calendar_event": re.compile(r"m[üu]lakat|takvim|calendar|randevu", re.I),
    "log_to_ats": re.compile(r"ats|crm|işe alım sistemi|kaydet|log", re.I),
}


_AUTOMATION_TR: dict[str, str] = {
    "notify_slack":          "Slack bildirimi",
    "shortlist_to_sheets":   "Kısa liste tablosu",
    "send_outreach_email":   "İşe davet e-postası",
    "create_calendar_event": "Takvim etkinliği",
    "log_to_ats":            "Aday takip sistemine kayıt",
}


def _automation_note(actions: list[str]) -> str:
    labels = ", ".join(_AUTOMATION_TR.get(a, a) for a in actions)
    return f"---\nOTOMASYON: {labels} n8n otomasyonuna iletildi."


def _get_or_create_session(session_id: str, recruiter_id: str) -> HRSessionContext:
    if session_id not in _hr_sessions:
        _hr_sessions[session_id] = HRSessionContext(
            session_id=session_id,
            recruiter_id=recruiter_id,
        )
    return _hr_sessions[session_id]


# ── Intent detection ──────────────────────────────────────────────────────────

_INTENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Eksik yetkinlikler — önce kontrol et (pozisyon bağlamıyla çakışmayı önler)
    ("missing_skills",      re.compile(
        r"eksik yetkinlik|eksik beceri|ne eksik|hangi yetkinlik.*eksik"
        r"|eksikler neler|yetkinlik.*eksik|önemli eksik", re.I)),
    # Özgeçmiş / CV analizi
    ("cv_analyze",          re.compile(
        r"cv.*analiz|özgeçmiş.*incele|özgeçmiş.*değerlendir|cv.*değerlendir"
        r"|özgeçmiş.*analiz|bu cv|cv.*hakkında", re.I)),
    # İlan analizi
    ("req_parse",           re.compile(
        r"ilan.*analiz|pozisyon.*analiz|iş.*ilanı.*incele"
        r"|ilan.*değerlendir|bu ilan", re.I)),
    # Aday–pozisyon eşleştirme
    ("candidate_match",     re.compile(
        r"uygun mu|eşleştir|bu (cv|aday).*(ilan|pozisyon)"
        r"|bu pozisyon için.*aday|bu ilan için.*aday"
        r"|aday.*uyum|uyum değerlendir", re.I)),
    # Kısa liste
    ("shortlist",           re.compile(
        r"kısa liste|en iyi \d+|ilk \d+ aday|\d+ aday.*sırala"
        r"|adayları sırala|en uygun aday", re.I)),
    # Mülakat soruları
    ("interview_questions",  re.compile(
        r"mülakat soru|hangi soru.*soray|soru.*hazırla"
        r"|mülakat.*hazırla|\d+ soru|soru öner", re.I)),
    # İşe davet mesajı
    ("outreach_draft",      re.compile(
        r"e.?posta yaz|mesaj yaz|davet.*yaz|davet mesajı"
        r"|ulaşım mesajı|adaya.*yaz", re.I)),
    # Bütçe durumu
    ("budget_status",       re.compile(
        r"bütçe|token.*(kaldı|durum|kaç)|ne kadar token|işlem hakkım", re.I)),
    # Profil göster
    ("profile_show",        re.compile(
        r"profilim|tercihlerim|ayarlarım|profil.*göster|uzman profilim", re.I)),
    # Profil güncelle
    ("profile_update",      re.compile(
        r"profil.*güncelle|tercih.*değiştir|ayar.*değiştir"
        r"|ton.*değiştir|filtre.*değiştir", re.I)),
    # Karar geçmişi
    ("recruiter_history",   re.compile(
        r"geçmişim|daha önce.*seç|karar geçmişi|nasıl seçmiş"
        r"|hangi karar|geçmiş kararlar", re.I)),
    # Denetim kaydı
    ("audit_show",          re.compile(
        r"denetim kaydı|işlem geçmişi|yaptıklarım|denetim", re.I)),
    # Maliyet tablosu
    ("cost_table",          re.compile(
        r"maliyet tablosu|kaça mal olur|token maliyeti|işlem maliyeti", re.I)),
]


def _detect_intent(text: str) -> str:
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(text):
            return intent
    return "general"


# ── LLM call helper ───────────────────────────────────────────────────────────

def _llm(messages: list[dict], model: str = "llama3.2") -> str:
    try:
        resp = chat(model=model, messages=messages)
        return resp.message.content.strip()
    except Exception as exc:
        logger.error("HR LLM call failed: %s", exc)
        return f"LLM çağrısı başarısız oldu: {exc}"


# ── Gate pass ─────────────────────────────────────────────────────────────────

def _gate_pass(draft: str) -> tuple[str, dict]:
    report = _agent.gate_report(draft)
    if not report["conjunction"]:
        failed = {k: v for k, v in report["gates"].items() if not v[0]}
        warning = " | ".join(f"{k}: {v[1]}" for k, v in failed.items())
        draft = (
            f"[Güvenlik uyarısı: {warning}]\n\n"
            "Yanıt güvenlik kontrolünden geçemedi. "
            "Lütfen isteğinizi PII içermeyecek ve belirsizlik ifadesi taşımayacak biçimde yeniden iletin."
        )
    return draft, report["gates"]


def _line_value(text: str, patterns: list[str]) -> str:
    for p in patterns:
        m = re.search(p, text, flags=re.I)
        if m:
            return m.group(1).strip()
    return ""


def _split_list(raw: str) -> list[str]:
    if not raw:
        return []
    parts = [p.strip(" .-") for p in re.split(r"[,\n;|]+", raw) if p.strip()]
    return [p for p in parts if p.lower() not in {"yok", "none"}]


def _extract_score(text: str) -> float:
    m = re.search(r"(\d{1,3})(?:[.,]\d+)?\s*/\s*100", text)
    if not m:
        m = re.search(r"puan(?:ı)?[^0-9]{0,12}(\d{1,3})", text, flags=re.I)
    if not m:
        return 0.0
    try:
        return float(max(0, min(100, int(m.group(1)))))
    except ValueError:
        return 0.0


def _extract_recommended_actions(text: str) -> list[str]:
    block = _line_value(
        text,
        [
            r"önerilen aksiyonlar?\s*[:\-]\s*(.+)",
            r"önerilen işlemler?\s*[:\-]\s*(.+)",
            r"önerilen adımlar?\s*[:\-]\s*(.+)",
            r"sonraki adımlar?\s*[:\-]\s*(.+)",
        ],
    )
    if block:
        return _split_list(block)
    return []


def _action_allowed_for_intent(intent: str, action: str) -> bool:
    return action in INTENT_AUTOMATION_MAP.get(intent, [])


def _decide_automation_targets(
    intent: str,
    decision: str,
    user_input: str,
    draft: str,
    recommended_actions: list[str],
) -> tuple[bool, list[str]]:
    allowed_actions = list(INTENT_AUTOMATION_MAP.get(intent, []))
    if not allowed_actions:
        return False, []

    signal = bool(_AUTOMATION_SIGNAL_RE.search(user_input) or _AUTOMATION_SIGNAL_RE.search(draft))

    selected: list[str] = []
    for action in allowed_actions:
        # If action is explicitly recommended in text, keep it.
        if any(action == a for a in recommended_actions):
            selected.append(action)
            continue
        # Otherwise infer action from intent-specific hints.
        hint = _ACTION_HINT_RE.get(action)
        if hint and (hint.search(user_input) or hint.search(draft)):
            selected.append(action)

    if intent == "shortlist" and not selected:
        selected = allowed_actions
    elif intent == "candidate_match" and decision in {"Önerilir", "Şartlı Önerilir"} and not selected:
        selected = [a for a in allowed_actions if a == "log_to_ats"]

    should_trigger = bool(selected) and (signal or intent in {"shortlist", "candidate_match", "outreach_draft", "interview_questions"})
    selected = [a for a in selected if _action_allowed_for_intent(intent, a)]
    return should_trigger, selected


def _build_structured_response(
    *,
    intent: str,
    session: HRSessionContext,
    user_input: str,
    draft: str,
    token_cost: int,
    remaining_budget: int,
) -> HRStructuredResponse:
    decision = _line_value(
        draft,
        [
            r"karar\s*[:\-]\s*(Önerilir|Şartlı Önerilir|Önerilmez)",
            r"\b(Önerilir|Şartlı Önerilir|Önerilmez)\b",
        ],
    )
    candidate_name = _line_value(
        draft,
        [
            r"ad soyad\s*[:\-]\s*(.+)",
            r"aday(?:\s*adı)?\s*[:\-]\s*(.+)",
        ],
    )
    if not candidate_name:
        candidate_name = _line_value(user_input, [r"aday(?:\s*adı)?\s*[:\-]\s*(.+)"])

    strengths_raw = _line_value(draft, [r"g[üu]çl[üu] y[öo]nler\s*[:\-]\s*(.+)"])
    missing_raw = _line_value(draft, [r"eksik yetkinlikler\s*[:\-]\s*(.+)", r"zorunlu eksikler\s*[:\-]\s*(.+)"])
    risks = _line_value(draft, [r"risk analizi\s*[:\-]\s*(.+)", r"riskler\s*[:\-]\s*(.+)"])
    recruiter_summary = _line_value(draft, [r"karar gerekçesi\s*[:\-]\s*(.+)", r"özet[ıi]\s*[:\-]\s*(.+)"])
    if not recruiter_summary:
        recruiter_summary = draft[:240].strip()

    recommended_actions = _extract_recommended_actions(draft)
    normalized: list[str] = []
    for action in recommended_actions:
        a = action.strip()
        if a in _ACTION_HINT_RE:
            normalized.append(a)
    recommended_actions = normalized

    should_trigger, selected_actions = _decide_automation_targets(
        intent=intent,
        decision=decision,
        user_input=user_input,
        draft=draft,
        recommended_actions=recommended_actions,
    )

    return HRStructuredResponse(
        intent=intent,
        decision=decision,
        candidate_name=candidate_name,
        job_title=(session.current_req.title if session.current_req else ""),
        score=_extract_score(draft),
        strengths=_split_list(strengths_raw),
        missing_skills=_split_list(missing_raw),
        risks=risks,
        shortlist_status=("oluşturuldu" if intent == "shortlist" else ""),
        automation_targets=selected_actions,
        recommended_actions=selected_actions,
        follow_up_actions=selected_actions,
        should_trigger_automation=should_trigger,
        recruiter_summary=recruiter_summary,
        token_cost=token_cost,
        remaining_budget=remaining_budget,
        text_response=draft,
    )


def _dispatch_automation(
    *,
    structured: HRStructuredResponse,
    recruiter_id: str,
    session_id: str,
    token_remaining: int,
) -> None:
    if not structured.should_trigger_automation or not structured.recommended_actions:
        return

    payloads = build_payloads(
        structured=structured,
        recruiter_id=recruiter_id,
        session_id=session_id,
    )
    if not payloads:
        return

    try:
        trigger_all(payloads)
        log_action(
            recruiter_id,
            "automation_dispatch",
            session_id,
            token_cost=0,
            token_remaining=token_remaining,
            details={
                "intent": structured.intent,
                "actions": [p.action_type for p in payloads],
                "should_trigger_automation": structured.should_trigger_automation,
            },
            result_summary="n8n webhook tetikleme kuyruğa alındı",
        )
    except Exception as exc:
        log_action(
            recruiter_id,
            "automation_dispatch_failed",
            session_id,
            token_cost=0,
            token_remaining=token_remaining,
            details={
                "intent": structured.intent,
                "actions": structured.recommended_actions,
                "error": str(exc),
            },
            result_summary="n8n webhook tetikleme başarısız",
        )
        logger.warning("HR automation dispatch failed: %s", exc)


# ── Intent handlers ───────────────────────────────────────────────────────────

def _handle_cv_analyze(
    user_input: str,
    session: HRSessionContext,
    profile_summary: str,
    ledger: Any,
    session_id: str,
) -> str:
    ok, msg = check_and_deduct(ledger, "cv_parse", note=session_id)
    if not ok:
        return msg
    budget = budget_status_block(ledger)
    messages = _agent.build_cv_analysis_prompt(user_input, profile_summary, budget)
    draft = _llm(messages)
    draft, _ = _gate_pass(draft)
    log_action(
        session.recruiter_id, "cv_parse", session_id,
        token_cost=TOKEN_ACTION_COSTS["cv_parse"],
        token_remaining=ledger.remaining,
        result_summary=draft[:100],
    )
    return draft


def _handle_match(
    user_input: str,
    session: HRSessionContext,
    profile_summary: str,
    ledger: Any,
    session_id: str,
) -> str:
    ok, msg = check_and_deduct(ledger, "candidate_match", note=session_id)
    if not ok:
        return msg
    req_summary = (
        f"Başlık: {session.current_req.title}, "
        f"Gerekli beceriler: {', '.join(session.current_req.required_skills)}, "
        f"Kıdem: {session.current_req.seniority}"
    ) if session.current_req else "Aktif iş ilanı yok — kullanıcı mesajındaki bilgilere göre değerlendir."
    budget = budget_status_block(ledger)
    messages = _agent.build_match_prompt(user_input, req_summary, profile_summary, budget)
    draft = _llm(messages)
    draft, _ = _gate_pass(draft)
    log_action(
        session.recruiter_id, "candidate_match", session_id,
        token_cost=TOKEN_ACTION_COSTS["candidate_match"],
        token_remaining=ledger.remaining,
        result_summary=draft[:100],
    )
    return draft


def _handle_shortlist(
    user_input: str,
    session: HRSessionContext,
    profile_summary: str,
    ledger: Any,
    session_id: str,
    recruiter_shortlist_size: int,
) -> str:
    size_match = re.search(r"(\d+)\s*aday", user_input, re.I)
    size = int(size_match.group(1)) if size_match else recruiter_shortlist_size
    action = "shortlist_10" if size > 5 else "shortlist_5"
    ok, msg = check_and_deduct(ledger, action, note=session_id)
    if not ok:
        return msg
    req_summary = (
        f"Başlık: {session.current_req.title}, "
        f"Gerekli: {', '.join(session.current_req.required_skills)}, "
        f"Kıdem: {session.current_req.seniority}"
    ) if session.current_req else "İlan bilgisi yok — mesajdaki adayları değerlendir."
    candidates_block = user_input
    budget = budget_status_block(ledger)
    messages = _agent.build_shortlist_prompt(
        candidates_block, req_summary, profile_summary, size, budget
    )
    draft = _llm(messages)
    draft, _ = _gate_pass(draft)
    log_action(
        session.recruiter_id, action, session_id,
        token_cost=TOKEN_ACTION_COSTS[action],
        token_remaining=ledger.remaining,
        result_summary=draft[:100],
    )
    return draft


def _handle_interview_questions(
    user_input: str,
    session: HRSessionContext,
    profile_summary: str,
    ledger: Any,
    session_id: str,
) -> str:
    ok, msg = check_and_deduct(ledger, "interview_questions", note=session_id)
    if not ok:
        return msg
    req_summary = (
        f"Başlık: {session.current_req.title}, "
        f"Gerekli: {', '.join(session.current_req.required_skills)}"
    ) if session.current_req else "İlan bilgisi yok — mesajdaki pozisyon bilgilerini kullan."
    budget = budget_status_block(ledger)
    messages = _agent.build_interview_prompt(
        candidate_summary=user_input,
        requisition_summary=req_summary,
        missing_skills=[],
        profile_summary=profile_summary,
        budget_block=budget,
    )
    draft = _llm(messages)
    draft, _ = _gate_pass(draft)
    log_action(
        session.recruiter_id, "interview_questions", session_id,
        token_cost=TOKEN_ACTION_COSTS["interview_questions"],
        token_remaining=ledger.remaining,
        result_summary=draft[:100],
    )
    return draft


def _handle_outreach(
    user_input: str,
    session: HRSessionContext,
    profile: Any,
    profile_summary: str,
    ledger: Any,
    session_id: str,
) -> str:
    ok, msg = check_and_deduct(ledger, "outreach_draft", note=session_id)
    if not ok:
        return msg
    req_summary = (
        f"Başlık: {session.current_req.title}"
    ) if session.current_req else "Pozisyon bilgisi yok — mesajdaki bilgilere göre yaz."
    budget = budget_status_block(ledger)
    messages = _agent.build_outreach_prompt(
        candidate_summary=user_input,
        requisition_summary=req_summary,
        profile_summary=profile_summary,
        tone=profile.tone_preference,
        language=profile.language_preference,
        budget_block=budget,
    )
    draft = _llm(messages)
    draft, _ = _gate_pass(draft)
    log_action(
        session.recruiter_id, "outreach_draft", session_id,
        token_cost=TOKEN_ACTION_COSTS["outreach_draft"],
        token_remaining=ledger.remaining,
        result_summary=draft[:100],
    )
    return draft


def _handle_missing_skills(
    user_input: str,
    session: HRSessionContext,
    profile_summary: str,
    ledger: Any,
    session_id: str,
) -> str:
    ok, msg = check_and_deduct(ledger, "explanation", note=session_id)
    if not ok:
        return msg
    req_summary = (
        f"Gerekli yetkinlikler: {', '.join(session.current_req.required_skills)}"
    ) if session.current_req else "İlan bilgisi yok — mesajdaki ilan bilgilerini kullan."
    budget = budget_status_block(ledger)
    messages = _agent.build_missing_skills_prompt(
        candidate_summary=user_input,
        requisition_summary=req_summary,
        profile_summary=profile_summary,
        budget_block=budget,
    )
    draft = _llm(messages)
    draft, _ = _gate_pass(draft)
    log_action(
        session.recruiter_id, "explanation", session_id,
        token_cost=TOKEN_ACTION_COSTS["explanation"],
        token_remaining=ledger.remaining,
        result_summary=draft[:100],
    )
    return draft


# ── Main runner entry point ───────────────────────────────────────────────────

def run_hr_pipeline(task: AgentTask) -> AgentResponse:
    """
    HR agent pipeline.  Called by pipeline.py for mode='hr'.
    """
    user_input  = task.masked_input
    session_id  = task.session_id or "hr-default"

    # Derive a stable recruiter_id from session prefix
    recruiter_id = f"recruiter-{session_id[:8]}"

    profile  = load_profile(recruiter_id)
    ledger   = load_ledger(recruiter_id)
    session  = _get_or_create_session(session_id, recruiter_id)
    session.conversation_turns += 1

    profile_summary = build_profile_summary(profile)
    intent          = _detect_intent(user_input)
    remaining_before = ledger.remaining

    logger.info("HR runner: session=%s recruiter=%s intent=%s", session_id, recruiter_id, intent)

    # ── Intent dispatch ───────────────────────────────────────────────────────

    if intent == "budget_status":
        draft = (
            f"Mevcut bütçe durumunuz:\n{budget_status_block(ledger)}\n\n"
            f"{action_cost_table()}"
        )
        log_action(recruiter_id, "budget_status", session_id, token_cost=0, token_remaining=ledger.remaining)

    elif intent == "profile_show":
        draft = f"Recruiter profili:\n\n{profile_summary}"
        log_action(recruiter_id, "profile_show", session_id, token_cost=0, token_remaining=ledger.remaining)

    elif intent == "profile_update":
        ok, msg = check_and_deduct(ledger, "profile_update", note=session_id)
        if not ok:
            draft = msg
        else:
            # Parse simple key=value pairs from user input for preference updates
            tone_m = re.search(r"ton[u]?[:=\s]+(\w+)", user_input, re.I)
            strict_m = re.search(r"(strict|moderate|flexible|katı|orta|esnek)", user_input, re.I)
            size_m = re.search(r"liste\s*boyutu?[:=\s]*(\d+)", user_input, re.I)
            _strictness_map = {"katı": "strict", "orta": "moderate", "esnek": "flexible"}
            update_preferences(
                profile,
                tone=tone_m.group(1) if tone_m else None,
                strictness=(
                    _strictness_map.get(strict_m.group(1).lower(), strict_m.group(1).lower())
                    if strict_m else None
                ),
                shortlist_size=int(size_m.group(1)) if size_m else None,
            )
            draft = (
                "Profiliniz güncellendi.\n\n"
                f"Güncel profil:\n{build_profile_summary(profile)}\n\n"
                f"{budget_status_block(ledger)}"
            )
            log_action(recruiter_id, "profile_update", session_id,
                       token_cost=TOKEN_ACTION_COSTS["profile_update"],
                       token_remaining=ledger.remaining)

    elif intent == "recruiter_history":
        if profile.decision_history:
            lines = ["Geçmiş kararlarınız:"]
            for d in profile.decision_history[-10:]:
                lines.append(
                    f"  [{d.get('decision','?').upper()}] {d.get('candidate_name','?')} "
                    f"({d.get('req_title','')}) — {d.get('reason','')}"
                )
            draft = "\n".join(lines)
        else:
            draft = "Henüz kaydedilmiş karar geçmişiniz yok."
        log_action(recruiter_id, "recruiter_history", session_id, token_cost=0, token_remaining=ledger.remaining)

    elif intent == "audit_show":
        from src.pipeline.hr.audit_logger import read_audit_tail
        entries = read_audit_tail(recruiter_id, n=15)
        if entries:
            lines = ["Son 15 işlem:"]
            for e in entries:
                lines.append(
                    f"  [{e.get('ts','')[:19]}] {e.get('action','')} "
                    f"— {e.get('token_cost',0)} token — {e.get('result_summary','')}"
                )
            draft = "\n".join(lines)
        else:
            draft = "Henüz audit kaydı yok."

    elif intent == "cost_table":
        draft = action_cost_table()

    elif intent == "cv_analyze":
        draft = _handle_cv_analyze(user_input, session, profile_summary, ledger, session_id)

    elif intent == "candidate_match":
        draft = _handle_match(user_input, session, profile_summary, ledger, session_id)

    elif intent == "shortlist":
        draft = _handle_shortlist(
            user_input, session, profile_summary, ledger, session_id, profile.shortlist_size
        )

    elif intent == "interview_questions":
        draft = _handle_interview_questions(user_input, session, profile_summary, ledger, session_id)

    elif intent == "outreach_draft":
        draft = _handle_outreach(user_input, session, profile, profile_summary, ledger, session_id)

    elif intent == "missing_skills":
        draft = _handle_missing_skills(user_input, session, profile_summary, ledger, session_id)

    else:
        # General HR question — LLM with full recruiter context
        ok, msg = check_and_deduct(ledger, "explanation", note=session_id)
        if not ok:
            draft = msg
        else:
            session_ctx = f"Önceki işlem: {session.last_action}" if session.last_action else ""
            budget = budget_status_block(ledger)
            messages = _agent.build_general_prompt(user_input, profile_summary, session_ctx, budget)
            draft = _llm(messages)
            draft, _ = _gate_pass(draft)
            log_action(
                recruiter_id, "general", session_id,
                token_cost=TOKEN_ACTION_COSTS["explanation"],
                token_remaining=ledger.remaining,
                result_summary=draft[:100],
            )

    session.last_action = intent
    token_cost = max(0, remaining_before - ledger.remaining)

    structured = _build_structured_response(
        intent=intent,
        session=session,
        user_input=user_input,
        draft=draft,
        token_cost=token_cost,
        remaining_budget=ledger.remaining,
    )
    _dispatch_automation(
        structured=structured,
        recruiter_id=recruiter_id,
        session_id=session_id,
        token_remaining=ledger.remaining,
    )

    if structured.should_trigger_automation and structured.recommended_actions and _n8n_enabled():
        draft = draft + "\n\n" + _automation_note(structured.recommended_actions)

    return AgentResponse(
        task_id=task.task_id,
        agent_role=AgentRole.HR_AGENT,
        draft=draft,
        status=TaskStatus.COMPLETED,
        metadata={
            "intent":         intent,
            "recruiter_id":   recruiter_id,
            "token_remaining": ledger.remaining,
            "structured_response": {
                "intent": structured.intent,
                "decision": structured.decision,
                "candidate_name": structured.candidate_name,
                "job_title": structured.job_title,
                "score": structured.score,
                "strengths": structured.strengths,
                "missing_skills": structured.missing_skills,
                "risks": structured.risks,
                "shortlist_status": structured.shortlist_status,
                "automation_targets": structured.automation_targets,
                "recommended_actions": structured.recommended_actions,
                "follow_up_actions": structured.follow_up_actions,
                "should_trigger_automation": structured.should_trigger_automation,
                "recruiter_summary": structured.recruiter_summary,
                "token_cost": structured.token_cost,
                "remaining_budget": structured.remaining_budget,
            },
        },
    )
