"""Unit tests for HR structured response + n8n trigger bridge."""
from __future__ import annotations

from src.pipeline.hr.hr_schemas import HRSessionContext, HRStructuredResponse
from src.pipeline.hr.n8n_client import build_payloads
from src.pipeline.hr_runner import (
    _build_structured_response,
    _clean_draft,
    _strip_leaked_sections,
    _extract_first,
    _CANDIDATE_NAME_RES,
    _JOB_TITLE_RES,
    _skill_match_score,
    _auto_decision,
)


def test_shortlist_intent_triggers_allowed_actions() -> None:
    draft = (
        "KARAR: Önerilir\n"
        "GÜÇLÜ YÖNLER: Python, SQL\n"
        "EKSİK YETKİNLİKLER: Yok\n"
        "Kısa liste oluşturuldu."
    )
    structured = _build_structured_response(
        intent="shortlist",
        session=HRSessionContext(session_id="s1", recruiter_id="r1"),
        user_input="Bu kısa listeyi n8n ile sheet ve slack'e aktar.",
        draft=draft,
        token_cost=50,
        remaining_budget=950,
    )
    assert structured.should_trigger_automation is True
    assert "shortlist_to_sheets" in structured.recommended_actions
    assert "notify_slack" in structured.recommended_actions


def test_candidate_match_reject_does_not_trigger_ats() -> None:
    structured = _build_structured_response(
        intent="candidate_match",
        session=HRSessionContext(session_id="s2", recruiter_id="r2"),
        user_input="Adayı değerlendir.",
        draft="KARAR: Önerilmez\nKARAR GEREKÇESİ: Zorunlu yetkinlik eksik.",
        token_cost=20,
        remaining_budget=980,
    )
    assert structured.should_trigger_automation is False
    assert "log_to_ats" not in structured.recommended_actions


def test_build_payloads_blocks_disallowed_action() -> None:
    structured = HRStructuredResponse(
        intent="cv_analyze",
        recommended_actions=["notify_slack"],
        should_trigger_automation=True,
        text_response="CV analizi tamamlandı.",
    )
    payloads = build_payloads(
        structured=structured,
        recruiter_id="rec-1",
        session_id="sess-1",
    )
    assert payloads == []


def test_explicit_aksiyon_directive_sends_notify_slack() -> None:
    """'Aksiyon: notify_slack' in user input must produce action_type=notify_slack
    even when intent is 'general' (no shortlist context)."""
    structured = _build_structured_response(
        intent="general",
        session=HRSessionContext(session_id="s3", recruiter_id="r3"),
        user_input="Ahmet Yılmaz adayını değerlendir. Aksiyon: notify_slack",
        draft="Genel değerlendirme tamamlandı.",
        token_cost=5,
        remaining_budget=995,
    )
    assert structured.should_trigger_automation is True
    assert "notify_slack" in structured.recommended_actions
    payloads = build_payloads(
        structured=structured,
        recruiter_id="r3",
        session_id="s3",
    )
    assert len(payloads) == 1
    assert payloads[0].action_type == "notify_slack"


def test_strip_leaked_sections_removes_internal_headers() -> None:
    raw = (
        "=== İŞE ALIM UZMANI PROFİLİ ===\n"
        "Recruiter: Unknown @ Unknown company\n"
        "Industry focus: any\n\n"
        "=== KULLANICI SORUSU ===\n"
        "Aksiyon: notify_slack\n\n"
        "Genel bir değerlendirme yapıldı."
    )
    cleaned = _strip_leaked_sections(raw)
    assert "===" not in cleaned
    assert "KULLANICI SORUSU" not in cleaned
    assert "Recruiter:" not in cleaned
    assert "Genel bir değerlendirme yapıldı." in cleaned


def test_clean_draft_returns_fallback_for_banned_words() -> None:
    raw = "This candidate has great experience and skillset in backend development."
    result = _clean_draft(raw)
    assert "experience" not in result.lower()
    assert "skillset" not in result.lower()
    assert "ADAY ÖN DEĞERLENDİRME SONUCU" in result


def test_extract_candidate_name_from_turkish_prompt() -> None:
    text = "Ahmet Yılmaz adayını Backend Developer pozisyonu için değerlendir."
    assert _extract_first(text, _CANDIDATE_NAME_RES) == "Ahmet Yılmaz"


def test_extract_job_title_from_turkish_prompt() -> None:
    text = "Ahmet Yılmaz adayını Backend Developer pozisyonu için değerlendir."
    assert _extract_first(text, _JOB_TITLE_RES) == "Backend Developer"


def test_full_skill_match_gives_nonzero_score() -> None:
    candidate = ["Python", "FastAPI", "Docker", "PostgreSQL"]
    required = ["Python", "FastAPI", "Docker", "PostgreSQL"]
    assert _skill_match_score(candidate, required) == 100.0


def test_notify_slack_payload_has_slack_text() -> None:
    from src.pipeline.hr.n8n_client import build_payloads
    structured = _build_structured_response(
        intent="shortlist",
        session=HRSessionContext(session_id="s4", recruiter_id="r4"),
        user_input="Ahmet Yılmaz adayını Backend Developer pozisyonu için değerlendir. Kısa liste oluştur.",
        draft=(
            "KARAR: Önerilir\n"
            "GÜÇLÜ YÖNLER: Python, FastAPI\n"
            "EKSİK YETKİNLİKLER: Kubernetes\n"
            "PUAN: 85\n"
            "Değerlendirme tamamlandı."
        ),
        token_cost=50,
        remaining_budget=950,
    )
    payloads = build_payloads(structured=structured, recruiter_id="r4", session_id="s4")
    slack_payloads = [p for p in payloads if p.action_type == "notify_slack"]
    assert slack_payloads, "notify_slack payload must be generated for shortlist intent"
    p = slack_payloads[0]
    assert p.slack_text, "slack_text must not be empty"
    assert "Yeni İK Bildirimi" in p.slack_text
    assert "CognitWin HR Agent" in p.slack_text


def test_auto_decision_from_score() -> None:
    assert _auto_decision(90.0) == "Önerilir"
    assert _auto_decision(60.0) == "Şartlı Önerilir"
    assert _auto_decision(30.0) == "Önerilmez"
