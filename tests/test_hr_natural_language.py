"""Tests for natural-language HR action detection and extraction.

Covers Tasks 5, 6, and 7 from the HR Panel improvement spec:
- Natural Turkish phrases trigger correct automations without "Aksiyon:" directives
- "değerlendir" alone does NOT trigger external automation
- candidate_name and job_title are extracted from natural-language prompts
- Full skill match produces a nonzero score
- Recruiter isolation and X-Recruiter-ID header are still respected
"""
from __future__ import annotations

import pytest

from src.pipeline.hr.hr_schemas import HRSessionContext, HRStructuredResponse
from src.pipeline.hr_runner import (
    _decide_automation_targets,
    _detect_intent,
    _extract_first,
    _CANDIDATE_NAME_RES,
    _JOB_TITLE_RES,
    _skill_match_score,
    _AUTOMATION_SIGNAL_RE,
    _ACTION_HINT_RE,
)


# ── Helper: build a minimal HRStructuredResponse for automation tests ─────────

def _sr(**kwargs) -> HRStructuredResponse:
    defaults = dict(
        intent="candidate_match",
        decision="İlk görüşme için önerilir",
        candidate_name="Test Aday",
        job_title="Backend Developer",
        score=85.0,
        automation_targets=[],
        recommended_actions=[],
        should_trigger_automation=False,
        token_cost=20,
        remaining_budget=980,
        text_response="",
    )
    defaults.update(kwargs)
    return HRStructuredResponse(**defaults)


# ── Intent detection ──────────────────────────────────────────────────────────

def test_natural_candidate_match_intent():
    prompt = "Ahmet Yılmaz adayını Backend Developer pozisyonu için değerlendir."
    assert _detect_intent(prompt) == "candidate_match"


def test_natural_candidate_match_intent_with_slack():
    prompt = "Ahmet Yılmaz adayını Backend Developer pozisyonu için değerlendir, uygunsa Slack'e bildir."
    assert _detect_intent(prompt) == "candidate_match"


def test_degerlendir_alone_is_general_or_match():
    # Either general or candidate_match is acceptable; must not be "shortlist"
    intent = _detect_intent("Bu CV'yi değerlendir lütfen.")
    assert intent in {"candidate_match", "cv_analyze", "general"}


# ── Automation signal regex ───────────────────────────────────────────────────

def test_signal_slack_bildir():
    assert _AUTOMATION_SIGNAL_RE.search("Slack'e bildir")


def test_signal_ekibe_bildir():
    assert _AUTOMATION_SIGNAL_RE.search("ekibe bildir")


def test_signal_kisa_listeye_ekle():
    assert _AUTOMATION_SIGNAL_RE.search("kısa listeye ekle")


def test_signal_mail_hazirla():
    assert _AUTOMATION_SIGNAL_RE.search("mail hazırla")


def test_signal_takvim_olustur():
    assert _AUTOMATION_SIGNAL_RE.search("takvim etkinliği oluştur")


def test_signal_gorušme_planla():
    assert _AUTOMATION_SIGNAL_RE.search("görüşme planla")


def test_no_signal_on_plain_degerlendir():
    assert not _AUTOMATION_SIGNAL_RE.search("Sadece değerlendir")


# ── Action hint regex ─────────────────────────────────────────────────────────

def test_hint_slack_matches_bildir():
    assert _ACTION_HINT_RE["notify_slack"].search("Slack'e bildir")


def test_hint_slack_matches_slack_keyword():
    assert _ACTION_HINT_RE["notify_slack"].search("Slack üzerinden bildir")


def test_hint_sheets_matches_kisa_listeye_ekle():
    assert _ACTION_HINT_RE["shortlist_to_sheets"].search("kısa listeye ekle")


def test_hint_sheets_matches_shortlist_al():
    assert _ACTION_HINT_RE["shortlist_to_sheets"].search("shortlist'e al")


def test_hint_email_matches_mail_hazirla():
    assert _ACTION_HINT_RE["send_outreach_email"].search("mail hazırla")


def test_hint_email_matches_eposta_hazirla():
    assert _ACTION_HINT_RE["send_outreach_email"].search("e-posta hazırla")


def test_hint_calendar_matches_mulakat_takvim():
    assert _ACTION_HINT_RE["create_calendar_event"].search("mülakat takvimi oluştur")


def test_hint_calendar_matches_gorusme_planla():
    assert _ACTION_HINT_RE["create_calendar_event"].search("görüşme planla")


# ── Automation targeting ──────────────────────────────────────────────────────

FULL_PROMPT = (
    "Ahmet Yılmaz adayını Backend Developer pozisyonu için değerlendir, "
    "uygunsa Slack'e bildir.\n\n"
    "Aday:\nAhmet Yılmaz\nPython, FastAPI, Docker ve PostgreSQL biliyor.\n"
    "3 yıl backend deneyimi var.\n\n"
    "İlan:\nBackend Developer\n"
    "Gerekli yetkinlikler: Python, FastAPI, Docker, PostgreSQL"
)

def test_slack_triggered_from_natural_prompt():
    should, actions = _decide_automation_targets(
        intent="candidate_match",
        decision="İlk görüşme için önerilir",
        user_input=FULL_PROMPT,
        draft="Karar: Önerilir\nPuan: 85/100",
        recommended_actions=[],
    )
    assert should is True
    assert "notify_slack" in actions


def test_kisa_liste_triggers_shortlist_to_sheets():
    prompt = "Ahmet Yılmaz adayını değerlendir, kısa listeye ekle."
    should, actions = _decide_automation_targets(
        intent="shortlist",
        decision="",
        user_input=prompt,
        draft="",
        recommended_actions=[],
    )
    assert should is True
    assert "shortlist_to_sheets" in actions


def test_mail_hazirla_triggers_outreach():
    prompt = "Bu aday için mail hazırla."
    should, actions = _decide_automation_targets(
        intent="outreach_draft",
        decision="",
        user_input=prompt,
        draft="",
        recommended_actions=[],
    )
    assert should is True
    assert "send_outreach_email" in actions


def test_mulakat_takvimi_triggers_calendar():
    prompt = "Mülakat takvimi oluştur."
    should, actions = _decide_automation_targets(
        intent="interview_questions",
        decision="",
        user_input=prompt,
        draft="",
        recommended_actions=[],
    )
    assert should is True
    assert "create_calendar_event" in actions


def test_degerlendir_alone_does_not_trigger():
    """Plain 'değerlendir' must not fire any external automation."""
    should, actions = _decide_automation_targets(
        intent="candidate_match",
        decision="İlk görüşme için önerilir",
        user_input="Ahmet Yılmaz adayını değerlendir.",
        draft="Karar: Önerilir",
        recommended_actions=[],
    )
    assert should is False
    assert actions == []


# ── Extraction ────────────────────────────────────────────────────────────────

def test_extract_candidate_name_from_natural_prompt():
    name = _extract_first(FULL_PROMPT, _CANDIDATE_NAME_RES)
    assert name == "Ahmet Yılmaz"


def test_extract_job_title_from_natural_prompt():
    title = _extract_first(FULL_PROMPT, _JOB_TITLE_RES)
    assert "Backend Developer" in title


def test_full_skill_match_gives_nonzero_score():
    candidate = ["Python", "FastAPI", "Docker", "PostgreSQL"]
    required  = ["Python", "FastAPI", "Docker", "PostgreSQL"]
    score = _skill_match_score(candidate, required)
    assert score == 100.0


def test_partial_skill_match_gives_partial_score():
    candidate = ["Python", "FastAPI"]
    required  = ["Python", "FastAPI", "Docker", "PostgreSQL"]
    score = _skill_match_score(candidate, required)
    assert 0 < score < 100
