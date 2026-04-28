"""Unit tests for HR structured response + n8n trigger bridge."""
from __future__ import annotations

from src.pipeline.hr.hr_schemas import HRSessionContext, HRStructuredResponse
from src.pipeline.hr.n8n_client import build_payloads
from src.pipeline.hr_runner import _build_structured_response


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
