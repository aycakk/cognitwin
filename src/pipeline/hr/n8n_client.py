"""pipeline/hr/n8n_client.py — n8n webhook client for HR automation.

Sends structured HR results to n8n workflows via HTTP POST webhook.

Design:
  - Fire-and-forget: webhook calls run in a background daemon thread so they
    NEVER block or delay the recruiter's answer in LibreChat.
  - Graceful degradation: if n8n is unreachable, errors are logged and the
    recruiter sees nothing unusual — the answer is already delivered.
  - Configurable: all webhook URLs come from environment variables.
    Set N8N_ENABLED=false to disable all automation silently.
  - Auditable: each trigger attempt is logged at INFO level; failures at WARNING.
  - No new dependencies: uses only stdlib urllib.request.

Environment variables (all optional):
  N8N_ENABLED                bool   "true" / "false" (default: false)
  HR_N8N_WEBHOOK_URL         str    highest-priority single URL for all HR actions
  N8N_WEBHOOK_URL            str    fallback single webhook URL for all actions
  N8N_WEBHOOK_BASE_URL       str    base for auto-constructed paths
                                    (default: http://n8n:5678/webhook)
  N8N_SHORTLIST_WEBHOOK      str    override for shortlist_to_sheets action
  N8N_SLACK_WEBHOOK          str    override for notify_slack action
  N8N_OUTREACH_WEBHOOK       str    override for send_outreach_email action
  N8N_INTERVIEW_WEBHOOK      str    override for create_calendar_event action
  N8N_MATCH_WEBHOOK          str    override for log_to_ats action
  N8N_CV_WEBHOOK             str    override for log_cv_analysis action
"""
from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline.hr.hr_schemas import HRStructuredResponse, N8nWebhookPayload

from src.pipeline.hr.hr_schemas import INTENT_AUTOMATION_MAP, N8nWebhookPayload

logger = logging.getLogger(__name__)

# ── Environment helpers ───────────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _is_enabled() -> bool:
    return _env("N8N_ENABLED", "false").lower() == "true"


_BASE_URL = _env("N8N_WEBHOOK_BASE_URL", "http://n8n:5678/webhook")
_GLOBAL_WEBHOOK = _env("N8N_WEBHOOK_URL", "")

# Map action_type → environment variable name → fallback path under base URL
_ACTION_ENV_MAP: dict[str, tuple[str, str]] = {
    "shortlist_to_sheets":   ("N8N_SHORTLIST_WEBHOOK",  f"{_BASE_URL}/hr-shortlist"),
    "notify_slack":          ("N8N_SLACK_WEBHOOK",      f"{_BASE_URL}/hr-slack"),
    "send_outreach_email":   ("N8N_OUTREACH_WEBHOOK",   f"{_BASE_URL}/hr-outreach"),
    "create_calendar_event": ("N8N_INTERVIEW_WEBHOOK",  f"{_BASE_URL}/hr-interview"),
    "log_to_ats":            ("N8N_MATCH_WEBHOOK",      f"{_BASE_URL}/hr-match"),
    "log_cv_analysis":       ("N8N_CV_WEBHOOK",         f"{_BASE_URL}/hr-cv"),
}


def _resolve_url(action_type: str) -> str:
    """Return the webhook URL for action_type, respecting env overrides."""
    # HR_N8N_WEBHOOK_URL is highest priority (local/testing convenience variable)
    hr_url = _env("HR_N8N_WEBHOOK_URL")
    if hr_url:
        return hr_url
    if _GLOBAL_WEBHOOK:
        return _GLOBAL_WEBHOOK
    env_key, fallback = _ACTION_ENV_MAP.get(action_type, ("", ""))
    if env_key:
        override = _env(env_key)
        if override:
            return override
    if fallback:
        return fallback
    # Unknown action — construct a path from base URL
    slug = action_type.replace("_", "-")
    return f"{_BASE_URL}/hr-{slug}"


# ── HTTP fire ─────────────────────────────────────────────────────────────────

def _fire(payload_dict: dict, url: str) -> None:
    """POST JSON to url.  Never raises — all exceptions are caught and logged."""
    action = payload_dict.get("action_type", "?")
    try:
        data = json.dumps(payload_dict, ensure_ascii=False, default=str).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.getcode()
            logger.info(
                "n8n webhook OK  action=%s url=%s status=%d  session=%s",
                action, url, status, payload_dict.get("session_id", "?"),
            )
    except urllib.error.URLError as exc:
        logger.warning(
            "n8n webhook unreachable  action=%s url=%s: %s  "
            "(n8n may be down — recruiter answer was already delivered)",
            action, url, exc,
        )
    except Exception as exc:
        logger.error(
            "n8n webhook unexpected error  action=%s url=%s: %s",
            action, url, exc,
        )


# ── Public API ────────────────────────────────────────────────────────────────

def trigger_automation(payload: "N8nWebhookPayload") -> None:
    """
    Send one automation webhook to n8n in a background daemon thread.

    Returns immediately — the caller (hr_runner) is never blocked.
    If N8N_ENABLED is not 'true', this is a no-op.
    """
    if not _is_enabled():
        logger.debug(
            "n8n disabled (N8N_ENABLED != true) — skipping action=%s",
            payload.action_type,
        )
        return

    url = _resolve_url(payload.action_type)

    payload_dict = {
        "action_type":      payload.action_type,
        "intent":           payload.intent,
        "recruiter_id":     payload.recruiter_id,
        "session_id":       payload.session_id,
        "decision":         payload.decision,
        "candidate_name":   payload.candidate_name,
        "candidate_id":     payload.candidate_id,
        "job_title":        payload.job_title,
        "job_id":           payload.job_id,
        "score":            payload.score,
        "strengths":        payload.strengths,
        "missing_skills":   payload.missing_skills,
        "risks":            payload.risks,
        "shortlist_status": payload.shortlist_status,
        # Truncate text so webhook payloads stay under 2 KB
        "text_response":    payload.text_response[:2000],
        "token_cost":       payload.token_cost,
        "remaining_budget": payload.remaining_budget,
        "source":           payload.source,
        "triggered_at":     datetime.now(timezone.utc).isoformat(),
        "extra":            payload.extra or {},
    }

    thread = threading.Thread(
        target=_fire,
        args=(payload_dict, url),
        daemon=True,   # daemon=True: thread dies if main process exits
        name=f"n8n-{payload.action_type}-{payload.session_id[:8]}",
    )
    thread.start()
    logger.info(
        "n8n trigger dispatched  action=%s session=%s",
        payload.action_type, payload.session_id,
    )


def trigger_all(payloads: "list[N8nWebhookPayload]") -> None:
    """Fire all automation webhooks from a list.  No-op if list is empty."""
    for p in payloads:
        trigger_automation(p)


def _is_allowed_action(intent: str, action_type: str) -> bool:
    return action_type in INTENT_AUTOMATION_MAP.get(intent, [])


def _build_extra_fields(structured: "HRStructuredResponse") -> dict:
    return {
        "recruiter_summary": structured.recruiter_summary,
        "follow_up_actions": structured.follow_up_actions,
        "recommended_actions": structured.recommended_actions,
    }


def build_payloads(
    *,
    structured: "HRStructuredResponse",
    recruiter_id: str,
    session_id: str,
) -> list[N8nWebhookPayload]:
    """Build one webhook payload per validated recommended action."""
    payloads: list[N8nWebhookPayload] = []
    for action in structured.recommended_actions:
        if not _is_allowed_action(structured.intent, action):
            logger.warning(
                "n8n action blocked (not allowed for intent) intent=%s action=%s",
                structured.intent,
                action,
            )
            continue
        payloads.append(
            N8nWebhookPayload(
                action_type=action,
                intent=structured.intent,
                recruiter_id=recruiter_id,
                session_id=session_id,
                decision=structured.decision,
                candidate_name=structured.candidate_name,
                job_title=structured.job_title,
                score=structured.score,
                strengths=structured.strengths,
                missing_skills=structured.missing_skills,
                risks=structured.risks,
                shortlist_status=structured.shortlist_status,
                text_response=structured.text_response,
                token_cost=structured.token_cost,
                remaining_budget=structured.remaining_budget,
                extra=_build_extra_fields(structured),
            )
        )
    return payloads
