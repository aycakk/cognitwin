"""Tests for recruiter-scoped candidate and automation JSONL logging.

Covers:
- log_candidate_event / log_automation_event write correct fields
- read_candidate_events / read_automation_events are scoped per recruiter
- GET /api/hr/candidates/me returns only the caller's records
- GET /api/hr/automation-history/me returns only the caller's records
- Missing X-Recruiter-ID still returns 400
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_candidate(data_dir: Path, recruiter_id: str, **kwargs) -> None:
    subdir = data_dir / recruiter_id
    subdir.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"recruiter_id": recruiter_id, **kwargs})
    (subdir / "candidates.jsonl").open("a", encoding="utf-8").write(line + "\n")


def _write_automation(data_dir: Path, recruiter_id: str, **kwargs) -> None:
    subdir = data_dir / recruiter_id
    subdir.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"recruiter_id": recruiter_id, **kwargs})
    (subdir / "automation_log.jsonl").open("a", encoding="utf-8").write(line + "\n")


# ---------------------------------------------------------------------------
# Unit tests for audit_logger helpers
# ---------------------------------------------------------------------------

def test_log_candidate_event_creates_file(tmp_path):
    import src.pipeline.hr.audit_logger as al
    al._DATA_DIR = tmp_path

    al.log_candidate_event(
        "rec-unit",
        candidate_name="Ali Veli",
        job_title="Backend Developer",
        decision="Önerilir",
        score=85.0,
        action_type="candidate_match",
        automation_status="none",
        token_cost=20,
        remaining_budget=980,
    )

    events = al.read_candidate_events("rec-unit")
    assert len(events) == 1
    e = events[0]
    assert e["candidate_name"] == "Ali Veli"
    assert e["job_title"] == "Backend Developer"
    assert e["score"] == 85.0
    assert e["recruiter_id"] == "rec-unit"
    assert "created_at" in e


def test_log_automation_event_creates_file(tmp_path):
    import src.pipeline.hr.audit_logger as al
    al._DATA_DIR = tmp_path

    al.log_automation_event(
        "rec-auto",
        action_type="notify_slack",
        candidate_name="Ali Veli",
        job_title="Backend Developer",
        status="dispatched",
        n8n_status="",
    )

    events = al.read_automation_events("rec-auto")
    assert len(events) == 1
    e = events[0]
    assert e["action_type"] == "notify_slack"
    assert e["status"] == "dispatched"
    assert e["recruiter_id"] == "rec-auto"
    assert "created_at" in e


def test_candidate_events_isolated_between_recruiters(tmp_path):
    import src.pipeline.hr.audit_logger as al
    al._DATA_DIR = tmp_path

    al.log_candidate_event("rec-a", candidate_name="Aday A")
    al.log_candidate_event("rec-b", candidate_name="Aday B")

    a_events = al.read_candidate_events("rec-a")
    b_events = al.read_candidate_events("rec-b")

    assert len(a_events) == 1
    assert a_events[0]["candidate_name"] == "Aday A"
    assert len(b_events) == 1
    assert b_events[0]["candidate_name"] == "Aday B"


def test_automation_events_isolated_between_recruiters(tmp_path):
    import src.pipeline.hr.audit_logger as al
    al._DATA_DIR = tmp_path

    al.log_automation_event("rec-x", action_type="notify_slack")
    al.log_automation_event("rec-y", action_type="shortlist_to_sheets")

    x_events = al.read_automation_events("rec-x")
    y_events = al.read_automation_events("rec-y")

    assert x_events[0]["action_type"] == "notify_slack"
    assert y_events[0]["action_type"] == "shortlist_to_sheets"
    assert len(x_events) == 1
    assert len(y_events) == 1


def test_read_returns_empty_for_unknown_recruiter(tmp_path):
    import src.pipeline.hr.audit_logger as al
    al._DATA_DIR = tmp_path

    assert al.read_candidate_events("nobody") == []
    assert al.read_automation_events("nobody") == []


# ---------------------------------------------------------------------------
# FastAPI integration tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client_and_dir(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("hr_profiles_log")

    import src.pipeline.hr.audit_logger as al
    import src.pipeline.hr.recruiter_profile_store as rps
    import src.pipeline.hr.token_ledger as tl

    al._DATA_DIR = data_dir
    rps._DATA_DIR = data_dir
    tl._DATA_DIR = data_dir

    from src.core.schemas import AgentRole, AgentResponse, TaskStatus

    fake_response = AgentResponse(
        task_id="t1",
        agent_role=AgentRole.HR_AGENT,
        draft="Test yanıtı",
        status=TaskStatus.COMPLETED,
        metadata={
            "intent": "candidate_match",
            "recruiter_id": "panel-recruiter",
            "token_remaining": 980,
            "structured_response": {
                "candidate_name": "Test Aday",
                "job_title": "Backend Developer",
                "decision": "Önerilir",
                "score": 88.0,
                "token_cost": 20,
                "automation_targets": [],
                "should_trigger_automation": False,
            },
        },
    )

    with patch("src.pipeline.hr_runner.run_hr_pipeline", return_value=fake_response):
        from src.services.api.app import app
        yield TestClient(app), data_dir


# ── Header validation ────────────────────────────────────────────────────────

def test_candidates_missing_header_returns_400(client_and_dir):
    client, _ = client_and_dir
    r = client.get("/api/hr/candidates/me")
    assert r.status_code == 400


def test_automation_missing_header_returns_400(client_and_dir):
    client, _ = client_and_dir
    r = client.get("/api/hr/automation-history/me")
    assert r.status_code == 400


# ── Candidates/me reads from JSONL ───────────────────────────────────────────

def test_candidates_me_reads_own_jsonl(client_and_dir):
    client, data_dir = client_and_dir

    # Seed a record directly into the JSONL for recruiter-alice
    _write_candidate(data_dir, "recruiter-alice", candidate_name="Zeynep Ay", job_title="PM")

    r = client.get("/api/hr/candidates/me", headers={"X-Recruiter-ID": "recruiter-alice"})
    assert r.status_code == 200
    body = r.json()
    assert body["recruiter_id"] == "recruiter-alice"
    names = [c["candidate_name"] for c in body["candidates"]]
    assert "Zeynep Ay" in names


def test_candidates_me_does_not_see_other_recruiter(client_and_dir):
    client, data_dir = client_and_dir

    _write_candidate(data_dir, "recruiter-bob", candidate_name="Kemal Taş", job_title="DevOps")

    r = client.get("/api/hr/candidates/me", headers={"X-Recruiter-ID": "recruiter-alice"})
    names = [c["candidate_name"] for c in r.json()["candidates"]]
    assert "Kemal Taş" not in names


# ── Automation-history/me reads from JSONL ───────────────────────────────────

def test_automation_history_reads_own_jsonl(client_and_dir):
    client, data_dir = client_and_dir

    _write_automation(data_dir, "recruiter-carol", action_type="notify_slack", status="dispatched")

    r = client.get("/api/hr/automation-history/me", headers={"X-Recruiter-ID": "recruiter-carol"})
    assert r.status_code == 200
    body = r.json()
    assert body["recruiter_id"] == "recruiter-carol"
    actions = [e["action_type"] for e in body["automation_history"]]
    assert "notify_slack" in actions


def test_automation_history_does_not_see_other_recruiter(client_and_dir):
    client, data_dir = client_and_dir

    _write_automation(data_dir, "recruiter-dave", action_type="shortlist_to_sheets", status="dispatched")

    r = client.get("/api/hr/automation-history/me", headers={"X-Recruiter-ID": "recruiter-carol"})
    actions = [e["action_type"] for e in r.json()["automation_history"]]
    assert "shortlist_to_sheets" not in actions


# ── agent/run writes candidate event ─────────────────────────────────────────

def test_agent_run_writes_candidate_event(client_and_dir):
    client, data_dir = client_and_dir

    r = client.post(
        "/api/hr/agent/run",
        json={"prompt": "CV değerlendir"},
        headers={"X-Recruiter-ID": "recruiter-eve"},
    )
    assert r.status_code == 200

    import src.pipeline.hr.audit_logger as al
    al._DATA_DIR = data_dir
    events = al.read_candidate_events("recruiter-eve")
    assert len(events) >= 1
    assert all(e["recruiter_id"] == "recruiter-eve" for e in events)
