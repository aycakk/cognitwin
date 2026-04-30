"""Smoke tests for the personal HR Panel endpoints.

These tests use FastAPI's TestClient and mock run_hr_pipeline so they run
without Ollama or n8n being present.  They verify routing, header validation,
and recruiter_id scoping — not HR agent logic (that is covered by the existing
test_hr_automation_bridge.py suite).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Minimal stubs so the app can be imported without a running Ollama / ChromaDB
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("hr_profiles")

    # Patch the data directories BEFORE importing the app so file I/O lands in
    # the temp directory instead of the real data/hr_profiles/.
    import src.pipeline.hr.recruiter_profile_store as rps
    import src.pipeline.hr.token_ledger as tl
    import src.pipeline.hr.audit_logger as al

    rps._DATA_DIR = data_dir
    tl._DATA_DIR = data_dir
    al._DATA_DIR = data_dir

    from src.core.schemas import AgentRole, AgentResponse, TaskStatus

    fake_response = AgentResponse(
        task_id="test-task",
        agent_role=AgentRole.HR_AGENT,
        draft="Test yanıtı",
        status=TaskStatus.COMPLETED,
        metadata={
            "intent": "general",
            "recruiter_id": "test-recruiter",
            "token_remaining": 990,
            "structured_response": {},
        },
    )

    with patch("src.services.api.hr_router.run_hr_pipeline", return_value=fake_response):
        from src.services.api.app import app
        yield TestClient(app)


# ---------------------------------------------------------------------------
# Header validation
# ---------------------------------------------------------------------------

def test_missing_recruiter_id_returns_400(client):
    r = client.post("/api/hr/agent/run", json={"prompt": "merhaba"})
    assert r.status_code == 400
    assert "X-Recruiter-ID" in r.json()["detail"]


def test_invalid_recruiter_id_returns_400(client):
    r = client.post(
        "/api/hr/agent/run",
        json={"prompt": "merhaba"},
        headers={"X-Recruiter-ID": "../../etc/passwd"},
    )
    assert r.status_code == 400


def test_valid_recruiter_id_accepted(client):
    r = client.post(
        "/api/hr/agent/run",
        json={"prompt": "merhaba"},
        headers={"X-Recruiter-ID": "recruiter-ayse"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["recruiter_id"] == "recruiter-ayse"
    assert "response" in body
    assert "session_id" in body


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def test_dashboard_me_returns_budget(client):
    r = client.get("/api/hr/dashboard/me", headers={"X-Recruiter-ID": "recruiter-ayse"})
    assert r.status_code == 200
    body = r.json()
    assert "token_budget" in body
    assert "token_remaining" in body
    assert body["recruiter_id"] == "recruiter-ayse"


def test_dashboard_missing_header_returns_400(client):
    r = client.get("/api/hr/dashboard/me")
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Recruiter profile
# ---------------------------------------------------------------------------

def test_get_profile_returns_recruiter_id(client):
    r = client.get("/api/hr/recruiter-profile/me", headers={"X-Recruiter-ID": "recruiter-mehmet"})
    assert r.status_code == 200
    body = r.json()
    assert body["recruiter_id"] == "recruiter-mehmet"


def test_put_profile_updates_name(client):
    r = client.put(
        "/api/hr/recruiter-profile/me",
        json={"name": "Mehmet Yılmaz", "company": "TechCorp"},
        headers={"X-Recruiter-ID": "recruiter-mehmet"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "Mehmet Yılmaz"
    assert body["company"] == "TechCorp"

    # Verify persisted
    r2 = client.get("/api/hr/recruiter-profile/me", headers={"X-Recruiter-ID": "recruiter-mehmet"})
    assert r2.json()["name"] == "Mehmet Yılmaz"


# ---------------------------------------------------------------------------
# Candidates and automation history
# ---------------------------------------------------------------------------

def test_candidates_me_returns_list(client):
    r = client.get("/api/hr/candidates/me", headers={"X-Recruiter-ID": "recruiter-ayse"})
    assert r.status_code == 200
    assert "candidates" in r.json()


def test_automation_history_me_returns_list(client):
    r = client.get("/api/hr/automation-history/me", headers={"X-Recruiter-ID": "recruiter-ayse"})
    assert r.status_code == 200
    assert "automation_history" in r.json()


# ---------------------------------------------------------------------------
# Recruiter isolation — two recruiters must not share data
# ---------------------------------------------------------------------------

def test_recruiter_isolation(client):
    client.put(
        "/api/hr/recruiter-profile/me",
        json={"name": "Ayşe Kaya"},
        headers={"X-Recruiter-ID": "recruiter-ayse"},
    )
    client.put(
        "/api/hr/recruiter-profile/me",
        json={"name": "Fatma Demir"},
        headers={"X-Recruiter-ID": "recruiter-fatma"},
    )
    ayse = client.get("/api/hr/recruiter-profile/me", headers={"X-Recruiter-ID": "recruiter-ayse"}).json()
    fatma = client.get("/api/hr/recruiter-profile/me", headers={"X-Recruiter-ID": "recruiter-fatma"}).json()
    assert ayse["name"] == "Ayşe Kaya"
    assert fatma["name"] == "Fatma Demir"
    assert ayse["name"] != fatma["name"]
