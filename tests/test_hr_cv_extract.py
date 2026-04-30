from __future__ import annotations

from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("hr_profiles_cv")

    import src.pipeline.hr.audit_logger as al
    import src.pipeline.hr.recruiter_profile_store as rps
    import src.pipeline.hr.token_ledger as tl

    al._DATA_DIR = data_dir
    rps._DATA_DIR = data_dir
    tl._DATA_DIR = data_dir

    from src.services.api.app import app
    return TestClient(app)


def test_cv_extract_requires_header(client: TestClient):
    file_data = BytesIO(b"Ali Veli\nPython FastAPI")
    r = client.post("/api/hr/cv/extract", files={"file": ("cv.txt", file_data, "text/plain")})
    assert r.status_code == 400
    assert "X-Recruiter-ID" in r.json()["detail"]


def test_cv_extract_txt_ok(client: TestClient):
    text = "Ali Veli\nBackend Developer\nYetkinlikler: Python, FastAPI, Docker"
    r = client.post(
        "/api/hr/cv/extract",
        headers={"X-Recruiter-ID": "recruiter-test"},
        files={"file": ("cv.txt", BytesIO(text.encode("utf-8")), "text/plain")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "Ali Veli" in body["extracted_text"]
    assert body["filename"] == "cv.txt"


def test_cv_extract_size_limit(client: TestClient):
    big = b"a" * (10 * 1024 * 1024 + 10)
    r = client.post(
        "/api/hr/cv/extract",
        headers={"X-Recruiter-ID": "recruiter-test"},
        files={"file": ("big.txt", BytesIO(big), "text/plain")},
    )
    assert r.status_code == 413


def test_cv_extract_infers_candidate_name_and_position(client: TestClient):
    text = (
        "Aday: Derya Aksu\n"
        "Pozisyon: Backend Developer\n"
        "Python, FastAPI, Docker ve PostgreSQL biliyor.\n"
        "3 yıl backend deneyimi var."
    )
    r = client.post(
        "/api/hr/cv/extract",
        headers={"X-Recruiter-ID": "recruiter-test"},
        files={"file": ("cv.txt", BytesIO(text.encode("utf-8")), "text/plain")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["candidate_name"] == "Derya Aksu"
    assert body["likely_position"] == "Backend Developer"
    assert "Python" in body["skills"]
    assert "FastAPI" in body["skills"]
    assert "Docker" in body["skills"]
    assert "PostgreSQL" in body["skills"]
    assert "deneyim" in body["experience_summary"].lower()


def test_cv_extract_no_name_does_not_crash(client: TestClient):
    text = "Bu bir CV metnidir. Herhangi bir aday adı içermez."
    r = client.post(
        "/api/hr/cv/extract",
        headers={"X-Recruiter-ID": "recruiter-test"},
        files={"file": ("cv.txt", BytesIO(text.encode("utf-8")), "text/plain")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "extracted_text" in body
    assert body["candidate_name"] == ""


def test_cv_extract_heuristic_name_fallback(client: TestClient):
    text = "Ayşe Kaya\nSoftware Engineer\nPython, Django, PostgreSQL kullanıyor."
    r = client.post(
        "/api/hr/cv/extract",
        headers={"X-Recruiter-ID": "recruiter-test"},
        files={"file": ("cv.txt", BytesIO(text.encode("utf-8")), "text/plain")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["candidate_name"] == "Ayşe Kaya"
    assert body["likely_position"] == "Software Engineer"


def test_agent_run_does_not_log_blank_candidate(client: TestClient):
    from src.core.schemas import AgentRole, AgentResponse, TaskStatus
    import src.pipeline.hr.audit_logger as al

    fake_response = AgentResponse(
        task_id="t-blank",
        agent_role=AgentRole.HR_AGENT,
        draft="Genel yanıt",
        status=TaskStatus.COMPLETED,
        metadata={
            "intent": "general",
            "token_remaining": 999,
            "structured_response": {
                "candidate_name": "",
                "job_title": "",
                "decision": "",
                "score": 0.0,
                "token_cost": 1,
                "automation_targets": [],
                "should_trigger_automation": False,
            },
        },
    )

    with patch("src.services.api.hr_router.run_hr_pipeline", return_value=fake_response):
        r = client.post(
            "/api/hr/agent/run",
            headers={"X-Recruiter-ID": "recruiter-blank"},
            json={"prompt": "değerlendir"},
        )
        assert r.status_code == 200

    events = al.read_candidate_events("recruiter-blank")
    assert events == []
