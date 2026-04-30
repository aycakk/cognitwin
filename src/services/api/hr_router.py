"""services/api/hr_router.py — Personal HR Panel API endpoints.

Each recruiter identifies themselves via the X-Recruiter-ID request header.
All storage is scoped to that recruiter_id so records never cross boundaries.

LibreChat /chat is untouched — this router is additive only.
"""
from __future__ import annotations

import io
import re
import uuid
from dataclasses import asdict
from typing import List, Optional

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from pydantic import BaseModel

from src.core.schemas import AgentRole, AgentTask
from src.pipeline.hr.audit_logger import (
    read_audit_tail,
    log_candidate_event,
    log_automation_event,
    read_candidate_events,
    read_automation_events,
)
from src.pipeline.hr.recruiter_profile_store import load_profile, save_profile, update_preferences
from src.pipeline.hr.token_ledger import load_ledger
from src.pipeline.hr_runner import run_hr_pipeline

hr_router = APIRouter(tags=["hr"])

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_MAX_CV_SIZE = 10 * 1024 * 1024  # 10 MB


def _require_recruiter_id(x_recruiter_id: Optional[str]) -> str:
    if not x_recruiter_id:
        raise HTTPException(status_code=400, detail="X-Recruiter-ID header is required.")
    if not _SAFE_ID_RE.match(x_recruiter_id):
        raise HTTPException(
            status_code=400,
            detail="X-Recruiter-ID must be 1–64 chars: letters, digits, hyphens, or underscores.",
        )
    return x_recruiter_id


# ── POST /api/hr/agent/run ────────────────────────────────────────────────────

class AgentRunRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None


@hr_router.post("/agent/run")
async def agent_run(
    body: AgentRunRequest,
    x_recruiter_id: Optional[str] = Header(default=None),
):
    """Run the HR agent for a specific recruiter."""
    recruiter_id = _require_recruiter_id(x_recruiter_id)
    session_id = body.session_id or f"hr-panel-{recruiter_id}-{uuid.uuid4().hex[:8]}"

    task = AgentTask(
        session_id=session_id,
        role=AgentRole.HR_AGENT,
        masked_input=body.prompt,
        metadata={"recruiter_id": recruiter_id},
    )
    response = run_hr_pipeline(task)

    # ── Persist recruiter-scoped logs ─────────────────────────────────────────
    meta = response.metadata or {}
    structured = meta.get("structured_response", {})
    candidate_name  = structured.get("candidate_name", "")
    job_title       = structured.get("job_title", "")
    decision        = structured.get("decision", "")
    score           = float(structured.get("score", 0.0))
    action_type     = meta.get("intent", "")
    token_cost      = int(structured.get("token_cost", 0))
    remaining       = int(meta.get("token_remaining", 0))
    automation_targets = structured.get("automation_targets", [])
    should_trigger  = structured.get("should_trigger_automation", False)

    has_candidate_signal = any([
        str(candidate_name or "").strip(),
        str(job_title or "").strip() and str(job_title).strip().lower() != "belirtilmedi",
        str(decision or "").strip(),
        float(score or 0) > 0,
    ])
    if has_candidate_signal:
        log_candidate_event(
            recruiter_id,
            candidate_name=candidate_name,
            job_title=job_title,
            decision=decision,
            score=score,
            action_type=action_type,
            automation_status="triggered" if should_trigger else "none",
            token_cost=token_cost,
            remaining_budget=remaining,
        )

    if should_trigger and automation_targets:
        for at in automation_targets:
            log_automation_event(
                recruiter_id,
                action_type=at,
                candidate_name=candidate_name,
                job_title=job_title,
                status="dispatched",
            )
    # ─────────────────────────────────────────────────────────────────────────

    return {
        "session_id": session_id,
        "recruiter_id": recruiter_id,
        "response": response.draft,
        "status": response.status,
        "metadata": response.metadata,
    }


def _friendly_cv_error() -> dict:
    return {
        "status": "error",
        "message": "CV içeriği okunamadı. Lütfen farklı bir dosya deneyin veya metni manuel ekleyin.",
    }


# ── CV metadata extraction helpers ───────────────────────────────────────────

_NAME_PREFIXES = re.compile(
    r"^(Aday|Ad Soyad|Ad ve Soyad|İsim|Name|Full Name)\s*[:：]\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_POSITION_PREFIXES = re.compile(
    r"^(Pozisyon|Başvurulan Pozisyon|Hedef Pozisyon|Position|Role|Unvan|Görev)\s*[:：]\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_EXPERIENCE_RE = re.compile(
    r"[^\n.]*\d+\s*(?:yıl|year)[^\n.]*(?:deneyim|experience)[^\n.]*",
    re.IGNORECASE,
)
_NAME_WORD = re.compile(r"^[A-ZÇĞİÖŞÜ][a-zçğışöü]+$")
_NOT_NAME = re.compile(
    r"\b(developer|engineer|manager|analyst|designer|intern|stajyer|"
    r"pozisyon|position|role|experience|deneyim|yıl|year|university|üniversite)\b",
    re.IGNORECASE,
)
_KNOWN_POSITIONS = [
    "Backend Developer", "Frontend Developer", "Full Stack Developer",
    "Full-Stack Developer", "Data Analyst", "Data Scientist",
    "Product Manager", "QA Engineer", "DevOps Engineer",
    "Mobile Developer", "Software Engineer", "Yazılım Mühendisi",
    "Yazılım Stajyeri", "Stajyer", "Intern", "Machine Learning Engineer",
    "Cloud Engineer", "Security Engineer", "Android Developer", "iOS Developer",
]
_KNOWN_SKILLS = [
    "Python", "FastAPI", "Django", "Flask", "JavaScript", "TypeScript",
    "React", "Vue", "Angular", "Node.js", "Java", "Spring", "Kotlin",
    "Swift", "Go", "Rust", "C#", ".NET", "PHP", "Ruby",
    "Docker", "Kubernetes", "AWS", "Azure", "GCP", "PostgreSQL",
    "MySQL", "MongoDB", "Redis", "Elasticsearch", "Kafka", "RabbitMQ",
    "Git", "Linux", "Terraform", "SQL", "GraphQL", "REST", "gRPC",
    "pandas", "NumPy", "scikit-learn", "TensorFlow", "PyTorch",
]


def _skill_in_text(skill: str, text_lower: str) -> bool:
    s = skill.lower()
    if re.search(r"[^a-zA-Z]", s):
        return s in text_lower
    return bool(re.search(r"\b" + re.escape(s) + r"\b", text_lower))


def _parse_cv_metadata(text: str) -> dict:
    candidate_name = ""
    likely_position = ""
    skills: list[str] = []
    experience_summary = ""

    m = _NAME_PREFIXES.search(text)
    if m:
        candidate_name = m.group(2).strip()
    else:
        for line in text.splitlines():
            line = line.strip()
            words = line.split()
            if (
                2 <= len(words) <= 4
                and all(_NAME_WORD.match(w) for w in words)
                and not _NOT_NAME.search(line)
            ):
                candidate_name = line
                break

    m = _POSITION_PREFIXES.search(text)
    if m:
        likely_position = m.group(2).strip()
    else:
        text_lower = text.lower()
        for pos in _KNOWN_POSITIONS:
            if pos.lower() in text_lower:
                likely_position = pos
                break

    text_lower = text.lower()
    skills = [s for s in _KNOWN_SKILLS if _skill_in_text(s, text_lower)]

    m = _EXPERIENCE_RE.search(text)
    if m:
        experience_summary = m.group(0).strip()

    return {
        "candidate_name": candidate_name,
        "likely_position": likely_position,
        "skills": skills,
        "experience_summary": experience_summary,
    }


def _extract_txt(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")


def _extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    try:
        reader = PdfReader(io.BytesIO(data))
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)
    except Exception:
        return ""


def _extract_docx(data: bytes) -> str:
    try:
        from docx import Document
    except Exception:
        return ""
    try:
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text)
    except Exception:
        return ""


@hr_router.post("/cv/extract")
async def cv_extract(
    file: UploadFile = File(...),
    x_recruiter_id: Optional[str] = Header(default=None),
):
    recruiter_id = _require_recruiter_id(x_recruiter_id)
    _ = recruiter_id  # recruiter scope validation only

    filename = file.filename or "dosya"
    content_type = file.content_type or "application/octet-stream"
    raw = await file.read()
    size_bytes = len(raw)

    if size_bytes <= 0:
        return _friendly_cv_error()
    if size_bytes > _MAX_CV_SIZE:
        raise HTTPException(status_code=413, detail="Dosya boyutu sınırı aşıldı (10 MB).")

    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    extracted = ""
    if ext == "txt":
        extracted = _extract_txt(raw)
    elif ext == "pdf":
        extracted = _extract_pdf(raw)
    elif ext == "docx":
        extracted = _extract_docx(raw)

    extracted = (extracted or "").strip()
    if not extracted:
        return _friendly_cv_error()

    preview = extracted[:400]
    meta = _parse_cv_metadata(extracted)
    return {
        "filename": filename,
        "size_bytes": size_bytes,
        "content_type": content_type,
        "extracted_text": extracted,
        "text_preview": preview,
        "candidate_name": meta["candidate_name"],
        "likely_position": meta["likely_position"],
        "skills": meta["skills"],
        "experience_summary": meta["experience_summary"],
        "status": "ok",
    }


# ── GET /api/hr/dashboard/me ──────────────────────────────────────────────────

@hr_router.get("/dashboard/me")
async def dashboard_me(
    x_recruiter_id: Optional[str] = Header(default=None),
):
    """Return token usage summary and recent activity for the caller."""
    recruiter_id = _require_recruiter_id(x_recruiter_id)
    ledger = load_ledger(recruiter_id)
    recent = read_audit_tail(recruiter_id, n=10)
    return {
        "recruiter_id": recruiter_id,
        "token_budget": ledger.total_budget,
        "token_used": ledger.used_budget,
        "token_remaining": ledger.remaining,
        "recent_actions": recent,
    }


# ── GET /api/hr/candidates/me ─────────────────────────────────────────────────

@hr_router.get("/candidates/me")
async def candidates_me(
    x_recruiter_id: Optional[str] = Header(default=None),
):
    """Return this recruiter's candidate evaluation log."""
    recruiter_id = _require_recruiter_id(x_recruiter_id)
    return {
        "recruiter_id": recruiter_id,
        "candidates": read_candidate_events(recruiter_id),
    }


# ── GET /api/hr/automation-history/me ────────────────────────────────────────

@hr_router.get("/automation-history/me")
async def automation_history_me(
    x_recruiter_id: Optional[str] = Header(default=None),
    n: int = 20,
):
    """Return the last n automation dispatches for this recruiter."""
    recruiter_id = _require_recruiter_id(x_recruiter_id)
    return {
        "recruiter_id": recruiter_id,
        "automation_history": read_automation_events(recruiter_id, n=n),
    }


# ── GET /api/hr/recruiter-profile/me ─────────────────────────────────────────

@hr_router.get("/recruiter-profile/me")
async def get_recruiter_profile(
    x_recruiter_id: Optional[str] = Header(default=None),
):
    """Return the full recruiter profile."""
    recruiter_id = _require_recruiter_id(x_recruiter_id)
    profile = load_profile(recruiter_id)
    return asdict(profile)


# ── PUT /api/hr/recruiter-profile/me ─────────────────────────────────────────

class ProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    company: Optional[str] = None
    tone: Optional[str] = None
    strictness: Optional[str] = None
    work_mode: Optional[str] = None
    shortlist_size: Optional[int] = None
    locations: Optional[List[str]] = None
    seniority: Optional[List[str]] = None
    role_types: Optional[List[str]] = None
    industry: Optional[List[str]] = None
    policies: Optional[List[str]] = None
    notes: Optional[str] = None


@hr_router.put("/recruiter-profile/me")
async def update_recruiter_profile(
    body: ProfileUpdateRequest,
    x_recruiter_id: Optional[str] = Header(default=None),
):
    """Patch recruiter preferences and persist them."""
    recruiter_id = _require_recruiter_id(x_recruiter_id)
    profile = load_profile(recruiter_id)

    if body.name is not None:
        profile.name = body.name
    if body.company is not None:
        profile.company = body.company

    update_preferences(
        profile,
        tone=body.tone,
        strictness=body.strictness,
        work_mode=body.work_mode,
        shortlist_size=body.shortlist_size,
        locations=body.locations,
        seniority=body.seniority,
        role_types=body.role_types,
        industry=body.industry,
        policies=body.policies,
        notes=body.notes,
    )
    # name/company are not covered by update_preferences, so save explicitly
    save_profile(profile)
    return asdict(profile)
