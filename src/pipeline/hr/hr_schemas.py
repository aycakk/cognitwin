"""pipeline/hr/hr_schemas.py — Typed data contracts for the HR/Recruiter agent.

Isolated from core/schemas.py so HR concerns never bleed into student/agile paths.
All fields are explicit, typed, and carry defaults so partial initialization works.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ── Enumerations ──────────────────────────────────────────────────────────────

class SeniorityLevel(str, Enum):
    INTERN    = "intern"
    JUNIOR    = "junior"
    MID       = "mid"
    SENIOR    = "senior"
    LEAD      = "lead"
    PRINCIPAL = "principal"
    DIRECTOR  = "director"


class WorkMode(str, Enum):
    ONSITE = "onsite"
    REMOTE = "remote"
    HYBRID = "hybrid"
    ANY    = "any"


class TonePreference(str, Enum):
    FORMAL       = "formal"
    PROFESSIONAL = "professional"
    FRIENDLY     = "friendly"
    CASUAL       = "casual"


class FilterStrictness(str, Enum):
    STRICT   = "strict"    # every hard requirement must match
    MODERATE = "moderate"  # most requirements must match
    FLEXIBLE = "flexible"  # nice-to-have; compensation considered


# ── Token action costs (deducted from TokenLedger) ───────────────────────────

TOKEN_ACTION_COSTS: dict[str, int] = {
    "cv_summary":          10,
    "cv_parse":            15,
    "req_parse":           10,
    "candidate_match":     20,
    "shortlist_5":         50,
    "shortlist_10":        90,
    "interview_questions": 30,
    "outreach_draft":      25,
    "batch_rank_5":        80,
    "batch_rank_10":      150,
    "explanation":         15,
    "profile_update":       5,
}


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class RecruiterProfile:
    """Persistent personalization profile for an individual recruiter.

    Stored as JSON in data/hr_profiles/<recruiter_id>.json.
    Updated incrementally based on recruiter feedback and accepted/rejected decisions.
    """
    recruiter_id:         str                    = field(default_factory=lambda: str(uuid.uuid4()))
    name:                 str                    = ""
    company:              str                    = ""

    # Targeting preferences
    industry_focus:       list[str]              = field(default_factory=list)
    preferred_seniority:  list[str]              = field(default_factory=list)
    preferred_role_types: list[str]              = field(default_factory=list)
    work_mode_preference: str                    = WorkMode.ANY.value
    preferred_locations:  list[str]              = field(default_factory=list)

    # Communication preferences
    language_preference:  str                    = "tr"
    tone_preference:      str                    = TonePreference.PROFESSIONAL.value
    filter_strictness:    str                    = FilterStrictness.MODERATE.value
    shortlist_size:       int                    = 5

    # Memory: last 20 decisions with signal (accepted / rejected + reason)
    decision_history:     list[dict[str, Any]]   = field(default_factory=list)

    # Company-wide hiring rules (plain text, enforced via prompt injection)
    company_policies:     list[str]              = field(default_factory=list)

    # Shortlist feedback (accepted/rejected + comment)
    shortlist_feedback:   list[dict[str, Any]]   = field(default_factory=list)

    # Free-form recruiter notes
    notes:                str                    = ""


@dataclass
class JobRequisition:
    """A parsed job posting / vacancy."""
    req_id:           str             = field(default_factory=lambda: str(uuid.uuid4()))
    title:            str             = ""
    department:       str             = ""
    required_skills:  list[str]       = field(default_factory=list)
    nice_to_have:     list[str]       = field(default_factory=list)
    seniority:        str             = SeniorityLevel.MID.value
    work_mode:        str             = WorkMode.HYBRID.value
    location:         str             = ""
    years_experience: int             = 0
    education_level:  str             = ""
    salary_range:     str             = ""
    description:      str             = ""
    raw_text:         str             = ""


@dataclass
class CandidateProfile:
    """Normalized candidate extracted from a CV."""
    candidate_id:   str                   = field(default_factory=lambda: str(uuid.uuid4()))
    name:           str                   = ""
    email:          str                   = ""
    phone:          str                   = ""
    skills:         list[str]             = field(default_factory=list)
    years_experience: int                 = 0
    seniority:      str                   = SeniorityLevel.MID.value
    education:      list[str]             = field(default_factory=list)
    work_history:   list[dict[str, Any]]  = field(default_factory=list)
    location:       str                   = ""
    work_mode_open: str                   = WorkMode.ANY.value
    languages:      list[str]             = field(default_factory=list)
    raw_cv_text:    str                   = ""
    summary:        str                   = ""


@dataclass
class CandidateMatchResult:
    """Scored and explained match between a candidate and a requisition."""
    match_id:        str        = field(default_factory=lambda: str(uuid.uuid4()))
    candidate_id:    str        = ""
    candidate_name:  str        = ""
    req_id:          str        = ""
    overall_score:   float      = 0.0
    skill_score:     float      = 0.0
    seniority_score: float      = 0.0
    location_score:  float      = 0.0
    education_score: float      = 0.0
    matched_skills:  list[str]  = field(default_factory=list)
    missing_skills:  list[str]  = field(default_factory=list)
    bonus_skills:    list[str]  = field(default_factory=list)
    explanation:     str        = ""   # why matched / why not
    recruiter_angle: str        = ""   # how recruiter profile shaped the result
    recommended:     bool       = False


@dataclass
class ShortlistResult:
    """Ranked shortlist of candidates for a requisition."""
    shortlist_id:          str                        = field(default_factory=lambda: str(uuid.uuid4()))
    req_id:                str                        = ""
    recruiter_id:          str                        = ""
    candidates:            list[CandidateMatchResult] = field(default_factory=list)
    generated_at:          str                        = ""
    personalization_notes: str                        = ""


@dataclass
class InterviewPlan:
    """Interview question set tailored to a candidate + requisition."""
    plan_id:              str        = field(default_factory=lambda: str(uuid.uuid4()))
    candidate_id:         str        = ""
    req_id:               str        = ""
    technical_questions:  list[str]  = field(default_factory=list)
    behavioral_questions: list[str]  = field(default_factory=list)
    gap_probe_questions:  list[str]  = field(default_factory=list)
    culture_questions:    list[str]  = field(default_factory=list)
    notes:                str        = ""


@dataclass
class OutreachDraft:
    """Drafted outreach message from recruiter to candidate."""
    draft_id:        str  = field(default_factory=lambda: str(uuid.uuid4()))
    candidate_id:    str  = ""
    req_id:          str  = ""
    subject:         str  = ""
    body:            str  = ""
    tone:            str  = TonePreference.PROFESSIONAL.value
    language:        str  = "tr"
    personalization: str  = ""   # what was personalized and why


@dataclass
class TokenLedger:
    """Token/usage budget for a recruiter session.

    Stored in data/hr_profiles/<recruiter_id>_ledger.json.
    """
    recruiter_id:  str              = ""
    total_budget:  int              = 1000
    used_budget:   int              = 0
    transactions:  list[dict]       = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(0, self.total_budget - self.used_budget)

    def can_afford(self, cost: int) -> bool:
        return self.remaining >= cost

    def deduct(self, cost: int, action: str, note: str = "") -> None:
        self.used_budget += cost
        self.transactions.append({
            "action":    action,
            "cost":      cost,
            "note":      note,
            "remaining": self.remaining,
        })


@dataclass
class HRSessionContext:
    """Short-term session memory for the HR agent (per-conversation state)."""
    session_id:           str                        = ""
    recruiter_id:         str                        = ""
    current_req:          Optional[JobRequisition]   = None
    candidates:           list[CandidateProfile]     = field(default_factory=list)
    last_shortlist:       Optional[ShortlistResult]  = None
    last_action:          str                        = ""
    conversation_turns:   int                        = 0


# ── n8n automation integration ────────────────────────────────────────────────

@dataclass
class HRStructuredResponse:
    """Backend-side structured result extracted from the HR agent LLM output.

    Never shown raw in LibreChat — the `text_response` field is what the user
    sees.  The structured fields are used by the n8n webhook layer.

    Fields are populated best-effort via regex extraction from `text_response`.
    Unknown/unparseable fields stay at their defaults.
    """
    intent:                    str        = ""
    decision:                  str        = ""   # Önerilir / Şartlı Önerilir / Önerilmez
    candidate_name:            str        = ""
    job_title:                 str        = ""
    score:                     float      = 0.0
    strengths:                 list[str]  = field(default_factory=list)
    missing_skills:            list[str]  = field(default_factory=list)
    risks:                     str        = ""
    shortlist_status:          str        = ""
    automation_targets:        list[str]  = field(default_factory=list)
    recommended_actions:       list[str]  = field(default_factory=list)
    follow_up_actions:         list[str]  = field(default_factory=list)
    should_trigger_automation: bool       = False
    recruiter_summary:         str        = ""
    token_cost:                int        = 0
    remaining_budget:          int        = 0
    text_response:             str        = ""   # full Turkish LLM text for LibreChat


@dataclass
class N8nWebhookPayload:
    """JSON payload POSTed to an n8n webhook endpoint.

    Built from `HRStructuredResponse` by `n8n_client.build_payload()`.
    One payload per automation action (e.g. shortlist triggers two payloads:
    'shortlist_to_sheets' and 'notify_slack').
    """
    action_type:      str        = ""   # e.g. "shortlist_to_sheets"
    intent:           str        = ""
    recruiter_id:     str        = ""
    session_id:       str        = ""
    decision:         str        = ""
    candidate_name:   str        = ""
    candidate_id:     str        = ""
    job_title:        str        = ""
    job_id:           str        = ""
    score:            float      = 0.0
    strengths:        list[str]  = field(default_factory=list)
    missing_skills:   list[str]  = field(default_factory=list)
    risks:            str        = ""
    shortlist_status: str        = ""
    text_response:    str        = ""   # truncated to 2000 chars
    token_cost:       int        = 0
    remaining_budget: int        = 0
    source:           str        = "cognitwin_hr_agent"
    slack_text:       str        = ""
    extra:            dict       = field(default_factory=dict)


@dataclass
class HRAutomationAction:
    """Validated automation action planned by backend for n8n dispatch."""
    action_type:      str = ""
    intent:           str = ""
    should_dispatch:  bool = False
    reason:           str = ""


# ── Intent → automation action mapping ───────────────────────────────────────
# Keys are hr_runner intent names; values are the n8n action_type strings
# that should be fired after a successful response for that intent.
# Empty list = no automation for this intent.

INTENT_AUTOMATION_MAP: dict[str, list[str]] = {
    "shortlist":           ["shortlist_to_sheets", "notify_slack"],
    "outreach_draft":      ["send_outreach_email"],
    "interview_questions": ["create_calendar_event"],
    "candidate_match":     ["log_to_ats", "notify_slack"],
    "cv_analyze":          [],
    "req_parse":           [],
    "missing_skills":      [],
    "budget_status":       [],
    "profile_show":        [],
    "profile_update":      [],
    "recruiter_history":   [],
    "audit_show":          [],
    "cost_table":          [],
    "general":             ["notify_slack"],  # only fires when explicitly requested
}
