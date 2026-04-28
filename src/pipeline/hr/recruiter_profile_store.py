"""pipeline/hr/recruiter_profile_store.py — Persistent recruiter profile storage.

Profiles are stored as JSON files in data/hr_profiles/.
One file per recruiter_id.  Thread-safe for single-process use.

Design mirrors DeveloperProfileStore: JSON-backed, disk-persisted, lazy-loaded.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from src.pipeline.hr.hr_schemas import (
    RecruiterProfile,
    TonePreference,
    FilterStrictness,
    WorkMode,
)

logger = logging.getLogger(__name__)

_DATA_DIR = Path("data/hr_profiles")
_MAX_DECISION_HISTORY = 20
_MAX_SHORTLIST_FEEDBACK = 30

_lock = threading.Lock()


def _profile_path(recruiter_id: str) -> Path:
    return _DATA_DIR / f"{recruiter_id}.json"


def _ensure_dir() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_profile(recruiter_id: str) -> RecruiterProfile:
    """Load profile from disk or return a fresh default."""
    path = _profile_path(recruiter_id)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            profile = RecruiterProfile(**{
                k: v for k, v in raw.items()
                if k in RecruiterProfile.__dataclass_fields__
            })
            profile.recruiter_id = recruiter_id
            return profile
        except Exception as exc:
            logger.warning("profile load failed for %s: %s — using default", recruiter_id, exc)
    return RecruiterProfile(recruiter_id=recruiter_id)


def save_profile(profile: RecruiterProfile) -> None:
    """Persist profile to disk."""
    _ensure_dir()
    path = _profile_path(profile.recruiter_id)
    with _lock:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(profile), f, ensure_ascii=False, indent=2)


def update_from_feedback(
    profile: RecruiterProfile,
    *,
    candidate_name: str,
    decision: str,       # "accepted" | "rejected"
    reason: str = "",
    req_title: str = "",
) -> RecruiterProfile:
    """Record a recruiter decision and update the profile accordingly.

    Keeps only the last MAX_DECISION_HISTORY entries.
    """
    entry = {
        "candidate_name": candidate_name,
        "decision":       decision,
        "reason":         reason,
        "req_title":      req_title,
    }
    profile.decision_history.append(entry)
    if len(profile.decision_history) > _MAX_DECISION_HISTORY:
        profile.decision_history = profile.decision_history[-_MAX_DECISION_HISTORY:]
    save_profile(profile)
    return profile


def update_preferences(
    profile: RecruiterProfile,
    *,
    tone: Optional[str] = None,
    strictness: Optional[str] = None,
    work_mode: Optional[str] = None,
    shortlist_size: Optional[int] = None,
    locations: Optional[list[str]] = None,
    seniority: Optional[list[str]] = None,
    role_types: Optional[list[str]] = None,
    industry: Optional[list[str]] = None,
    policies: Optional[list[str]] = None,
    notes: Optional[str] = None,
) -> RecruiterProfile:
    """Patch recruiter preferences in-place and save."""
    if tone is not None:
        profile.tone_preference = tone
    if strictness is not None:
        profile.filter_strictness = strictness
    if work_mode is not None:
        profile.work_mode_preference = work_mode
    if shortlist_size is not None:
        profile.shortlist_size = max(1, min(shortlist_size, 20))
    if locations is not None:
        profile.preferred_locations = locations
    if seniority is not None:
        profile.preferred_seniority = seniority
    if role_types is not None:
        profile.preferred_role_types = role_types
    if industry is not None:
        profile.industry_focus = industry
    if policies is not None:
        profile.company_policies = policies
    if notes is not None:
        profile.notes = notes
    save_profile(profile)
    return profile


def record_shortlist_feedback(
    profile: RecruiterProfile,
    *,
    shortlist_id: str,
    accepted_ids: list[str],
    rejected_ids: list[str],
    comment: str = "",
) -> RecruiterProfile:
    """Store shortlist-level feedback signal."""
    profile.shortlist_feedback.append({
        "shortlist_id":  shortlist_id,
        "accepted_ids":  accepted_ids,
        "rejected_ids":  rejected_ids,
        "comment":       comment,
    })
    if len(profile.shortlist_feedback) > _MAX_SHORTLIST_FEEDBACK:
        profile.shortlist_feedback = profile.shortlist_feedback[-_MAX_SHORTLIST_FEEDBACK:]
    save_profile(profile)
    return profile


def build_profile_summary(profile: RecruiterProfile) -> str:
    """Return a concise text block injected into the HR agent's system prompt."""
    lines = [
        f"Recruiter: {profile.name or 'Unknown'} @ {profile.company or 'Unknown company'}",
        f"Industry focus: {', '.join(profile.industry_focus) or 'any'}",
        f"Preferred seniority: {', '.join(profile.preferred_seniority) or 'any'}",
        f"Preferred role types: {', '.join(profile.preferred_role_types) or 'any'}",
        f"Work mode preference: {profile.work_mode_preference}",
        f"Preferred locations: {', '.join(profile.preferred_locations) or 'any'}",
        f"Communication tone: {profile.tone_preference}",
        f"Filter strictness: {profile.filter_strictness}",
        f"Default shortlist size: {profile.shortlist_size}",
        f"Language: {profile.language_preference}",
    ]
    if profile.company_policies:
        lines.append("Company policies:")
        for p in profile.company_policies:
            lines.append(f"  - {p}")
    if profile.decision_history:
        last = profile.decision_history[-5:]
        lines.append("Recent decisions (last 5):")
        for d in last:
            lines.append(
                f"  [{d.get('decision','?').upper()}] {d.get('candidate_name','?')} "
                f"for {d.get('req_title','?')}: {d.get('reason','')}"
            )
    if profile.notes:
        lines.append(f"Recruiter notes: {profile.notes}")
    return "\n".join(lines)
