"""src/services/api/sprint_review_store.py — read/write helpers for the
`sprint_review` and `next_sprint_notes` blocks on a sprint_index entry.

Keeps cockpit_routes.py thin and provides one place to evolve the schema.
"""
from __future__ import annotations

from typing import Any

from src.loop import sprint_index


def get(sprint_id: str) -> dict[str, Any]:
    """Return the review payload (system_review, human_review, merged_notes,
    next_sprint_notes) for a sprint. Always returns a dict; missing keys are
    omitted."""
    entry = sprint_index.get_entry(sprint_id) or {}
    out: dict[str, Any] = {}
    for k in ("sprint_review", "next_sprint_notes"):
        v = entry.get(k)
        if v:
            out[k] = v
    return out


def update_review(sprint_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    """Merge `patch` into the sprint's `sprint_review` block."""
    entry = sprint_index.get_entry(sprint_id) or {}
    current = dict(entry.get("sprint_review") or {})
    current.update(patch or {})
    sprint_index.update_entry(sprint_id, {"sprint_review": current})
    return current


def set_next_sprint_notes(sprint_id: str, notes: dict[str, Any]) -> dict[str, Any]:
    sprint_index.update_entry(sprint_id, {"next_sprint_notes": notes or {}})
    return notes or {}


def get_review(sprint_id: str) -> dict[str, Any]:
    entry = sprint_index.get_entry(sprint_id) or {}
    return dict(entry.get("sprint_review") or {})


def get_next_sprint_notes(sprint_id: str) -> dict[str, Any]:
    entry = sprint_index.get_entry(sprint_id) or {}
    return dict(entry.get("next_sprint_notes") or {})
