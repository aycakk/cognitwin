"""services/api/student_agent_routes.py — architecture-native Student Agent API.

Endpoints (single-user, no auth):

    GET  /api/student/agent/status     -> subsystem availability for header chips
    POST /api/student/agent/chat       -> structured agent response with memory,
                                          ontology, gates, footprint, study output
    POST /api/student/agent/note       -> persist a study note to data/student_notes.json

This router wraps the existing ZT4SWE pipeline (`run_pipeline`) and exposes the
internal grounding signals that `/v1/chat/completions` collapses away. It does
NOT replace the lightweight Lumi router (`/api/student/*`) — the two coexist.

All response strings on this surface are English. The model's free-text
`answer` field may still be Turkish (the StudentAgent system prompt enforces
Turkish responses); a `language_detected` field is provided so the UI can show
a one-line note when that happens.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

student_agent_router = APIRouter()

# ─────────────────────────────────────────────────────────────────────────────
#  Friendly labels — server-side only so the UI doesn't duplicate the mapping.
# ─────────────────────────────────────────────────────────────────────────────

GATE_LABELS: dict[str, str] = {
    "C1":      "PII safety",
    "C2":      "Grounding",
    "C2_DEV":  "Code grounding",
    "C3":      "Ontology compliance",
    "C3_AGILE": "Agile compliance",
    "C4":      "Hallucination check",
    "C5":      "Role compliance",
    "C6":      "Anti-sycophancy",
    "C7":      "Blindspot disclosure",
    "C8":      "Acceptance criteria",
    "A1":      "Stability",
}

# Per-gate guidance shown in the "main warning / recommended action" strip.
GATE_FAIL_GUIDANCE: dict[str, tuple[str, str]] = {
    "C1": ("Raw PII detected in the answer.",
           "Do not share this answer outside the workspace."),
    "C2": ("Answer is not grounded in your footprint memory.",
           "Try rephrasing or pick a topic from your recent courses."),
    "C3": ("Answer conflicts with the academic ontology.",
           "Re-ask using terminology from your course list."),
    "C4": ("Possible hallucination markers detected.",
           "Cross-check the answer with your textbook before relying on it."),
    "C5": ("Role boundary violation in the answer.",
           "Ask only about your own academic context."),
    "C6": ("Answer style flagged as sycophantic.",
           "Re-ask requesting a critical, neutral response."),
    "C7": ("Empty memory but missing blindspot disclosure.",
           "Treat the answer as unverified."),
    "A1": ("REDO loop did not close cleanly.",
           "Retry the question."),
}

# Heuristic regexes for footprint summary extraction (over masked text).
_RX_COURSE  = re.compile(r"\b(?:[A-Z]{2,4}\s?\d{3,4}|[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü]+\s+\d{2,4}|MATH|PHYS|HIST|CHEM|BIO|CS|EE)\b")
_RX_DATE    = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?)\b")
_RX_UPCOMING = re.compile(r"\b(?:exam|midterm|final|quiz|assignment|due|deadline|sınav|ödev|teslim)\b", re.I)
_RX_WEAK    = re.compile(r"\b(?:weak|low|fail|missed|absent|incomplete|zayıf|eksik)\b", re.I)

# ── Snippet-cleanup regexes ─────────────────────────────────────────────────
# Structured footprint syntax: [TAG][timestamp] src=... course=... text=...
_RX_TAG_PREFIX  = re.compile(r"^\s*\[(?P<tag>[A-Z_]+)\](?:\[(?P<ts>[^\]]+)\])?\s*", re.I)
_RX_KV_PAIR     = re.compile(r"(?P<key>\w+)\s*=\s*(?P<val>\"[^\"]*\"|'[^']*'|\S+)")
_RX_BLINDSPOT_BOX = re.compile(r"^[\s┌│└─━╔╚╗╝║═]+.*$", re.M)
_TAG_TO_TYPE: dict[str, str] = {
    "EMAIL": "email", "LMS": "lms", "MAIL": "email", "NOTE": "note",
    "EXAM": "exam", "QUIZ": "quiz", "ASSIGNMENT": "assignment",
    "REMINDER": "reminder", "LECTURE": "lecture", "FOOTPRINT": "footprint",
}

# Turkish narrative footprint line shapes (from src/data/masked/footprints_masked.txt).
_RX_NARR_DATE   = re.compile(r"^(\d{4}-\d{2}-\d{2})\s*[:：]\s*")
_RX_NARR_EMAIL  = re.compile(r"^E[-]?posta\s*[:：]\s*", re.I)
_RX_NARR_REMIND = re.compile(r"^Hatırlatma\s*[:：]\s*", re.I)
_RX_NARR_LECT   = re.compile(r"^Ders\s*[:：]\s*", re.I)
_RX_NARR_USER   = re.compile(r"^\[USER_NAME_MASKED\]\s*[:：]\s*")

# Study-output filtering.
_TOPIC_BLOCKLIST: set[str] = {
    "course", "exam", "result", "vector memory", "ontology", "memory",
    "ontoloji", "sınav", "ders", "sonuç", "blindspot", "person", "agent",
    "role", "footprint",
}
_TR_MONTHS: set[str] = {
    "ocak", "şubat", "mart", "nisan", "mayıs", "haziran",
    "temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık",
}
_RX_DATE_FRAGMENT = re.compile(
    r"^\s*(?:\d{1,2}\s+\S+|\S+\s+\d{1,2}|\d{4}-\d{2}-\d{2}|\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?)\s*$",
    re.U,
)
_ACADEMIC_KEYWORDS: list[str] = [
    "Newton", "thermodynamics", "friction", "momentum", "energy",
    "midterm", "final exam", "quiz", "lab", "assignment", "homework",
    "weekly report", "chapter", "problem set", "derivative", "integral",
    "matrix", "algorithm", "complexity",
]

# Sentence (or short paragraph) the StudentAgent emits when memory empty.
_NO_MEMORY_PHRASE = "Bunu hafızamda bulamadım."
_FALLBACK_NOTE = "I found related memory records, but some details may be incomplete."


# ─────────────────────────────────────────────────────────────────────────────
#  Request / response models
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    mode: str = Field(default="ask")
    topic_hint: Optional[str] = None


class NoteRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    body: str = Field(..., min_length=1)


# ─────────────────────────────────────────────────────────────────────────────
#  Subsystem probes (each fully isolated; never raise)
# ─────────────────────────────────────────────────────────────────────────────

def _project_root() -> Path:
    # src/services/api/student_agent_routes.py → project root is 3 levels up.
    return Path(__file__).resolve().parents[3]


def _footprint_status() -> dict[str, Any]:
    root = _project_root()
    masked = root / "src" / "data" / "masked" / "footprints_masked.txt"
    raw = root / "src" / "data" / "footprints.txt"
    chosen: Optional[Path] = None
    is_masked = False
    if masked.exists():
        chosen, is_masked = masked, True
    elif raw.exists():
        chosen, is_masked = raw, False
    if chosen is None:
        return {"available": False, "file": None, "record_count": 0, "masked": False}
    try:
        with chosen.open("r", encoding="utf-8", errors="replace") as f:
            count = sum(1 for line in f if line.strip())
    except Exception:
        count = 0
    return {
        "available": True,
        "file": str(chosen.relative_to(root)),
        "record_count": count,
        "masked": is_masked,
    }


def _chroma_status() -> dict[str, Any]:
    try:
        from src.database.chroma_manager import db_manager
        try:
            doc_count = db_manager.collection.count()
        except Exception:
            doc_count = None
        return {"available": True, "namespace": "academic", "doc_count": doc_count}
    except Exception as exc:
        logger.info("Chroma probe failed: %s", exc)
        return {"available": False, "namespace": "academic", "doc_count": None}


def _ontology_status() -> dict[str, Any]:
    try:
        from src.pipeline.shared import build_ontology_context
        block = build_ontology_context()
        if not block or "[ONTOLOGY: unavailable]" in block:
            return {"available": False, "blocks_detected": 0}
        # Count individual entries (each starts with a "  [" prefix).
        detected = sum(1 for line in block.splitlines() if line.startswith("  ["))
        return {"available": True, "blocks_detected": detected}
    except Exception as exc:
        logger.info("Ontology probe failed: %s", exc)
        return {"available": False, "blocks_detected": 0}


def _gates_status() -> dict[str, Any]:
    try:
        from src.gates.evaluator import GATE_POLICY
        active = list(GATE_POLICY.get("StudentAgent", []))
        return {"available": True, "active_gates": active}
    except Exception as exc:
        logger.info("Gates probe failed: %s", exc)
        return {"available": False, "active_gates": []}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers — parsing pipeline outputs into UI-friendly shapes
# ─────────────────────────────────────────────────────────────────────────────

def _clean_snippet(raw: str) -> tuple[str, dict[str, Any]]:
    """Strip raw footprint syntax → readable body + parsed metadata.

    Handles two shapes:
      1) Structured: ``[EMAIL][2026-04-01] src=lms course=PHYS101 text=Newton``
      2) Turkish narrative: ``2026-02-10: COM 8090 dersi …``,
         ``E-posta: …``, ``Hatırlatma: …``, ``Ders: …``, ``[USER_NAME_MASKED]: …``

    Returns ``(body, meta)`` where ``meta`` may carry
    ``{source_type, timestamp, course_or_topic}``.
    """
    if not raw:
        return "", {}
    text = raw.strip()
    meta: dict[str, Any] = {}

    # ── Structured form first ──────────────────────────────────────────────
    m_tag = _RX_TAG_PREFIX.match(text)
    if m_tag:
        tag = (m_tag.group("tag") or "").upper()
        ts = m_tag.group("ts")
        if tag and tag in _TAG_TO_TYPE:
            meta["source_type"] = _TAG_TO_TYPE[tag]
        if ts:
            meta["timestamp"] = ts.strip()
        rest = text[m_tag.end():]
        # `text=...` is multi-word and conventionally trailing; everything
        # after it belongs to the body. Split there first.
        m_text = re.search(r"\btext\s*=\s*", rest, re.I)
        if m_text:
            head, body = rest[:m_text.start()], rest[m_text.end():]
        else:
            head, body = rest, ""
        # Parse single-word kv pairs from the head (src=, course=, etc).
        for m_kv in _RX_KV_PAIR.finditer(head):
            key = m_kv.group("key").lower()
            val = m_kv.group("val").strip().strip("\"'")
            if key == "course":
                meta["course_or_topic"] = val
        # Strip parsed kv pairs out of the head; whatever remains joins body.
        head_remainder = _RX_KV_PAIR.sub("", head).strip()
        text = (head_remainder + " " + body).strip() if body else head_remainder

    # ── Turkish narrative shapes ───────────────────────────────────────────
    m_date = _RX_NARR_DATE.match(text)
    if m_date and "timestamp" not in meta:
        meta["timestamp"] = m_date.group(1)
        meta.setdefault("source_type", "footprint")
        text = text[m_date.end():]
    elif _RX_NARR_EMAIL.match(text):
        meta.setdefault("source_type", "email")
        text = _RX_NARR_EMAIL.sub("", text, count=1)
    elif _RX_NARR_REMIND.match(text):
        meta.setdefault("source_type", "reminder")
        text = _RX_NARR_REMIND.sub("", text, count=1)
    elif _RX_NARR_LECT.match(text):
        meta.setdefault("source_type", "lecture")
        text = _RX_NARR_LECT.sub("", text, count=1)
    elif _RX_NARR_USER.match(text):
        meta.setdefault("source_type", "note")
        text = _RX_NARR_USER.sub("", text, count=1)

    # Default source type if nothing matched.
    meta.setdefault("source_type", "footprint")

    # Course code if we haven't grabbed one already.
    if "course_or_topic" not in meta:
        m_course = _RX_COURSE.search(text)
        if m_course:
            meta["course_or_topic"] = m_course.group(0)

    # Timestamp if we haven't grabbed one already.
    if "timestamp" not in meta:
        m_ts = _RX_DATE.search(text)
        if m_ts:
            meta["timestamp"] = m_ts.group(0)

    # Final cosmetic cleanup: collapse whitespace; drop stray bracket leftovers.
    text = re.sub(r"\s+", " ", text).strip(" .,-—:;")
    return text, meta


def _split_vector_context(context_block: str) -> list[dict[str, Any]]:
    """Parse the VectorMemory.retrieve() context block into source cards.

    Drops empty/near-empty entries, deduplicates by normalized body, and
    renumbers ranks ``1..N`` so the UI shows clean, sequential cards.

    The block format (see src/pipeline/shared.py:127):
        === VECTOR MEMORY (...) ===
        [Result 1]
        <doc>
        ---
        [Result 2]
        ...
        === END VECTOR MEMORY ===
    """
    if not context_block or context_block.startswith("[ChromaDB query error"):
        return []

    chunks: list[str] = []
    current: list[str] = []
    for line in context_block.splitlines():
        line = line.rstrip()
        if line.startswith("=== VECTOR MEMORY") or line.startswith("=== END VECTOR MEMORY"):
            continue
        if re.match(r"\[Result\s+\d+\]", line) or line == "---":
            if current:
                chunks.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        chunks.append("\n".join(current).strip())

    seen: set[str] = set()
    sources: list[dict[str, Any]] = []
    for raw in chunks:
        body, meta = _clean_snippet(raw)
        if not body or len(body.strip()) < 8:
            continue
        norm = re.sub(r"\s+", " ", body.lower()).strip()
        if norm in seen:
            continue
        seen.add(norm)
        sources.append(_make_source_card(len(sources) + 1, body, meta))
    return sources


def _make_source_card(rank: int, body: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": rank,
        "snippet": body[:600],  # cap to keep the response small
        "source_type": meta.get("source_type", "footprint"),
        "timestamp": meta.get("timestamp"),
        "course_or_topic": meta.get("course_or_topic"),
    }


def _sanitize_answer(answer: str, has_sources: bool) -> str:
    """Scrub pipeline-leak artifacts from the top-level model answer.

    Always runs (regardless of `has_sources`) so injected retrieval labels,
    ontology brackets, BLOKLANDI literals, and Turkish prompt-echo phrases
    never reach the UI. The full regex set in `_ANSWER_SCRUB_REGEXES` is
    applied line-by-line; the legacy "no memory phrase" stripping and the
    BlindSpot box-line filter still run when sources are present.
    """
    if not answer:
        return ""

    cleaned = answer
    for rx in _ANSWER_SCRUB_REGEXES:
        cleaned = rx.sub(" ", cleaned)

    if has_sources:
        # When real memory cards exist, drop the contradictory
        # "no memory" sentence so the answer doesn't argue with itself.
        cleaned = re.sub(
            r"\s*" + re.escape(_NO_MEMORY_PHRASE) + r"\s*", " ", cleaned,
        )

    # Drop residual lines that are nothing but punctuation / box-drawing
    # / leftover separators after the regex sweep.
    kept_lines: list[str] = []
    for ln in cleaned.splitlines():
        stripped = ln.strip()
        if not stripped:
            kept_lines.append("")
            continue
        if re.match(r"^[\s┌│└─━╔╚╗╝║═=─\-]+$", stripped):
            continue
        # Collapse intra-line multi-space runs once per line.
        kept_lines.append(_RX_MULTI_WS.sub(" ", ln))
    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    if has_sources and len(re.sub(r"\s+", "", cleaned)) < 12:
        cleaned = (cleaned + "\n\n" + _FALLBACK_NOTE).strip()
    return cleaned


def _split_ontology(block: str) -> dict[str, Any]:
    if not block or block.startswith("[ONTOLOGY"):
        return {"available": False, "blocks": [], "concepts_used": [], "relations_used": []}
    blocks: list[str] = []
    concepts: set[str] = set()
    relations: set[str] = set()
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("==="):
            continue
        blocks.append(line)
        if line.startswith("[Exam]"):
            concepts.update({"Exam", "Course"})
            relations.add("Exam → Course")
        elif line.startswith("[Person]"):
            concepts.update({"Person", "Agent"})
            relations.add("Person → Agent")
        elif line.startswith("[Role]"):
            concepts.add("Role")
            relations.add("Agent → Role")
    return {
        "available": True,
        "blocks": blocks[:30],
        "concepts_used": sorted(concepts),
        "relations_used": sorted(relations),
    }


def _structure_gates(report: dict[str, Any]) -> dict[str, Any]:
    """Translate evaluator output into UI-friendly structure with confidence."""
    if not report or "gates" not in report:
        return {
            "conjunction": False,
            "active_gates": [],
            "gates": {},
            "overall_confidence": "low",
            "main_warning": "Gate validation unavailable.",
            "recommended_user_action": "Treat this answer as unverified.",
        }
    raw_gates = report.get("gates", {})
    active = list(report.get("active_gates", []))
    structured_gates: dict[str, Any] = {}
    failures: list[str] = []
    for code, entry in raw_gates.items():
        passed = bool(entry.get("pass"))
        evidence = entry.get("evidence", "")
        status = "PASS" if passed else "FAIL"
        structured_gates[code] = {
            "pass": passed,
            "label": GATE_LABELS.get(code, code),
            "evidence": evidence,
            "status": status,
        }
        if not passed:
            failures.append(code)
    # Add SKIP rows for gates listed in active_gates but not in the report,
    # so the UI table is stable.
    for code in active:
        if code not in structured_gates:
            structured_gates[code] = {
                "pass": False,
                "label": GATE_LABELS.get(code, code),
                "evidence": "Gate skipped for this role/path.",
                "status": "SKIP",
            }
    conjunction = bool(report.get("conjunction"))
    if conjunction:
        confidence = "high"
        main_warning = None
        action = None
    else:
        critical = {"C1", "C2", "C4"}
        if any(code in critical for code in failures):
            confidence = "low"
        else:
            confidence = "medium"
        first_fail = next((c for c in failures if c in GATE_FAIL_GUIDANCE), failures[0] if failures else None)
        if first_fail and first_fail in GATE_FAIL_GUIDANCE:
            main_warning, action = GATE_FAIL_GUIDANCE[first_fail]
        else:
            main_warning = "Gate validation flagged a concern."
            action = "Review the gate panel for details."
    return {
        "conjunction": conjunction,
        "active_gates": active,
        "gates": structured_gates,
        "overall_confidence": confidence,
        "main_warning": main_warning,
        "recommended_user_action": action,
    }


def _build_footprint_summary(status: dict[str, Any]) -> dict[str, Any]:
    """Enrich the basic file probe with heuristic course/exam signals."""
    if not status.get("available"):
        return {**status, "recent_course_signals": [], "weak_hints": [], "upcoming_mentions": []}
    path = _project_root() / status["file"]
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except Exception:
        lines = []
    courses: list[str] = []
    upcoming: list[str] = []
    weak: list[str] = []
    for line in lines:
        if _RX_UPCOMING.search(line):
            upcoming.append(line[:120])
        elif _RX_COURSE.search(line):
            courses.append(line[:120])
        if _RX_WEAK.search(line):
            weak.append(line[:120])
    return {
        **status,
        "recent_course_signals": courses[-5:][::-1],  # last 5, newest first
        "weak_hints": weak[:5],
        "upcoming_mentions": upcoming[-5:][::-1],
    }


def _detect_language(text: str) -> str:
    if not text:
        return "unknown"
    if re.search(r"[ğşıİĞŞı]", text):
        return "tr"
    tr_words = re.findall(
        r"\b(?:bunu|hafıza|akademik|sınav|ders|öğrenci|yapıya|göre)\b",
        text, re.I,
    )
    if tr_words:
        return "tr"
    if re.search(r"\b(?:the|and|with|exam|course|student)\b", text, re.I):
        return "en"
    return "unknown"


def _is_blocked_topic(token: str) -> bool:
    """True if `token` is a generic word, month, or pure date fragment."""
    if not token:
        return True
    t = token.strip()
    if len(t) < 2:
        return True
    if t.isdigit():
        return True
    low = t.lower()
    if low in _TOPIC_BLOCKLIST:
        return True
    if low in _TR_MONTHS:
        return True
    if _RX_DATE_FRAGMENT.match(t):
        return True
    # Filter "Şubat 09", "09 Mart", etc. (month + number combos).
    parts = [p.lower() for p in re.split(r"\s+", t) if p]
    if any(p in _TR_MONTHS for p in parts):
        return True
    return False


def _promote_academic_keywords(text: str) -> list[str]:
    """Return academic-keyword hits (case-insensitive) in their canonical form."""
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for kw in _ACADEMIC_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", text, re.I):
            key = kw.lower()
            if key not in seen:
                seen.add(key)
                found.append(kw)
    return found


def _build_study_output(
    answer: str,
    sources: list[dict[str, Any]],
    ontology: dict[str, Any],
    footprint: dict[str, Any],
    confidence: str,
) -> dict[str, Any]:
    """Rule-based study-output extractor (no extra LLM call).

    Pipeline:
      1. Promote real academic keywords found in the answer / ontology blocks.
      2. Add course codes from source cards (after blocklist filter).
      3. Drop any tokens in `_TOPIC_BLOCKLIST`, Turkish month names, or
         pure date fragments.
      4. Cap at 5; build student-actionable next steps from a phrase template.
    """
    blocks_text = " ".join(ontology.get("blocks", []) or [])
    haystack = "\n".join([answer or "", blocks_text])
    focus_topics: list[str] = []
    seen_low: set[str] = set()

    for kw in _promote_academic_keywords(haystack):
        if kw.lower() not in seen_low:
            seen_low.add(kw.lower())
            focus_topics.append(kw)

    for src in sources[:5]:
        topic = (src.get("course_or_topic") or "").strip()
        if not topic or _is_blocked_topic(topic):
            continue
        if topic.lower() in seen_low:
            continue
        seen_low.add(topic.lower())
        focus_topics.append(topic)

    focus_topics = [t for t in focus_topics if not _is_blocked_topic(t)][:5]

    # ── Next steps: clean templated phrases (never raw log lines) ──────────
    next_steps: list[str] = []
    if focus_topics:
        next_steps.append(f"Review {focus_topics[0]}")
    if len(focus_topics) > 1:
        next_steps.append(f"Solve practice problems on {focus_topics[1]}")
    elif focus_topics:
        next_steps.append(f"Solve practice problems on {focus_topics[0]}")
    if footprint.get("upcoming_mentions"):
        # Trim the raw log line to its course code or key phrase.
        raw = footprint["upcoming_mentions"][0]
        m_course = _RX_COURSE.search(raw)
        anchor = m_course.group(0) if m_course else "your upcoming items"
        next_steps.append(f"Prepare for {anchor}")
    if not next_steps:
        next_steps.append("Re-ask with a more specific course or topic.")

    checklist = [
        "Review class notes",
        "Solve a practice problem set",
        "Skim the relevant textbook chapter",
    ]

    risk_map = {
        "high":   "Answer well-grounded; standard study plan.",
        "medium": "Some grounding concerns; verify key facts before exam day.",
        "low":    "Low grounding; do not rely on the answer without cross-checking.",
    }
    risk_summary = risk_map.get(confidence, "Confidence unknown.")
    return {
        "focus_topics": focus_topics,
        "suggested_next_steps": next_steps[:3],
        "checklist": checklist,
        "risk_summary": risk_summary,
    }


def _build_why(
    answer: str,
    sources: list[dict[str, Any]],
    ontology: dict[str, Any],
    gates: dict[str, Any],
    degraded: bool,
    fallback_mode: Optional[str],
    next_step: Optional[str] = None,
) -> str:
    """One-sentence English explanation of how the answer was assembled."""
    n_src = len(sources)
    n_blocks = len(ontology.get("blocks", []) or []) if ontology.get("available") else 0
    gates_map = gates.get("gates", {}) or {}
    p = sum(1 for g in gates_map.values() if g.get("status") == "PASS")
    w = sum(1 for g in gates_map.values() if g.get("status") == "WARN")
    f = sum(1 for g in gates_map.values() if g.get("status") == "FAIL")
    mode = "degraded" if degraded else "normal"
    next_hint = next_step or gates.get("recommended_user_action") or "—"
    body = (
        f"Used {n_src} memory snippet(s) and {n_blocks} ontology block(s). "
        f"Gates: {p} PASS / {w} WARN / {f} FAIL. "
        f"Mode: {mode}. "
        f"Suggested next: {next_hint}."
    )
    if degraded:
        flagged = [k for k, v in {
            "Chroma":  "chroma" in (fallback_mode or "").lower(),
            "Ontology": "ontology" in (fallback_mode or "").lower(),
            "Gates":   "gates" in (fallback_mode or "").lower(),
            "Ollama":  "ollama" in (fallback_mode or "").lower(),
        }.items() if v]
        if flagged:
            body = f"Working in degraded mode ({', '.join(flagged)}). " + body
    return body


# ─────────────────────────────────────────────────────────────────────────────
#  Student Output — broader intent-aware composition
# ─────────────────────────────────────────────────────────────────────────────

# Recognised intents (the canonical seven).
_INTENTS = {
    "study_plan", "exam_prep", "assignment_help", "email_draft",
    "course_memory", "resource_recommendation", "academic_qa",
}

# Map UI mode tab → intent. Modes that don't dictate intent fall through to
# keyword-based detection on the user message.
_MODE_TO_INTENT: dict[str, str] = {
    "study_plan":           "study_plan",
    "exam_prep":            "exam_prep",
    "exam_focus":           "exam_prep",       # legacy alias
    "assignment_help":      "assignment_help",
    "email_draft":          "email_draft",
    "course_memory":        "course_memory",
    "footprint_explorer":   "course_memory",   # legacy alias
    "resources":            "resource_recommendation",
}

# Modes that explicitly defer to keyword classification on the message body.
# Listed for documentation: any mode not in `_MODE_TO_INTENT` falls through to
# the keyword path, but naming these makes the contract obvious — task buttons
# pin the intent, view buttons (and the bare "Ask a question" button) let the
# message decide.
_KEYWORD_FALLBACK_MODES: set[str] = {"ask", "dashboard", "validation"}

# Keyword cues for free-text classification (case-insensitive).
_INTENT_KEYWORDS: list[tuple[str, re.Pattern]] = [
    ("email_draft",
     re.compile(r"\b(email|e-?mail|e-?posta|mail\s+at|draft|write\s+a\s+mail|hocaya\s+mail)\b", re.I)),
    ("assignment_help",
     re.compile(r"\b(assignment|homework|ödev|teslim|submit\s+by|essay|report\s+due)\b", re.I)),
    ("exam_prep",
     re.compile(r"\b(exam|midterm|final|quiz|sınav|vize|bütünleme)\b", re.I)),
    ("study_plan",
     re.compile(r"\b(study\s+plan|schedule|weekly|daily|plan\s+my|haftalık|günlük|çalışma\s+plan)\b", re.I)),
    ("resource_recommendation",
     re.compile(r"\b(resource|link|book|video|reading|tutorial|kaynak|kitap)\b", re.I)),
    ("course_memory",
     re.compile(r"\b(footprint|course\s+memory|recent\s+course|history|geçmiş|son\s+ders)\b", re.I)),
]


def _classify_intent(message: str, mode: Optional[str]) -> str:
    """Pick a single intent.

    Task-mode buttons (in `_MODE_TO_INTENT`) dominate — the selected button
    pins the intent regardless of what the student typed. View/ask modes
    (`ask`, `dashboard`, `validation`) defer to keyword classification.
    """
    if mode and mode in _MODE_TO_INTENT:
        return _MODE_TO_INTENT[mode]
    # `mode in _KEYWORD_FALLBACK_MODES` or unset/unknown → keyword fallback.
    text = message or ""
    for intent, rx in _INTENT_KEYWORDS:
        if rx.search(text):
            return intent
    return "academic_qa"


def _empty_student_output(intent: str) -> dict[str, Any]:
    """Skeleton with all keys present — composers fill what they own."""
    return {
        "intent":           intent,
        "title":            "",
        "summary":          "",
        "draft":            "",
        "focus_topics":     [],
        "next_steps":       [],
        "checklist":        [],
        "schedule_blocks":  [],
        "resources":        [],
        "email":            {"subject": "", "body": "", "tone": ""},
        "risks":            [],
        "confidence":       "low",
    }


def _course_anchor(message: str, sources: list[dict[str, Any]],
                   focus_topics: list[str]) -> str:
    """Best guess at the course/topic the student is asking about."""
    m = _RX_COURSE.search(message or "")
    if m:
        return m.group(0)
    for src in sources:
        course = (src.get("course_or_topic") or "").strip()
        if course and not _is_blocked_topic(course):
            return course
    if focus_topics:
        return focus_topics[0]
    return "your course"


def _risks_from_gates(gate_report: dict[str, Any], degraded: bool,
                      sources_count: int) -> list[str]:
    """Translate gate failures + degraded subsystems into student-facing risks."""
    risks: list[str] = []
    if degraded:
        risks.append("Some subsystems are unavailable — output may be incomplete.")
    if sources_count == 0:
        risks.append("No memory snippets retrieved; output is not grounded.")
    for code, g in (gate_report.get("gates") or {}).items():
        if g.get("status") == "FAIL":
            warn, _ = GATE_FAIL_GUIDANCE.get(code, (g.get("evidence") or f"Gate {code} failed.", ""))
            risks.append(warn)
    main = gate_report.get("main_warning")
    if main and main not in risks:
        risks.append(main)
    return risks[:5]


# ── Brief-mode wording (used when the user's message is itself the brief) ──
#
# When the agent is operating on a self-contained brief, memory-grounding is
# not the criterion of correctness — the user's message is the source of
# truth. The "no memory" / "answer not grounded" warnings are misleading in
# that mode, so we replace the gate-derived risk list with a brief-aware one.

_BRIEF_MEMORY_RISK_FRAGMENTS = (
    "no memory snippets retrieved",
    "output is not grounded",
    "answer is not grounded in your footprint memory",
    "empty memory but missing blindspot disclosure",
    "low grounding",
    "do not rely on the answer without cross-checking",
    "blindspot disclosure",
)


def _is_memory_grounding_risk(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(frag in low for frag in _BRIEF_MEMORY_RISK_FRAGMENTS)


def _brief_risks(brief: dict[str, Any], message: str,
                 degraded: bool) -> list[str]:
    """Risk wording for self-contained briefs.

    Memory absence is normal in this mode; surface it as informational
    context, not a failure.
    """
    risks: list[str] = [
        "This output is based on your current brief. Memory was not needed.",
        "Verify missing details against your assignment sheet before submission.",
    ]
    for p in _detect_missing_question_pages(message or ""):
        risks.append(
            f"Page {p} questions are not included. "
            f"Please paste or upload page {p} to answer them accurately."
        )
    if degraded:
        risks.append(
            "Some subsystems are unavailable — Evidence drawer may be incomplete."
        )
    return risks[:5]


def _bump_confidence_for_brief(brief: dict[str, Any], current: str) -> str:
    """Self-contained briefs should not be rated LOW just because memory was
    skipped. Use MEDIUM by default; HIGH when the brief has clear required
    tasks or a goal section."""
    todos = brief.get("todos") or []
    has_goal = bool(brief.get("goal"))
    has_required = bool(brief.get("functions") or brief.get("deliverables"))
    if len(todos) >= 3 or (has_goal and has_required):
        target = "high"
    else:
        target = "medium"
    rank = {"low": 0, "medium": 1, "high": 2}
    return current if rank.get(current, 0) >= rank[target] else target


def _build_study_output_from_brief(
    brief: dict[str, Any], message: str, confidence: str,
) -> dict[str, Any]:
    """A study_output shape derived entirely from the brief.

    Same keys (`focus_topics`, `suggested_next_steps`, `checklist`,
    `risk_summary`) as the legacy `_build_study_output`, but populated
    from the parsed brief so it never echoes unrelated course codes,
    fallback "Prepare for OR701" lines, or memory-grounding warnings.
    """
    todos = brief.get("todos") or []
    functions = brief.get("functions") or []
    goal = brief.get("goal") or []
    deliverables = brief.get("deliverables") or []

    focus = list(todos) + list(functions) + list(goal)
    focus_topics = [t for t in focus if t][:5]

    next_steps: list[str] = []
    for t in (todos or functions or goal)[:3]:
        next_steps.append(t)
    if not next_steps:
        next_steps.append("Re-read your brief and confirm the required scope.")

    checklist: list[str] = list(deliverables)[:6] if deliverables else []
    if brief.get("screenshots_required") and not any(
        "screenshot" in (c or "").lower() or "evidence" in (c or "").lower()
        for c in checklist
    ):
        checklist.append("Screenshots / evidence collected with page references")
    if not checklist:
        checklist = [
            "All required tasks completed",
            "Required deliverables prepared",
            "Submitted before the deadline",
        ]

    risk_summary = (
        "This output is based on your current brief. Memory was not needed."
    )
    return {
        "focus_topics": focus_topics,
        "suggested_next_steps": next_steps[:3],
        "checklist": checklist[:6],
        "risk_summary": risk_summary,
    }


def _build_why_for_brief(brief: dict[str, Any], sources: list[dict[str, Any]],
                         ontology: dict[str, Any], gates: dict[str, Any],
                         degraded: bool, fallback_mode: Optional[str]) -> str:
    """Brief-mode reasoning sentence — never references memory grounding."""
    n_tasks = len(brief.get("todos") or [])
    n_deliv = len(brief.get("deliverables") or [])
    gates_map = gates.get("gates", {}) or {}
    p = sum(1 for g in gates_map.values() if g.get("status") == "PASS")
    w = sum(1 for g in gates_map.values() if g.get("status") == "WARN")
    f = sum(1 for g in gates_map.values() if g.get("status") == "FAIL")
    body = (
        "Built directly from your brief: "
        f"{n_tasks} task(s), {n_deliv} deliverable(s). "
        f"Gates: {p} PASS / {w} WARN / {f} FAIL. "
        "Verify missing details against your assignment sheet before submission."
    )
    if degraded:
        flagged = [k for k, v in {
            "Chroma":   "chroma" in (fallback_mode or "").lower(),
            "Ontology": "ontology" in (fallback_mode or "").lower(),
            "Gates":    "gates" in (fallback_mode or "").lower(),
            "Ollama":   "ollama" in (fallback_mode or "").lower(),
        }.items() if v]
        if flagged:
            body = f"Working in degraded mode ({', '.join(flagged)}). " + body
    return body


def _compose_email_draft(message: str, answer: str,
                         sources: list[dict[str, Any]],
                         course: str, confidence: str) -> dict[str, Any]:
    """Prepare an email draft (subject + body + tone). Never sends."""
    out = _empty_student_output("email_draft")
    out["title"] = f"Email draft for {course}".strip()
    out["summary"] = (
        "Drafted an email based on your request and recent course memory. "
        "Review before sending — nothing is sent automatically."
    )
    # Subject heuristic — pull a verb-y phrase, otherwise generic.
    subject_seed = (message or "").strip().splitlines()[0][:80] if message else ""
    if not subject_seed:
        subject_seed = f"Question about {course}"
    out["email"]["subject"] = subject_seed.rstrip(".? ").capitalize()
    # Body: re-use the model answer if present; else build a minimal greeting.
    body_lines = [
        "Dear Instructor,",
        "",
        (answer or "").strip()
            or f"I would like to ask a question about {course}.",
        "",
        "Best regards,",
        "[Your name]",
    ]
    out["email"]["body"] = "\n".join(body_lines).strip()
    out["email"]["tone"] = "polite, formal"
    out["draft"] = out["email"]["body"]
    out["confidence"] = confidence
    out["next_steps"] = [
        "Review the subject line.",
        "Personalize the greeting and signature.",
        "Send the email yourself when ready.",
    ]
    out["checklist"] = [
        "Recipient address is correct",
        "Subject is specific",
        "Body is polite and concise",
    ]
    out["risks"] = []  # filled by caller after gate inspection
    return out


# ── Self-contained brief detection, parsing, and memory-relevance gate ──────
#
# A "self-contained brief" is a user message that carries enough structure to
# answer from itself alone — assignment / lab / homework / project / report
# instructions, with at least some combination of section headers, bullet
# lists, imperative task verbs, deliverable language, and length. When such a
# brief is detected, the Student Agent treats the current message as the
# primary source of truth, skips the LLM round-trip (no raw VECTOR MEMORY /
# ONTOLOGY CONTEXT injection), and builds Model Answer + Student Output
# deterministically. Vector memory and live footprint stay in the Evidence
# drawer for audit but never inject new tasks/courses/files into the answer.
#
# All signals below are GENERAL — no example-specific keywords (no MQTT,
# Wireshark, course code, or page-21 hardcodes).

_RX_PAGE_REF = re.compile(
    r"(?:\bp(?:age|\.?)\s*[:\s]\s*(\d+)"     # page 21 / page:21 / page: 21 / p.21
    r"|\bsayfa\s*[:\s]?\s*(\d+)"             # sayfa 21 / sayfa:21
    r"|\b(\d+)\.\s*(?:page|sayfa)\b)",       # 21. page / 21. sayfa
    re.I,
)
_RX_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.\)])\s*")

# Section-header recognisers. Each maps a header token to a canonical section
# key. Headers must end with ":" on their own line to count.
_SECTION_HEADERS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("tasks",        re.compile(r"\b(?:list of\s+)?to-?dos?\b|\btasks?\b"
                                r"|\bsteps?\b|\bg[öo]revler\b", re.I)),
    ("deliverables", re.compile(r"\bdeliverables?\b|\bsubmission\b"
                                r"|\bteslim(?:ler)?\b|\bçıktılar\b", re.I)),
    ("goal",         re.compile(r"\bgoals?\b|\baim\b|\bobjectives?\b"
                                r"|\bhedef\b|\bama[çc]\b|\bdefinition\b"
                                r"|\bdescription\b", re.I)),
    ("inputs",       re.compile(r"\binputs?\b|\bgirdiler?\b", re.I)),
    ("outputs",      re.compile(r"\boutputs?\b|\bçıktı(?:lar)?\b", re.I)),
    ("constraints",  re.compile(r"\bconstraints?\b|\brequirements?\b"
                                r"|\brules?\b|\bk[ıi]s[ıi]tlamalar?\b"
                                r"|\bkurallar?\b", re.I)),
    ("functions",    re.compile(r"\b(?:required\s+)?functions?\b|\bAPI\b"
                                r"|\bfonksiyon(?:lar)?\b", re.I)),
    ("questions",    re.compile(r"\bquestions?\s*(?:&|and)?\s*discussion\b"
                                r"|\bquestions?\b|\bsorular?\b", re.I)),
    ("report",       re.compile(r"\breports?\b|\brapor\b", re.I)),
    ("grading",      re.compile(r"\bgrading\b|\brubric\b|\bnotland[ıi]rma\b",
                                re.I)),
)

# Imperative / task-action phrases that strongly indicate the message *is*
# the assignment instruction itself, not a question about one.
_RX_TASK_ACTION_PHRASE = re.compile(
    r"\b(?:write\s+a?\s*(?:program|function|class|script|report)"
    r"|implement(?:\s+(?:a|the))?"
    r"|build\s+(?:a|the)?"
    r"|create\s+(?:a|the)?"
    r"|design\s+(?:a|the)?"
    r"|develop\s+(?:a|the)?"
    r"|complete\s+the\s+(?:assignment|lab|homework|task)"
    r"|answer\s+(?:the\s+)?questions?"
    r"|submit\s+(?:your|the)"
    r"|deliver\s+(?:the|your)"
    r"|you\s+(?:must|should|are\s+(?:required|expected))"
    # Turkish equivalents
    r"|yaz[ıi]n?\b|uygulay[ıi]n?\b|geli[şs]tir(?:in)?\b|tasarlay[ıi]n?\b"
    r"|teslim\s+ed(?:in|iniz)?\b|cevaplay[ıi]n?\b|tamamlay[ıi]n?\b)",
    re.I,
)

# Domain-agnostic words that flag the message as assignment-shaped.
_RX_ASSIGNMENT_NOUN = re.compile(
    r"\b(?:assignment|homework|lab|project|exam|quiz|exercise|midterm|final"
    r"|essay|paper|report|coursework"
    r"|ödev|sınav|vize|proje|rapor|laboratuvar|deney)\b",
    re.I,
)

_RX_DELIVERABLE_NOUN = re.compile(
    r"\b(?:deliverable|deliverables|requirement|requirements|constraint"
    r"|constraints|rubric|grading"
    r"|teslim|kısıtlama|kural|gereksinim)\b",
    re.I,
)

# Generic evidence keywords (no Wireshark / Netualizer / log-tool names).
_RX_SCREENSHOT_HINT = re.compile(
    r"\b(?:picture|pictures|screenshot|screenshots|image|images|figure"
    r"|figures|log|logs|capture|captures"
    r"|ekran\s*g[öo]r[üu]nt[üu]s[üu]?|kan[ıi]t)\b",
    re.I,
)
_RX_TEXT_FILE_HINT = re.compile(
    r"\b(?:text\s+file|report|readme|markdown|pdf"
    r"|metin\s+dosyas[ıi]|rapor)\b",
    re.I,
)
_RX_QUESTIONS_PAGE = re.compile(
    r"\b(?:answer\s+(?:the\s+)?questions?|questions?\s*(?:&|and)?\s*discussion)\b"
    r"[\s\S]{0,40}?(?:p(?:age|\.?)\s*[:\s]?\s*(\d+)|sayfa\s*[:\s]?\s*(\d+))",
    re.I,
)
_RX_CASUAL_HELP = re.compile(
    r"\b(?:can\s+you\s+help\s+me|please\s+help|help\s+me\s+with"
    r"|bana\s+yard[ıi]m\s+eder\s+misin|yard[ıi]m\s+eder\s+misin"
    r"|yard[ıi]mc[ıi]\s+olur\s+musun)\b",
    re.I,
)
# Phrases left over from live footprint event records — must never resurface
# inside the answer or student output.
_RX_FOOTPRINT_EVENT = re.compile(
    r"Student\s+created\s+\w+(?:_\w+)*\s+output\s*:[^\n]*"
    r"|Original\s+request\s*:[^\n]*",
    re.I,
)


def _page_ref_value(match: "re.Match[str]") -> Optional[int]:
    """Pick the captured group from `_RX_PAGE_REF` and return as int."""
    for g in match.groups():
        if g:
            try:
                return int(g)
            except (ValueError, TypeError):
                return None
    return None


_GENERIC_TOKENS = frozenset({
    "page", "with", "the", "and", "for", "from", "this", "that", "your",
    "have", "been", "into", "must", "will", "show", "list", "todo", "todos",
    "tasks", "task", "deliverable", "deliverables", "picture", "pictures",
    "running", "setup", "file", "enough", "answer", "questions", "discussion",
    "both", "implement", "demo", "exercise", "bonus", "under",
})


def _is_self_contained_brief(message: str) -> bool:
    """Generic, weight-based detector for self-contained student briefs.

    Adds points across orthogonal signal families and triggers when the
    accumulated score is ≥3. No example-specific keywords. Designed to
    catch assignment / lab / homework / project / report briefs in any
    of: heading-and-bullet form, plain-bullet form, prose with imperative
    verbs, or mixed Turkish/English paste.
    """
    if not message or len(message) < 40:
        return False

    score = 0

    # Strong: at least one section-header line.
    if _has_section_headers(message):
        score += 2
    # Strong: 3+ bullet/numbered list items.
    bullet_count = sum(
        1 for ln in message.splitlines() if _RX_BULLET.match(ln)
    )
    if bullet_count >= 3:
        score += 2
    # Strong: imperative task-action phrase.
    if _RX_TASK_ACTION_PHRASE.search(message):
        score += 2
    # Medium: deliverables / requirements vocabulary.
    if _RX_DELIVERABLE_NOUN.search(message):
        score += 1
    # Medium: assignment-shaped noun.
    if _RX_ASSIGNMENT_NOUN.search(message):
        score += 1
    # Medium: page references.
    if _RX_PAGE_REF.search(message):
        score += 1
    # Medium: evidence/screenshot vocabulary.
    if _RX_SCREENSHOT_HINT.search(message):
        score += 1
    # Medium: substantial body of text (likely a paste, not a quick query).
    if len(message) >= 250:
        score += 1
    # Medium: multi-paragraph structure.
    if message.count("\n\n") >= 1 or bullet_count >= 2:
        score += 1

    return score >= 3


def _has_section_headers(message: str) -> bool:
    for raw in message.splitlines():
        ln = raw.strip()
        if not ln.endswith(":"):
            continue
        body = ln[:-1].strip()
        # Header must be short — long sentences ending in ":" are not
        # section markers.
        if len(body) > 60:
            continue
        for _key, rx in _SECTION_HEADERS:
            if rx.search(body):
                return True
    return False


def _section_for_header(header_line: str) -> Optional[str]:
    """Return the canonical section key for a `Foo:` line, or None."""
    body = header_line.strip()
    if not body.endswith(":"):
        return None
    body = body[:-1].strip()
    if len(body) > 60:
        return None
    for key, rx in _SECTION_HEADERS:
        if rx.search(body):
            return key
    return None


# Backward-compatible alias kept for existing callers / tests.
_has_explicit_assignment_brief = _is_self_contained_brief


def _parse_self_contained_task_brief(message: str) -> dict[str, Any]:
    """Generic parser for self-contained task briefs.

    Returns a dict with structured sections plus the legacy fields
    (`todos`, `deliverables`, `pages`, `has_brief`) so callers and tests
    that depend on the old shape continue to work.

    Handles: heading-and-bullet briefs, plain bullet lists, mixed
    English/Turkish, paragraphs with section labels, and prose with
    imperative task verbs.
    """
    empty = {
        "pages": [], "todos": [], "deliverables": [],
        "goal": [], "inputs": [], "outputs": [], "constraints": [],
        "functions": [], "questions": [], "report": [], "grading": [],
        "screenshots_required": False,
        "tags": [],
        "has_brief": False,
    }
    if not message:
        return empty

    # ── pages ──
    pages: list[int] = []
    seen_p: set[int] = set()
    for m in _RX_PAGE_REF.finditer(message):
        p = _page_ref_value(m)
        if p is None:
            continue
        if p not in seen_p:
            seen_p.add(p)
            pages.append(p)

    # ── section walk ──
    sections: dict[str, list[str]] = {
        key: [] for key, _ in _SECTION_HEADERS
    }
    section: Optional[str] = None
    free_lines: list[str] = []

    for raw in message.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        sec = _section_for_header(ln)
        if sec is not None:
            section = sec
            continue
        is_bullet = bool(_RX_BULLET.match(ln))
        clean = _RX_BULLET.sub("", ln).strip(" -*•").rstrip(".")
        if not clean:
            continue
        # Drop casual-help boilerplate everywhere; it is signal noise.
        if _RX_CASUAL_HELP.search(clean):
            continue
        if section is not None:
            sections[section].append(clean)
        elif is_bullet:
            free_lines.append(clean)
        elif _RX_PAGE_REF.search(clean) and len(clean) < 200:
            free_lines.append(clean)

    # ── promote free bullet/page lines to tasks when no explicit section ──
    if free_lines and not sections["tasks"]:
        sections["tasks"].extend(free_lines)

    # ── derive simple flags ──
    screenshots_required = bool(
        _RX_SCREENSHOT_HINT.search(message)
        or any(_RX_SCREENSHOT_HINT.search(d) for d in sections["deliverables"])
    )

    # ── tags: short uppercase / capitalized acronyms appearing in the body ──
    # Captures things like "MQTT", "REST", "TCP" — purely generic, not
    # an example-specific allow-list.
    tags: list[str] = []
    seen_t: set[str] = set()
    for tok in re.findall(r"\b[A-Z]{2,6}\b", message):
        if tok.isupper() and tok not in seen_t and tok not in _ASSIGNMENT_STOPWORDS:
            seen_t.add(tok)
            tags.append(tok)
    tags = tags[:6]

    has_brief = bool(
        pages or any(sections.values()) or _is_self_contained_brief(message)
    )

    return {
        "pages": pages,
        "todos": sections["tasks"],
        "deliverables": sections["deliverables"],
        "goal": sections["goal"],
        "inputs": sections["inputs"],
        "outputs": sections["outputs"],
        "constraints": sections["constraints"],
        "functions": sections["functions"],
        "questions": sections["questions"],
        "report": sections["report"],
        "grading": sections["grading"],
        "screenshots_required": screenshots_required,
        "tags": tags,
        "has_brief": has_brief,
    }


# Stop-list of frequent uppercase tokens that are not subject tags.
_ASSIGNMENT_STOPWORDS = frozenset({
    "TODO", "TODOS", "TASK", "TASKS", "GOAL", "AIM", "INPUT", "INPUTS",
    "OUTPUT", "OUTPUTS", "API", "AND", "OR", "FOR", "IF", "ELSE",
    "MUST", "WILL", "PDF", "URL",
})


# Backward-compatible alias.
_parse_assignment_brief = _parse_self_contained_task_brief


def _memory_is_relevant_to_prompt(memory_item: dict[str, Any], message: str) -> bool:
    """Cheap lexical-overlap test. True only if at least one informative
    token from `message` (≥4 chars, not generic) appears in the memory body."""
    if not memory_item or not message:
        return False
    text = (
        memory_item.get("snippet")
        or memory_item.get("text")
        or memory_item.get("content")
        or ""
    ).lower()
    if not text:
        return False
    msg_l = message.lower()
    tokens = {
        t for t in re.findall(r"[a-z0-9_+/]{4,}", msg_l)
        if t not in _GENERIC_TOKENS
    }
    if not tokens:
        return False
    return any(t in text for t in tokens)


def _filter_irrelevant_memory_for_student_output(
    sources: list[dict[str, Any]], message: str,
) -> list[dict[str, Any]]:
    """Return only the source cards that share informative tokens with the
    user's message. Used to decide whether memory should influence Student
    Output (Evidence drawer keeps the full unfiltered list)."""
    if not sources:
        return []
    return [s for s in sources if _memory_is_relevant_to_prompt(s, message)]


def _detect_missing_question_pages(message: str) -> list[int]:
    """Pages whose questions the brief asks the student to answer but whose
    actual question text isn't pasted into the message (no '?' present)."""
    if not message:
        return []
    pages: list[int] = []
    seen: set[int] = set()
    if "?" in message:
        return []
    for m in _RX_QUESTIONS_PAGE.finditer(message):
        p_str = next((g for g in m.groups() if g), None)
        if p_str is None:
            continue
        try:
            p = int(p_str)
        except (ValueError, TypeError):
            continue
        if p not in seen:
            seen.add(p)
            pages.append(p)
    return pages


def _compose_assignment_help_from_brief(
    message: str, brief: dict[str, Any],
    sources: list[dict[str, Any]], confidence: str,
) -> dict[str, Any]:
    """Build assignment_help output entirely from the pasted brief.

    Vector memory is intentionally ignored for body text and only consulted
    to decide whether to attach the 'memory did not add reliable details'
    risk message. Evidence drawer (response['sources']) is unaffected.

    Source-priority policy: brief content (pages, tasks, goal, inputs,
    outputs, constraints, functions, questions, deliverables) is the only
    body input. No tech-tag allow-lists, no example-specific keywords.
    """
    out = _empty_student_output("assignment_help")

    pages        = brief.get("pages") or []
    todos        = brief.get("todos") or []
    deliverables = brief.get("deliverables") or []
    goal         = brief.get("goal") or []
    inputs       = brief.get("inputs") or []
    outputs      = brief.get("outputs") or []
    constraints  = brief.get("constraints") or []
    functions    = brief.get("functions") or []
    questions    = brief.get("questions") or []
    tags         = brief.get("tags") or []

    out["title"] = "Assignment guidance"
    if tags:
        out["title"] += f" ({', '.join(tags[:3])})"

    summary_parts: list[str] = ["Assignment summary based on your brief."]
    if goal:
        summary_parts.append("Goal: " + " ".join(goal)[:200])
    if pages:
        summary_parts.append(
            "Pages referenced: " + ", ".join(f"page {p}" for p in pages) + "."
        )
    if todos:
        summary_parts.append(f"{len(todos)} required task(s) parsed.")
    if functions:
        summary_parts.append(f"{len(functions)} required function(s) parsed.")
    if deliverables:
        summary_parts.append(f"{len(deliverables)} deliverable(s) parsed.")
    if questions:
        summary_parts.append(f"{len(questions)} question(s) parsed.")
    out["summary"] = " ".join(summary_parts)

    focus = list(todos) + list(functions) + list(goal)
    out["focus_topics"] = focus[:8]

    steps_src = todos or functions or goal or deliverables
    next_steps = [f"Step {i}: {t}" for i, t in enumerate(steps_src, 1)]
    if not next_steps:
        next_steps = [
            "Re-read the assignment brief carefully.",
            "Outline before writing.",
            "Verify each deliverable before submitting.",
        ]
    out["next_steps"] = next_steps[:8]

    outline_src = steps_src[:8] if steps_src else [
        "Setup / environment",
        "Implementation",
        "Validation and report",
    ]
    out["schedule_blocks"] = [
        f"Section {i}: {t}" for i, t in enumerate(outline_src, 1)
    ]

    checklist: list[str] = list(deliverables) if deliverables else []
    for c in constraints:
        checklist.append(f"Constraint: {c}")
    if any(_RX_SCREENSHOT_HINT.search(d or "") for d in deliverables) or \
            brief.get("screenshots_required"):
        checklist.append(
            "Screenshots / evidence collected and labeled with the page reference"
        )
    if any(_RX_TEXT_FILE_HINT.search(d or "") for d in deliverables):
        checklist.append("Text/report file prepared with task summary")
    if not checklist:
        checklist = [
            "All required tasks completed",
            "Required deliverables prepared",
            "Submitted before the deadline",
        ]
    out["checklist"] = checklist[:10]

    risks: list[str] = []
    for p in _detect_missing_question_pages(message):
        risks.append(
            f"Page {p} questions are not included. "
            f"Please paste or upload page {p} to answer them accurately."
        )
    relevant = _filter_irrelevant_memory_for_student_output(sources, message)
    if not relevant:
        risks.append(
            "I used your current assignment brief as the primary source. "
            "Memory did not add reliable extra details."
        )
    out["risks"] = risks[:5]
    out["confidence"] = confidence
    return out


def _compose_answer_from_brief(message: str, brief: dict[str, Any]) -> str:
    """Build a deterministic, English Model Answer from the parsed brief.

    Pure string assembly — no LLM, no vector memory, no ontology. Source
    priority: only fields parsed from the user's message are used.

    Sections (rendered when the corresponding brief field is non-empty):
      • Assignment summary   — pages + generic tags
      • Goal
      • Inputs / Outputs
      • Required tasks
      • Required functions
      • Constraints
      • Deliverables
      • Questions
      • Execution plan        — fixed 4-step skeleton
      • Screenshot / evidence checklist
      • Missing pages note
    """
    pages        = brief.get("pages") or []
    todos        = (brief.get("todos") or [])[:8]
    deliverables = (brief.get("deliverables") or [])[:8]
    goal         = (brief.get("goal") or [])[:6]
    inputs       = (brief.get("inputs") or [])[:6]
    outputs      = (brief.get("outputs") or [])[:6]
    constraints  = (brief.get("constraints") or [])[:8]
    functions    = (brief.get("functions") or [])[:8]
    questions    = (brief.get("questions") or [])[:8]
    tags         = brief.get("tags") or []

    lines: list[str] = []
    title = "Assignment summary"
    if tags:
        title += f" ({', '.join(tags[:3])})"
    lines.append(title)
    if pages:
        lines.append(
            "Pages referenced: " + ", ".join(f"page {p}" for p in pages) + "."
        )
    if todos:
        lines.append(f"{len(todos)} required task(s) parsed from your brief.")
    if deliverables:
        lines.append(f"{len(deliverables)} deliverable(s) parsed.")

    def _block(header: str, items: list[str], bullet: str = "  - "):
        if not items:
            return
        lines.append("")
        lines.append(header + ":")
        for it in items:
            lines.append(f"{bullet}{it}")

    _block("Goal", goal)
    _block("Inputs", inputs)
    _block("Outputs", outputs)
    if todos:
        lines.append("")
        lines.append("Required tasks:")
        for i, t in enumerate(todos, 1):
            lines.append(f"  {i}. {t}")
    _block("Required functions", functions)
    _block("Constraints", constraints)
    _block("Deliverables", deliverables)
    _block("Questions", questions)

    lines.append("")
    lines.append("Execution plan:")
    lines.append("  1. Re-read the brief and confirm the required scope.")
    lines.append("  2. Set up the environment / inputs needed.")
    lines.append("  3. Implement each required task in order.")
    lines.append("  4. Capture evidence and review against deliverables before submission.")

    if any(_RX_SCREENSHOT_HINT.search(d or "") for d in deliverables) or \
            brief.get("screenshots_required") or \
            _RX_SCREENSHOT_HINT.search(message or ""):
        lines.append("")
        lines.append("Screenshot / evidence checklist:")
        lines.append(
            "  - Capture each running component clearly with a page reference."
        )
        lines.append(
            "  - Save logs / outputs in a labeled file alongside the screenshot."
        )

    missing_pages = _detect_missing_question_pages(message or "")
    if missing_pages:
        lines.append("")
        lines.append("Missing pages note:")
        for p in missing_pages:
            lines.append(
                f"  - Page {p} questions are not included in your brief. "
                f"Please paste page {p} so the questions can be answered."
            )

    return "\n".join(lines).strip()


def _compose_assignment_help(message: str, answer: str,
                             sources: list[dict[str, Any]],
                             course: str, focus_topics: list[str],
                             confidence: str) -> dict[str, Any]:
    # Explicit brief → trust the prompt; ignore vector memory in body text.
    if _has_explicit_assignment_brief(message):
        brief = _parse_assignment_brief(message)
        if brief.get("has_brief"):
            return _compose_assignment_help_from_brief(
                message, brief, sources, confidence,
            )

    out = _empty_student_output("assignment_help")
    out["title"] = f"Assignment guidance for {course}"
    out["summary"] = (
        (answer or "").strip()[:400]
        or "Outline and steps to approach this assignment, grounded in your course memory."
    )
    out["focus_topics"] = focus_topics[:5]
    out["next_steps"] = [
        f"Re-read the assignment brief for {course}",
        "Draft an outline before writing.",
        "Cross-check your draft against your class notes.",
    ]
    outline = []
    for i, topic in enumerate(focus_topics[:3], 1):
        outline.append(f"Section {i}: {topic}")
    if not outline:
        outline = [
            "Section 1: Introduction and goal",
            "Section 2: Method / approach",
            "Section 3: Results and reflection",
        ]
    out["schedule_blocks"] = outline  # reuse field for outline (UI maps to "outline")
    out["checklist"] = [
        "Title page / header included",
        "All required sections covered",
        "References cited where needed",
        "Spelling and grammar reviewed",
        "Submitted before the deadline",
    ]
    out["confidence"] = confidence
    return out


def _compose_study_plan(message: str, course: str,
                        focus_topics: list[str],
                        upcoming: list[str], confidence: str) -> dict[str, Any]:
    out = _empty_student_output("study_plan")
    priority = focus_topics[0] if focus_topics else course
    out["title"] = f"Study plan: {priority}"
    out["summary"] = (
        f"Weekly plan focused on {priority}. "
        f"Adjust block lengths to your own pace."
    )
    out["focus_topics"] = focus_topics[:5]
    blocks = []
    for i, topic in enumerate(focus_topics[:5], 1):
        blocks.append(f"Day {i}: 60 min — {topic}")
    if not blocks:
        blocks = [
            "Day 1: 60 min — Survey course materials",
            "Day 2: 60 min — Identify weak topics",
            "Day 3: 60 min — Practice problems",
        ]
    out["schedule_blocks"] = blocks
    out["next_steps"] = [
        f"Start with {priority} on Day 1.",
        "Track time spent per topic.",
        "Re-balance the plan after the first week.",
    ]
    out["checklist"] = [
        "Class notes reviewed",
        "Practice set attempted",
        "Weak topics revisited",
    ]
    if upcoming:
        out["risks"] = [f"Upcoming item detected: {upcoming[0][:80]}"]
    out["confidence"] = confidence
    return out


def _compose_exam_prep(message: str, course: str,
                       focus_topics: list[str],
                       upcoming: list[str], confidence: str) -> dict[str, Any]:
    out = _empty_student_output("exam_prep")
    priority = focus_topics[0] if focus_topics else course
    out["title"] = f"Exam preparation: {priority}"
    out["summary"] = (
        f"Targeted exam-prep plan for {priority}, "
        "weighted toward your weakest grounded topics."
    )
    out["focus_topics"] = focus_topics[:5]
    blocks = []
    for i, topic in enumerate(focus_topics[:4], 1):
        blocks.append(f"Block {i}: 45 min — Drill on {topic}")
    blocks.append("Block 5: 30 min — Past-paper attempt under timed conditions")
    out["schedule_blocks"] = blocks
    out["next_steps"] = [
        f"Drill {priority} first.",
        "Solve at least one timed past paper.",
        "Mark mistakes and revisit the next day.",
    ]
    out["checklist"] = [
        "Formula sheet reviewed",
        "Past papers attempted",
        "Mistakes journal updated",
        "Sleep before the exam",
    ]
    if upcoming:
        out["risks"] = [f"Upcoming item: {upcoming[0][:80]}"]
    out["confidence"] = confidence
    return out


def _compose_course_memory(message: str, sources: list[dict[str, Any]],
                           footprint: dict[str, Any], focus_topics: list[str],
                           confidence: str) -> dict[str, Any]:
    out = _empty_student_output("course_memory")
    record_count = footprint.get("record_count", 0)
    out["title"] = "Course memory analysis"
    out["summary"] = (
        f"Analyzed {len(sources)} retrieved snippet(s) from "
        f"{record_count} footprint record(s). "
        "Topics below are inferred from your own course history."
    )
    out["focus_topics"] = focus_topics[:5]
    out["next_steps"] = [
        "Review the most recent course signals.",
        "Identify topics with thin coverage.",
        "Plan study time for any weak topic.",
    ]
    out["checklist"] = [
        "Recent footprint reviewed",
        "Weak topics noted",
        "Upcoming deadlines acknowledged",
    ]
    out["confidence"] = confidence
    return out


def _compose_resources(message: str, focus_topics: list[str],
                       course: str, confidence: str) -> dict[str, Any]:
    out = _empty_student_output("resource_recommendation")
    priority = focus_topics[0] if focus_topics else course
    out["title"] = f"Resource suggestions: {priority}"
    out["summary"] = (
        f"Generic categories of resources for {priority}. "
        "Verify each resource fits your syllabus before relying on it."
    )
    resources: list[dict[str, str]] = []
    for topic in (focus_topics[:3] or [course]):
        resources.append({
            "type": "textbook chapter",
            "label": f"Course textbook: chapter on {topic}",
            "note": "Match against your syllabus.",
        })
        resources.append({
            "type": "practice set",
            "label": f"Practice problem set on {topic}",
            "note": "Use past-year sets if available.",
        })
        resources.append({
            "type": "video",
            "label": f"Lecture recording or open-courseware on {topic}",
            "note": "Prefer your instructor's recording first.",
        })
    out["resources"] = resources[:9]
    out["focus_topics"] = focus_topics[:5]
    out["next_steps"] = [
        f"Pick one textbook chapter for {priority}.",
        "Attempt one practice set.",
        "Add useful resources to your notes.",
    ]
    out["confidence"] = confidence
    return out


def _compose_academic_qa(message: str, answer: str,
                         focus_topics: list[str],
                         confidence: str) -> dict[str, Any]:
    out = _empty_student_output("academic_qa")
    out["title"] = "Answer summary"
    out["summary"] = (answer or "").strip()[:400] or (
        "No grounded answer was produced."
    )
    out["focus_topics"] = focus_topics[:5]
    if focus_topics:
        out["next_steps"] = [
            f"Review {focus_topics[0]}",
            "Cross-check the answer with your class notes.",
        ]
    else:
        out["next_steps"] = ["Re-ask with a more specific course or topic."]
    out["checklist"] = [
        "Class notes reviewed",
        "Source memory verified",
    ]
    out["confidence"] = confidence
    return out


# ── Output-string sanitization ───────────────────────────────────────────────
# Final-mile sweep across every student-facing string to strip pipeline-leak
# artifacts (raw retrieval markers, kv pairs, BlindSpot box characters). Runs
# right before _build_student_output returns; the rules here mirror the cleanup
# already performed on individual snippets in `_clean_snippet`, but applied to
# whatever text composers may have spliced in from the model draft.
_RX_BOX_CHARS = re.compile(r"[┌│└─━╔╚╗╝║═]+")
_RX_BRACKETED_TAG = re.compile(r"\[[A-Za-z_][A-Za-z0-9_ ]*\]")
_RX_RESULT_MARK = re.compile(r"\[Result\s+\d+\]", re.I)
_RX_KV_FRAGMENT = re.compile(r"\b(?:src|course|text|tag|user)\s*=\s*\S+", re.I)
_RX_EQ_SEPARATOR = re.compile(r"={3,}[^=\n]*={3,}")
_RX_EQ_RUN = re.compile(r"={3,}")
_RX_VECTOR_MEM_LITERAL = re.compile(r"\bVECTOR\s*MEMORY\b", re.I)
_RX_ONTOLOGY_LITERAL = re.compile(r"\bONTOLOGY\s*CONTEXT\b", re.I)
_RX_BLOKLANDI = re.compile(r"\bBLOKLANDI\b", re.I)
_RX_TR_VECTOR_MEM_PHRASE = re.compile(
    r"Vector\s*Memory'?(?:den|ye|ya|de|da)?"
    r"(?:\s*(?:aldı[kğ]ım[ıi]z[ıi]?|gelen|getirdi[kğ]i|ç[ıi]kard[ıi][kğ]ım[ıi]z))?",
    re.I,
)
_RX_TR_ONTOLOGY_PHRASE = re.compile(
    r"Ontoloji\s*ba[ğg]lam[ıi]ndan?"
    r"(?:\s*(?:aldı[kğ]ım[ıi]z[ıi]?|gelen|getirdi[kğ]i|ç[ıi]kard[ıi][kğ]ım[ıi]z))?",
    re.I,
)
_RX_PATH_LEAK = re.compile(r"\b[a-z_][a-z0-9_]*/[a-z_][a-z0-9_]*\.py\b", re.I)
_RX_COURSE_CODE = re.compile(r"\b[A-Z]{2,4}\d{3,5}\b")
_RX_AGENT_ALIAS = re.compile(
    r"\b\w+(?:Student|Instructor|Researcher|Head)Agent\b", re.I,
)
_RX_EXAM_ALIAS = re.compile(r"\b(?:midterm|final|quiz)\d+\b", re.I)
_RX_MULTI_WS = re.compile(r"\s{2,}")

# Regex set applied to top-level response['answer'] to scrub pipeline leaks.
# Order matters: literals/phrases first, then bracketed/equation tags, then
# fragment-level cleanup. Each substitutes with a single space; a final
# whitespace collapse normalises spacing.
_ANSWER_SCRUB_REGEXES: tuple[re.Pattern, ...] = (
    _RX_FOOTPRINT_EVENT,
    _RX_TR_VECTOR_MEM_PHRASE,
    _RX_TR_ONTOLOGY_PHRASE,
    _RX_VECTOR_MEM_LITERAL,
    _RX_ONTOLOGY_LITERAL,
    _RX_EQ_SEPARATOR,
    _RX_EQ_RUN,
    _RX_RESULT_MARK,
    _RX_BRACKETED_TAG,
    _RX_KV_FRAGMENT,
    _RX_BLOKLANDI,
    _RX_BOX_CHARS,
    _RX_PATH_LEAK,
    _RX_COURSE_CODE,
    _RX_AGENT_ALIAS,
    _RX_EXAM_ALIAS,
)


def _sanitize_str(s: str) -> str:
    if not s:
        return s or ""
    out = s
    for rx in (_RX_BOX_CHARS, _RX_EQ_SEPARATOR, _RX_VECTOR_MEM_LITERAL,
               _RX_BRACKETED_TAG, _RX_RESULT_MARK, _RX_KV_FRAGMENT,
               _RX_BLOKLANDI, _RX_FOOTPRINT_EVENT,
               _RX_TR_VECTOR_MEM_PHRASE, _RX_TR_ONTOLOGY_PHRASE):
        out = rx.sub(" ", out)
    out = _RX_MULTI_WS.sub(" ", out)
    return out.strip()


def _sanitize_student_output(out: dict[str, Any]) -> dict[str, Any]:
    """Strip pipeline-leak artifacts from every string field in `out`."""
    if not isinstance(out, dict):
        return out
    for key in ("title", "summary", "draft"):
        v = out.get(key)
        if isinstance(v, str):
            out[key] = _sanitize_str(v)
    for key in ("focus_topics", "next_steps", "checklist",
                "schedule_blocks", "risks"):
        v = out.get(key)
        if isinstance(v, list):
            out[key] = [_sanitize_str(x) if isinstance(x, str) else x for x in v]
    email = out.get("email")
    if isinstance(email, dict):
        for k in ("subject", "body", "tone"):
            if isinstance(email.get(k), str):
                email[k] = _sanitize_str(email[k])
    resources = out.get("resources")
    if isinstance(resources, list):
        for r in resources:
            if isinstance(r, dict):
                for k in ("label", "note", "type"):
                    if isinstance(r.get(k), str):
                        r[k] = _sanitize_str(r[k])
    return out


def _is_live_footprint_event_source(src: dict[str, Any]) -> bool:
    """True if a source card looks like a chat-turn live-footprint record.

    Such records are useful for audit (Evidence drawer) but must never
    inject prior intent/title text into a fresh Student Output."""
    if not isinstance(src, dict):
        return False
    snippet = (src.get("snippet") or src.get("text") or "") if isinstance(src, dict) else ""
    if not isinstance(snippet, str):
        return False
    return bool(_RX_FOOTPRINT_EVENT.search(snippet))


def _build_student_output(intent: str, message: str, answer: str,
                          sources: list[dict[str, Any]],
                          ontology: dict[str, Any],
                          footprint: dict[str, Any],
                          gate_report: dict[str, Any],
                          degraded: bool,
                          brief: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Compose the canonical `student_output` block for the given intent.

    When ``brief`` is supplied, the message is a self-contained task brief
    and serves as the primary source of truth: memory-grounding risks are
    suppressed and replaced with brief-aware wording.
    """
    confidence = gate_report.get("overall_confidence", "low")
    if degraded and confidence == "high":
        confidence = "medium"
    # Strip live-footprint event records from the sources used by the
    # composer. Evidence drawer keeps the full list; Student Output must
    # not inherit "Student created … output: <prior title>" phrasing.
    sources = [s for s in sources if not _is_live_footprint_event_source(s)]
    # Reuse the focus_topics logic from study_output for consistency.
    blocks_text = " ".join(ontology.get("blocks", []) or [])
    haystack = "\n".join([answer or "", blocks_text])
    focus_topics: list[str] = []
    seen: set[str] = set()
    for kw in _promote_academic_keywords(haystack):
        if kw.lower() not in seen:
            seen.add(kw.lower())
            focus_topics.append(kw)
    for src in sources[:5]:
        topic = (src.get("course_or_topic") or "").strip()
        if topic and not _is_blocked_topic(topic) and topic.lower() not in seen:
            seen.add(topic.lower())
            focus_topics.append(topic)
    focus_topics = [t for t in focus_topics if not _is_blocked_topic(t)][:5]

    course = _course_anchor(message, sources, focus_topics)
    upcoming = footprint.get("upcoming_mentions") or []

    if intent == "email_draft":
        out = _compose_email_draft(message, answer, sources, course, confidence)
    elif intent == "assignment_help":
        out = _compose_assignment_help(message, answer, sources, course,
                                       focus_topics, confidence)
    elif intent == "study_plan":
        out = _compose_study_plan(message, course, focus_topics, upcoming, confidence)
    elif intent == "exam_prep":
        out = _compose_exam_prep(message, course, focus_topics, upcoming, confidence)
    elif intent == "course_memory":
        out = _compose_course_memory(message, sources, footprint, focus_topics, confidence)
    elif intent == "resource_recommendation":
        out = _compose_resources(message, focus_topics, course, confidence)
    else:
        out = _compose_academic_qa(message, answer, focus_topics, confidence)

    # Risks: brief mode replaces memory-grounding warnings with brief-aware
    # wording; non-brief mode merges intent-specific risks with gate-derived
    # ones as before.
    if brief is not None:
        existing = [
            r for r in (out.get("risks") or [])
            if isinstance(r, str) and not _is_memory_grounding_risk(r)
        ]
        brief_risks = _brief_risks(brief, message, degraded)
        out["risks"] = list(dict.fromkeys(existing + brief_risks))[:5]
    else:
        gate_risks = _risks_from_gates(gate_report, degraded, len(sources))
        out["risks"] = list(dict.fromkeys((out.get("risks") or []) + gate_risks))[:5]
    return _sanitize_student_output(out)


# ─────────────────────────────────────────────────────────────────────────────
#  Workspace history (lightweight JSON persistence)
# ─────────────────────────────────────────────────────────────────────────────
#
# Single-user assumption matches the rest of the app — there is no auth and no
# per-user partitioning. A simple JSON file at ``data/student_agent_history.json``
# stores the most recent 20 chat responses (newest first). The store is
# deliberately defensive: missing file, malformed JSON, or write failures
# never raise out of the route handler — at worst the UI shows an empty list.

_HISTORY_LIMIT = 20


def _history_path() -> Path:
    return _project_root() / "data" / "student_agent_history.json"


def _load_history() -> list[dict[str, Any]]:
    path = _history_path()
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else []
    except (OSError, ValueError) as exc:
        logger.info("history file unreadable (%s); treating as empty", exc)
        return []
    if not isinstance(data, list):
        return []
    # Drop entries that aren't dicts so downstream code can rely on shape.
    return [item for item in data if isinstance(item, dict)]


def _save_history(items: list[dict[str, Any]]) -> None:
    path = _history_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("could not write history file: %s", exc)


def _append_history(item: dict[str, Any]) -> None:
    items = _load_history()
    items.insert(0, item)
    if len(items) > _HISTORY_LIMIT:
        items = items[:_HISTORY_LIMIT]
    _save_history(items)


def _build_history_item(message: str, mode: Optional[str],
                        response: dict[str, Any]) -> dict[str, Any]:
    """Wrap a chat response with workspace-history metadata.

    The response payload is preserved verbatim (so click-to-restore can repaint
    every evidence panel) and supplemented with: a stable `id`, `created_at`
    aliasing `timestamp`, and `intent` / `title` / `status` for the Activity
    Story and Agenda views.
    """
    student_out = response.get("student_output") or {}
    title = (student_out.get("title") or "").strip()
    if not title:
        title = (message or "").strip()[:80]
    intent = student_out.get("intent") or "academic_qa"
    timestamp = datetime.now(timezone.utc).isoformat()
    item: dict[str, Any] = {
        "id":         uuid.uuid4().hex[:12],
        "created_at": timestamp,
        "timestamp":  timestamp,  # back-compat with earlier tests/UI
        "mode":       mode,
        "intent":     intent,
        "title":      title,
        "status":     "saved",
        "message":    message,
    }
    item.update(response)
    return item


# ─────────────────────────────────────────────────────────────────────────────
#  Agenda — derive the Dashboard surface from history
# ─────────────────────────────────────────────────────────────────────────────

_AGENDA_INTENT_BUCKET: dict[str, str] = {
    "study_plan":              "study_blocks",
    "exam_prep":               "study_blocks",
    "assignment_help":         "assignment_tasks",
    "email_draft":             "saved_drafts",
    "resource_recommendation": "resources",
}


def _agenda_card(item: dict[str, Any]) -> Optional[dict[str, Any]]:
    if not item.get("id"):
        return None
    so = item.get("student_output") or {}
    gate = item.get("gate_report") or {}
    focus = so.get("focus_topics") or []
    subtitle = focus[0] if focus else None
    return {
        "id":         item.get("id"),
        "intent":     item.get("intent") or so.get("intent") or "academic_qa",
        "title":      item.get("title") or "",
        "subtitle":   subtitle,
        "status":     item.get("status", "saved"),
        "created_at": item.get("created_at") or item.get("timestamp"),
        "confidence": gate.get("overall_confidence", "low"),
        "extras":     {},
    }


def _build_agenda(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    today_iso = date.today().isoformat()
    out: dict[str, list[dict[str, Any]]] = {
        "today": [], "upcoming": [], "study_blocks": [],
        "assignment_tasks": [], "saved_drafts": [],
        "resources": [], "completed": [],
    }
    for item in items:
        card = _agenda_card(item)
        if not card:
            continue
        created = (item.get("created_at") or item.get("timestamp") or "")[:10]
        status = item.get("status") or "saved"
        intent = item.get("intent") or "academic_qa"
        if created == today_iso:
            out["today"].append(card)
        if status == "planned":
            out["upcoming"].append(card)
        if status == "done":
            out["completed"].append(card)
        bucket_name = _AGENDA_INTENT_BUCKET.get(intent)
        if bucket_name:
            so = item.get("student_output") or {}
            extras: dict[str, Any] = {}
            if bucket_name == "study_blocks":
                extras["blocks"] = list(so.get("schedule_blocks") or [])[:5]
            elif bucket_name == "assignment_tasks":
                extras["checklist"] = list(so.get("checklist") or [])[:6]
            elif bucket_name == "saved_drafts":
                email = so.get("email") or {}
                extras["subject"] = email.get("subject", "")
                extras["tone"] = email.get("tone", "")
            elif bucket_name == "resources":
                extras["resources"] = list(so.get("resources") or [])[:6]
            out[bucket_name].append({**card, "extras": extras})
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Activity-status PATCH model
# ─────────────────────────────────────────────────────────────────────────────

class HistoryStatusPatch(BaseModel):
    status: str = Field(..., pattern=r"^(saved|planned|done)$")


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@student_agent_router.get("/api/student/agent/status")
async def get_status() -> dict[str, Any]:
    return {
        "footprints": _footprint_status(),
        "chroma":     _chroma_status(),
        "ontology":   _ontology_status(),
        "gates":      _gates_status(),
    }


@student_agent_router.post("/api/student/agent/chat")
async def post_chat(req: ChatRequest) -> dict[str, Any]:
    masked_input = req.message.strip()
    # PII-mask via the existing masker; never crash on masker failure.
    try:
        from src.utils.masker import PIIMasker
        masked_input = PIIMasker().mask_data(masked_input)
    except Exception as exc:
        logger.info("PIIMasker failed; using raw input. (%s)", exc)

    degraded_details = {
        "chroma": False, "ontology": False, "gates": False,
        "ollama": False, "footprints": False,
    }
    answer = ""
    redo_log: list[dict] = []

    # ── Source-priority gate ────────────────────────────────────────────────
    # If the user's message is itself a self-contained task brief, treat that
    # message as the primary truth: skip the LLM round-trip entirely so no
    # raw VECTOR MEMORY / ONTOLOGY CONTEXT can be injected into a model
    # prompt and echoed back. The Evidence drawer (sources) is still
    # populated below from the independent vector retrieve so memory remains
    # auditable.
    is_self_contained = _is_self_contained_brief(req.message)
    parsed_brief: dict[str, Any] = {}
    if is_self_contained:
        parsed_brief = _parse_self_contained_task_brief(req.message)
        if parsed_brief.get("has_brief"):
            answer = _compose_answer_from_brief(req.message, parsed_brief)

    # ── Pipeline / LLM call ─────────────────────────────────────────────────
    if not (is_self_contained and parsed_brief.get("has_brief")):
        try:
            from src.core.schemas import AgentTask, AgentRole
            from src.pipeline.student_runner import run_pipeline
            task = AgentTask(
                session_id=f"student-agent-{uuid.uuid4().hex[:8]}",
                role=AgentRole.STUDENT,
                masked_input=masked_input,
            )
            response = run_pipeline(task)
            answer = (response.draft or "").strip()
            redo_log = list(response.redo_log or [])
        except Exception as exc:
            logger.warning("run_pipeline failed: %s", exc)
            degraded_details["ollama"] = True
            answer = "The agent is temporarily unavailable. Please try again."

    # ── Vector memory (independent surface for the UI) ──────────────────────
    sources: list[dict[str, Any]] = []
    vector_context = ""
    is_empty = True
    try:
        from src.pipeline.shared import VECTOR_MEM
        vector_context, is_empty = VECTOR_MEM.retrieve(masked_input, k=5, namespace="academic")
        if vector_context.startswith("[ChromaDB query error"):
            degraded_details["chroma"] = True
        else:
            sources = _split_vector_context(vector_context)
    except Exception as exc:
        logger.warning("VectorMemory.retrieve failed: %s", exc)
        degraded_details["chroma"] = True

    # ── Ontology context ────────────────────────────────────────────────────
    ontology_block = ""
    try:
        from src.pipeline.shared import build_ontology_context
        ontology_block = build_ontology_context()
    except Exception as exc:
        logger.warning("build_ontology_context failed: %s", exc)
        degraded_details["ontology"] = True
    ontology_ctx = _split_ontology(ontology_block)
    if not ontology_ctx.get("available"):
        degraded_details["ontology"] = True

    # ── Gate evaluation ─────────────────────────────────────────────────────
    # When a self-contained brief drove the answer, the LLM was not invoked
    # and the answer was not derived from vector memory — pass an empty
    # vector_context to the gates so grounding overlap reflects the actual
    # source of truth (the user's message), not a memory snapshot the
    # answer never depended on.
    raw_gate_report: dict[str, Any] = {}
    gate_vector_ctx = "" if (is_self_contained and parsed_brief.get("has_brief")) else vector_context
    gate_is_empty = True if (is_self_contained and parsed_brief.get("has_brief")) else is_empty
    try:
        from src.gates.evaluator import evaluate_all_gates
        raw_gate_report = evaluate_all_gates(
            answer, gate_vector_ctx, gate_is_empty, "StudentAgent", redo_log,
        )
    except Exception as exc:
        logger.warning("evaluate_all_gates failed: %s", exc)
        degraded_details["gates"] = True
    gate_report = _structure_gates(raw_gate_report)

    # ── Footprint summary ───────────────────────────────────────────────────
    fp_status = _footprint_status()
    if not fp_status.get("available"):
        degraded_details["footprints"] = True
    footprint_summary = _build_footprint_summary(fp_status)

    # ── Aggregate flags + outputs ───────────────────────────────────────────
    degraded = any(degraded_details.values())
    fallback_mode = None
    if degraded:
        flagged = [k for k, v in degraded_details.items() if v]
        fallback_mode = "Degraded subsystems: " + ", ".join(flagged)

    # Defense-in-depth: scrub any pipeline-leak artifacts. For the
    # self-contained brief path the answer is already deterministic English;
    # for the LLM path this strips raw retrieval labels, ontology brackets,
    # BLOKLANDI literals, course IDs, agent aliases, footprint event leaks,
    # and Turkish prompt-echo phrases.
    answer = _sanitize_answer(answer, has_sources=bool(sources))

    brief_mode = bool(is_self_contained and parsed_brief.get("has_brief"))

    if brief_mode:
        # Brief mode: confidence is not penalised for skipped memory.
        bumped = _bump_confidence_for_brief(
            parsed_brief, gate_report.get("overall_confidence", "low"),
        )
        gate_report["overall_confidence"] = bumped

    if brief_mode:
        study_output = _build_study_output_from_brief(
            parsed_brief, req.message, gate_report["overall_confidence"],
        )
    else:
        study_output = _build_study_output(
            answer, sources, ontology_ctx, footprint_summary,
            gate_report["overall_confidence"],
        )
    intent = _classify_intent(req.message, req.mode)
    student_output = _build_student_output(
        intent, req.message, answer, sources, ontology_ctx,
        footprint_summary, gate_report, degraded,
        brief=parsed_brief if brief_mode else None,
    )
    first_next = (study_output.get("suggested_next_steps") or [None])[0]
    if brief_mode:
        why = _build_why_for_brief(
            parsed_brief, sources, ontology_ctx, gate_report,
            degraded, fallback_mode,
        )
    else:
        why = _build_why(
            answer, sources, ontology_ctx, gate_report,
            degraded, fallback_mode, next_step=first_next,
        )
    language = _detect_language(answer)

    response = {
        "answer":            answer,
        "language_detected": language,
        "sources":           sources,
        "footprint_summary": footprint_summary,
        "ontology_context":  ontology_ctx,
        "gate_report":       gate_report,
        "study_output":      study_output,
        "student_output":    student_output,
        "degraded":          degraded,
        "degraded_details":  degraded_details,
        "fallback_mode":     fallback_mode,
        "why":               why,
    }

    # Persist to workspace history so a page refresh can restore the latest
    # result. Failure must never block the response.
    history_item: dict[str, Any] = {}
    try:
        history_item = _build_history_item(req.message, req.mode, response)
        _append_history(history_item)
    except Exception as exc:
        logger.warning("history append failed: %s", exc)

    # Emit a live footprint event so this turn is reachable through the
    # academic Chroma memory on later retrievals. Never break the response.
    try:
        from src.services.api.student_live_footprint import record_live_footprint
        so = response.get("student_output") or {}
        intent = so.get("intent") or "academic_qa"
        title = (so.get("title") or "").strip() or (req.message or "").strip()[:80]
        record_live_footprint(
            text=(
                f"Student created {intent} output: {title}. "
                f"Original request: {masked_input}"
            ),
            kind="chat_turn",
            source="student_agent_chat",
            occurred_at=history_item.get("created_at"),
            derived_from=history_item.get("id"),
            metadata={"intent": intent, "mode": req.mode or ""},
        )
    except Exception as exc:
        logger.info("live footprint (chat_turn) emit failed: %s", exc)

    return response


@student_agent_router.get("/api/student/agent/history")
async def get_history() -> dict[str, Any]:
    items = _load_history()
    return {"items": items[:_HISTORY_LIMIT]}


@student_agent_router.get("/api/student/agent/history/latest")
async def get_history_latest() -> Optional[dict[str, Any]]:
    items = _load_history()
    return items[0] if items else None


@student_agent_router.delete("/api/student/agent/history")
async def delete_history() -> dict[str, Any]:
    _save_history([])
    return {"ok": True}


@student_agent_router.patch("/api/student/agent/history/{item_id}")
async def patch_history(item_id: str, req: HistoryStatusPatch) -> dict[str, Any]:
    items = _load_history()
    for item in items:
        if item.get("id") == item_id:
            item["status"] = req.status
            _save_history(items)
            if req.status in ("planned", "done"):
                try:
                    from src.services.api.student_live_footprint import record_live_footprint
                    title = (item.get("title") or "(untitled)").strip()
                    record_live_footprint(
                        text=f"Student marked {title} as {req.status}.",
                        kind="agenda_event",
                        source="student_agent_history",
                        occurred_at=datetime.now(timezone.utc).isoformat(),
                        derived_from=item_id,
                        metadata={
                            "status": req.status,
                            "intent": item.get("intent") or "",
                        },
                    )
                except Exception as exc:
                    logger.info("live footprint (agenda_event) emit failed: %s", exc)
            return {"ok": True, "item": item}
    raise HTTPException(status_code=404, detail="not found")


@student_agent_router.get("/api/student/agent/agenda")
async def get_agenda() -> dict[str, list[dict[str, Any]]]:
    return _build_agenda(_load_history())


@student_agent_router.post("/api/student/agent/note")
async def post_note(req: NoteRequest) -> dict[str, Any]:
    notes_path = _project_root() / "data" / "student_notes.json"
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if notes_path.exists():
            existing = json.loads(notes_path.read_text(encoding="utf-8") or "[]")
            if not isinstance(existing, list):
                existing = []
        else:
            existing = []
    except Exception:
        existing = []
    note = {
        "id": f"note-{uuid.uuid4().hex[:10]}",
        "title": req.title.strip(),
        "body": req.body.strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    existing.append(note)
    try:
        notes_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to write student note: %s", exc)
        raise HTTPException(status_code=500, detail="Could not persist note.") from exc
    return {"ok": True, "id": note["id"]}
