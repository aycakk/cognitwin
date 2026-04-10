"""pipeline/scrum_master_runner.py — Scrum Master pipeline path.

Rule-based pipeline.  Does NOT use ChromaDB or vector memory.
Grounds responses in local sprint state (data/sprint_state.json).

Gate coverage (different from student/developer paths):
  C1  — PII leak detection (same as all paths)
  C4  — Hallucination marker sweep (same as all paths)
  C2/C7 are SKIPPED: this agent is not retrieval-based so
        "empty memory" and "grounding" gates do not apply.

The ScrumMasterAgent produces a fully deterministic response from
local sprint state before any gate check is applied.
"""

from __future__ import annotations

import re

from src.pipeline.shared import SCRUM_AGENT as _agent

# ─────────────────────────────────────────────────────────────────────────────
#  Gate patterns (local copies — no import from shared to keep the runner
#  self-contained and independent of the vector-memory singletons)
# ─────────────────────────────────────────────────────────────────────────────

_PII_RE = re.compile(
    r"\b\d{9,12}\b"                                        # TC / student ID
    r"|[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}"     # e-mail
    r"|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",                # phone
)

_HALLUCINATION_RE = re.compile(
    r"sanırım|galiba|muhtemelen|tahmin\s+etmek|belki|herhalde",
    re.I,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_scrum_master_pipeline(query: str) -> str:
    """
    Execute the Scrum Master rule pipeline.

    Stage 1 — Rule-based processing (ScrumMasterAgent)
              Reads local sprint state; no LLM, no ChromaDB.

    Stage 2 — C1 gate: PII leak check
              Blocks response if unmasked PII is present in the output.

    Stage 3 — C4 gate: hallucination marker sweep
              Replaces hedging phrases with a neutral marker.

    Stage 4 — Emission
              Returns the verified response.
    """
    # ── Stage 1 — Rule engine ─────────────────────────────────────────────
    response = _agent.handle_query(query)

    # ── Stage 2 — C1: PII leak ────────────────────────────────────────────
    if _PII_RE.search(response):
        return (
            "⚠ Scrum Master çıktısında kişisel veri tespit edildi. "
            "Bu bilgi paylaşılamaz."
        )

    # ── Stage 3 — C4: Hallucination markers ──────────────────────────────
    if _HALLUCINATION_RE.search(response):
        response = _HALLUCINATION_RE.sub("[doğrulanmamış]", response)

    # ── Stage 4 — Emission ────────────────────────────────────────────────
    return response
