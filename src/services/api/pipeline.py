"""
pipeline.py — CogniTwin Production Pipeline (ZT4SWE v2.3)

Architecture  : Multi-Agent Hybrid (Vector Memory + Structured Ontology)
Gate Array    : C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5 ∧ C6 ∧ C7 ∧ C8  (BOTH paths)
Memory        : ChromaDB  (k=15)
Ontology      : cognitwin-upper.ttl + student_ontology.ttl  (rdflib, lazy-loaded)
LLM           : Ollama llama3.2  (local)
Routing       : model name → student (default) | developer (cognitwin-developer)

Entry points (called by routes.py and openai_routes.py):
  process_user_message(user_text, agent_role, model, messages) -> dict
"""

from __future__ import annotations

from src.utils.masker import PIIMasker
from src.shared.permissions import ONTOLOGY_AGENT_ROLES
from src.gates.evaluator import evaluate_all_gates  # noqa: F401 (re-exported for main_cli)
from src.ontology.loader import _get_ontology_graph, _sparql  # noqa: F401 (re-exported for main_cli)
from src.pipeline.router import resolve_mode
from src.pipeline.shared import (           # noqa: F401 (several re-exported for main_cli)
    VECTOR_TOP_K,
    BLINDSPOT_TRIGGERS,
    _LABEL_RE,
    SYSTEM_PROMPT,
    CHROMA,
    VectorMemory,
    VECTOR_MEM,
    build_blindspot_block,
    build_ontology_context,
    _sanitize_output,
    _safe_chat,
)
from src.pipeline.student_runner import run_pipeline       # noqa: F401 (re-exported for main_cli)
from src.pipeline.developer_runner import _process_developer_message

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CHROMA_PATH = "static/chromadb"

# ONTOLOGY_AGENT_ROLES is imported from src.shared and re-exported for main_cli.py.
# VECTOR_TOP_K, BLINDSPOT_TRIGGERS, _LABEL_RE, SYSTEM_PROMPT, CHROMA, VectorMemory,
# VECTOR_MEM, build_blindspot_block, build_ontology_context, _sanitize_output, and
# _safe_chat are all imported from src.pipeline.shared (see imports above).

# ─────────────────────────────────────────────────────────────────────────────
#  MODULE-LEVEL SINGLETONS
# ─────────────────────────────────────────────────────────────────────────────

_masker = PIIMasker()


# ─────────────────────────────────────────────────────────────────────────────
#  LAZY ONTOLOGY LOADING
#  _get_ontology_graph and _sparql are imported from src.ontology.loader.
#  They are re-exported here so that main_cli.py's existing import
#  (from src.services.api.pipeline import _get_ontology_graph) continues
#  to work without modification.
# ─────────────────────────────────────────────────────────────────────────────

# VectorMemory, VECTOR_MEM, CHROMA are imported from src.pipeline.shared above.

# run_pipeline is imported from src.pipeline.student_runner (re-exported for main_cli).
# _process_developer_message is imported from src.pipeline.developer_runner.

# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API  (called by routes.py and openai_routes.py)
# ─────────────────────────────────────────────────────────────────────────────

def process_user_message(
    user_text: str,
    agent_role: str = "StudentAgent",
    model: str = "llama3.2",
    messages: list | None = None,
) -> dict:
    """
    Mask PII, resolve routing mode, execute the appropriate pipeline.

    Routing:
      model contains 'developer'  →  DeveloperOrchestrator + C1-C8 + REDO
      all other models             →  Student ZT4SWE pipeline (C1-C8 + REDO)

    Returns {"answer": str} for the FastAPI layer.
    """
    if "title for the conversation" in user_text.lower():
        return {"answer": "Conversation Title"}
    try:
        masked = _masker.mask_data(user_text)
        mode, strategy = resolve_mode(model)

        if mode == "developer":
            answer = _process_developer_message(
                user_text=masked,
                strategy=strategy,
                messages=messages,
            )
        else:
            answer = run_pipeline(masked, agent_role=agent_role)

        return {"answer": answer}
    except Exception as exc:
        return {"answer": f"İşlem sırasında bir hata oluştu: {exc}"}
