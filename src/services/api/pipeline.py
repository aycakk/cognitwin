"""
pipeline.py — CogniTwin Production Pipeline (ZT4SWE v2.3)

Architecture  : Multi-Agent Hybrid (Vector Memory + Structured Ontology)
Gate Array    : C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5 ∧ C6 ∧ C7 ∧ C8  (BOTH paths)
Memory        : ChromaDB  (k=15)
Ontology      : cognitwin-upper.ttl + student_ontology.ttl  (rdflib, lazy-loaded)
LLM           : Ollama llama3.2  (local)
Routing       : model name → student (default) | developer | scrum | product_owner | composer

Entry points (called by routes.py and openai_routes.py):
  process_user_message(user_text, agent_role, model, messages) -> dict
"""

from __future__ import annotations

from src.core.schemas import AgentTask, AgentRole
from src.utils.masker import PIIMasker
from src.shared.permissions import ONTOLOGY_AGENT_ROLES
from src.gates.evaluator import evaluate_all_gates  # noqa: F401 (re-exported for main_cli)
from src.ontology.loader import _get_ontology_graph, _sparql  # noqa: F401 (re-exported for main_cli)
from src.pipeline.router import resolve_mode, UnknownModelError
from src.pipeline.shared import (           # noqa: F401 (several re-exported for main_cli)
    DEFAULT_MODEL,
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
from src.pipeline.scrum_master_runner import run_scrum_master_pipeline
from src.pipeline.product_owner_runner import run_product_owner_pipeline
from src.pipeline.composer_runner import run_composer_pipeline

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
    model: str = DEFAULT_MODEL,
    messages: list | None = None,
    session_id: str = "",
) -> dict:
    """
    Mask PII, resolve routing mode, execute the appropriate pipeline.

    Routing:
      model contains 'composer'      →  Composer deterministic orchestration pipeline
      model contains 'product_owner' →  ProductOwner rule pipeline (backlog)
      model contains 'developer'     →  DeveloperOrchestrator + C1-C8 + REDO
      model contains 'scrum'         →  ScrumMaster rule pipeline
      all other models               →  Student ZT4SWE pipeline (C1-C8 + REDO)

    Returns {"answer": str} for the FastAPI layer.
    """
    if "title for the conversation" in user_text.lower():
        return {"answer": "Conversation Title"}
    try:
        masked = _masker.mask_data(user_text)
        mode, strategy = resolve_mode(model)

        if mode == "sprint":
            # Autonomous advisor-upgrade pipeline (run_sprint).
            # Bypasses the AgentTask/AgentResponse contract because run_sprint
            # is a multi-step sprint runner, not a single agent call.
            from src.services.api.sprint_bridge import run_sprint_for_ui  # noqa: PLC0415
            return run_sprint_for_ui(masked)

        if mode == "developer":
            task = AgentTask(
                session_id=session_id,
                role=AgentRole.DEVELOPER,
                masked_input=masked,
                metadata={
                    "strategy": strategy,
                    "messages": messages or [],
                },
            )
            response = _process_developer_message(task)
        elif mode == "composer":
            task = AgentTask(
                session_id=session_id,
                role=AgentRole.COMPOSER,
                masked_input=masked,
            )
            response = run_composer_pipeline(task)
        elif mode == "product_owner":
            from src.pipeline.agile_workflow import is_workflow_request, run_agile_workflow
            task = AgentTask(
                session_id=session_id,
                role=AgentRole.PRODUCT_OWNER,
                masked_input=masked,
            )
            if is_workflow_request(masked):
                # Full PO-first chain: PO → [C] → SM → [C] → Dev → [C] → PO review
                response = run_agile_workflow(task)
            else:
                response = run_product_owner_pipeline(task)
        elif mode == "scrum_master":
            task = AgentTask(
                session_id=session_id,
                role=AgentRole.SCRUM_MASTER,
                masked_input=masked,
            )
            response = run_scrum_master_pipeline(task)
        else:
            try:
                role = AgentRole(agent_role)
            except ValueError:
                role = AgentRole.STUDENT
            task = AgentTask(
                session_id=session_id,
                role=role,
                masked_input=masked,
            )
            response = run_pipeline(task)

        # Include workflow session metadata when available so the HTTP layer
        # can surface child session IDs to the client.
        result: dict = {"answer": response.draft}
        wf_meta = (response.metadata or {}).get("workflow")
        if wf_meta:
            result["workflow_meta"] = wf_meta
        return result
    except UnknownModelError as exc:
        # Do NOT swallow this — it means LibreChat sent a model name we have
        # never seen.  Return a clear message so the user (and logs) know.
        return {"answer": f"[Yönlendirme Hatası] {exc}"}
    except Exception as exc:
        return {"answer": f"İşlem sırasında bir hata oluştu: {exc}"}
