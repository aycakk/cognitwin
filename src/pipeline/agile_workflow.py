"""pipeline/agile_workflow.py — Sequential Agile team orchestration.

Architecture
────────────
This module implements the full project-to-delivery workflow.  Unlike the
parallel dispatcher in composer_runner.py (which routes a single query to
one or more agents simultaneously), this workflow chains agent outputs:

  STEP 1  Product Owner   — Receives raw project request.
                            Converts it into structured user stories with
                            acceptance criteria and priority.

  STEP 2  Scrum Master    — Receives PO output as enriched context.
                            Plans the sprint, assigns tasks, and evaluates
                            sprint health using the SM ontology (agile.ttl +
                            scrum_master.ttl) and the real sprint state.

  STEP 3  Developer       — Receives SM sprint plan as enriched context.
                            Executes assigned tasks, provides technical
                            assessment, and documents implementation steps.

  STEP 4  Composer        — Collects all three outputs and merges them into
                            one coherent, structured final response via
                            ComposerAgent.compose().

Context passing
───────────────
Each step builds a fresh AgentTask whose masked_input is an enriched
prompt containing the previous agent's output.  The original query is
always preserved in the enriched prompt so agents have full project context.
State (sprint_state.json) is shared through SprintStateStore — no extra
wiring needed.

Triggering
──────────
Activated from composer_runner.run_composer_pipeline() when
is_workflow_request(query) returns True.  All other Composer queries
continue to use the existing parallel dispatch path unchanged.

SM intent routing
─────────────────
The SM prompt is written to trigger the `sprint_analysis` intent
(keywords: analiz, değerlendir, öneri) which routes to the LLM-augmented
path.  This gives the SM access to ontology context + sprint state when
reasoning about the PO's plan — exactly the right behaviour for the
orchestration use case.

PO intent routing
─────────────────
The PO prompt includes "kullanıcı hikayeleri oluştur" to trigger the
`create_story` deterministic intent regardless of the original query phrasing.

Failure handling
────────────────
Each step is isolated in a try/except.  If a step fails or returns unusable
output the workflow continues with what it has.  The final ComposerAgent
merges only successful outputs and records warnings for failed steps in the
response metadata.
"""

from __future__ import annotations

import logging
import re

from src.agents.composer_agent import ComposerAgent
from src.core.schemas import AgentTask, AgentResponse, AgentRole, TaskStatus
from src.core.session_store import SESSION_STORE
from src.pipeline.developer_runner import _process_developer_message
from src.pipeline.product_owner_runner import run_product_owner_pipeline
from src.pipeline.scrum_master_runner import run_scrum_master_pipeline

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Workflow trigger detection
#  Patterns that indicate a full PO→SM→Developer chain is appropriate.
#  Single-agent targeted queries (e.g. "sprint durumu nedir?") are NOT
#  captured here — those continue through the parallel dispatch path.
# ─────────────────────────────────────────────────────────────────────────────

_WORKFLOW_RE = re.compile(
    r"proje\s+(?:başlat|planla|oluştur|yarat|kur|geliştir|teslim)"
    r"|yeni\s+proje"
    r"|sprint\s+(?:planla|başlat|organize\s+et)"
    r"|(?:tam|bütün|komple|eksiksiz)\s+(?:süreç|workflow|akış|sprint)"
    r"|agile\s+workflow"
    r"|(?:baştan\s+)?planla\s+ve\s+(?:geliştir|uygula|kodla|teslim\s+et)"
    r"|feature\s+request"
    r"|geliştirme\s+isteği"
    r"|proje\s+isteği"
    r"|end.to.end|u[çc]tan\s+uca"
    r"|tam\s+sprint"
    r"|sıfırdan\s+(?:planla|başlat|geliştir)"
    r"|proje\s+(?:teslim|delivery|deliverable)",
    re.I,
)


def is_workflow_request(query: str) -> bool:
    """Return True when the query describes a full project / sprint workflow.

    Used by composer_runner to decide between sequential orchestration
    (this module) and the existing parallel single-query dispatch.
    """
    return bool(_WORKFLOW_RE.search(query or ""))


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _effective_session_id(task: AgentTask) -> str:
    """Return a non-empty session ID for a task.

    Falls back to task_id when session_id is blank so every workflow run
    always has a stable, unique identifier even if the HTTP layer didn't
    supply one.
    """
    return task.session_id or task.task_id


def _agent_session_id(parent_session: str, role_slug: str) -> str:
    """Build the agent-scoped child session ID.

    Convention:  <parent_session>/<role_slug>
    Examples:    conv-abc-123/po   conv-abc-123/sm   conv-abc-123/dev
    """
    return f"{parent_session}/{role_slug}"


def _make_child_task(
    parent: AgentTask,
    role: AgentRole,
    enriched_input: str,
    agent_session_id: str,
    extra_context: dict | None = None,
) -> AgentTask:
    """Build a child AgentTask with its own agent session ID.

    Each step gets a UNIQUE session_id (e.g. conv-abc/po) so its output is
    stored and retrievable independently.  parent_task_id links it back to
    the originating task for the orchestrator.
    """
    ctx = {**parent.context}
    if extra_context:
        ctx.update(extra_context)
    return AgentTask(
        session_id=agent_session_id,          # ← agent-specific, not parent's
        role=role,
        masked_input=enriched_input,
        parent_task_id=parent.task_id,
        context=ctx,
        metadata={**parent.metadata},
    )


def _extract_draft(response: AgentResponse) -> str:
    """Pull the text from an AgentResponse, returning '' on failure."""
    draft = getattr(response, "draft", None)
    return str(draft).strip() if draft else ""


def _usable(text: str) -> bool:
    """Return True if the text is long enough to be meaningful."""
    return bool(text) and len(text.strip()) >= 15


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1 — Product Owner
# ─────────────────────────────────────────────────────────────────────────────

def _step_product_owner(
    parent: AgentTask,
    parent_session: str,
) -> tuple[AgentResponse, str, str]:
    """Convert the raw project request into structured user stories.

    Returns (response, draft_text, po_session_id).
    """
    original  = parent.masked_input
    po_sid    = _agent_session_id(parent_session, "po")

    if re.search(r"hikaye|story|backlog", original, re.I):
        po_input = original
    else:
        po_input = (
            f"Aşağıdaki proje isteği için kullanıcı hikayeleri oluştur "
            f"ve her hikaye için kabul kriterlerini ve öncelik sırasını belirle:\n\n"
            f"{original}"
        )

    SESSION_STORE.create_session(
        session_id=po_sid,
        agent_role=AgentRole.PRODUCT_OWNER.value,
        query=po_input,
        parent_session_id=parent_session,
        metadata={"workflow_step": "product_owner"},
    )

    task = _make_child_task(
        parent,
        AgentRole.PRODUCT_OWNER,
        po_input,
        agent_session_id=po_sid,
        extra_context={"workflow_step": "product_owner", "original_request": original},
    )
    response = run_product_owner_pipeline(task)
    draft    = _extract_draft(response)

    SESSION_STORE.record_output(
        session_id=po_sid,
        output=draft,
        status=response.status.value if hasattr(response.status, "value") else str(response.status),
        metadata={"task_id": task.task_id},
    )
    return response, draft, po_sid


# ─────────────────────────────────────────────────────────────────────────────
#  Step 2 — Scrum Master
# ─────────────────────────────────────────────────────────────────────────────

def _step_scrum_master(
    parent: AgentTask,
    po_output: str,
    parent_session: str,
) -> tuple[AgentResponse, str, str]:
    """Plan the sprint and assign tasks based on PO output.

    Returns (response, draft_text, sm_session_id).
    """
    original = parent.masked_input
    sm_sid   = _agent_session_id(parent_session, "sm")
    sm_input = (
        f"Proje isteği: {original}\n\n"
        f"Ürün Sahibi (PO) tarafından oluşturulan hikayeler ve iş kalemleri:\n"
        f"{po_output}\n\n"
        f"Yukarıdaki proje gereksinimlerini analiz et ve değerlendir: "
        f"mevcut sprint kapasitesine göre hangi görevler önceliklendirilmeli, "
        f"riskler neler, atamalar nasıl yapılmalı? "
        f"Sprint sağlığını değerlendir ve somut öneriler sun."
    )

    SESSION_STORE.create_session(
        session_id=sm_sid,
        agent_role=AgentRole.SCRUM_MASTER.value,
        query=sm_input,
        parent_session_id=parent_session,
        metadata={"workflow_step": "scrum_master"},
    )

    task = _make_child_task(
        parent,
        AgentRole.SCRUM_MASTER,
        sm_input,
        agent_session_id=sm_sid,
        extra_context={
            "workflow_step": "scrum_master",
            "po_output": po_output,
            "original_request": original,
        },
    )
    response = run_scrum_master_pipeline(task)
    draft    = _extract_draft(response)

    SESSION_STORE.record_output(
        session_id=sm_sid,
        output=draft,
        status=response.status.value if hasattr(response.status, "value") else str(response.status),
        metadata={"task_id": task.task_id},
    )
    return response, draft, sm_sid


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3 — Developer
# ─────────────────────────────────────────────────────────────────────────────

def _step_developer(
    parent: AgentTask,
    sm_output: str,
    po_output: str,
    parent_session: str,
) -> tuple[AgentResponse, str, str]:
    """Execute assigned tasks based on the SM sprint plan.

    Returns (response, draft_text, dev_session_id).
    """
    original  = parent.masked_input
    dev_sid   = _agent_session_id(parent_session, "dev")
    dev_input = (
        f"Proje isteği: {original}\n\n"
        f"Scrum Master sprint planı ve görev atamaları:\n{sm_output}\n\n"
        f"Referans — Ürün Sahibi hikayeleri:\n{po_output}\n\n"
        f"Yukarıdaki sprint planına göre atanan görevleri teknik açıdan "
        f"değerlendir ve uygulama adımlarını belirt."
    )

    SESSION_STORE.create_session(
        session_id=dev_sid,
        agent_role=AgentRole.DEVELOPER.value,
        query=dev_input,
        parent_session_id=parent_session,
        metadata={"workflow_step": "developer"},
    )

    task = _make_child_task(
        parent,
        AgentRole.DEVELOPER,
        dev_input,
        agent_session_id=dev_sid,
        extra_context={
            "workflow_step": "developer",
            "sm_output": sm_output,
            "po_output": po_output,
            "original_request": original,
        },
    )
    task.metadata["strategy"] = task.metadata.get("strategy", "auto")

    response = _process_developer_message(task)
    draft    = _extract_draft(response)

    SESSION_STORE.record_output(
        session_id=dev_sid,
        output=draft,
        status=response.status.value if hasattr(response.status, "value") else str(response.status),
        metadata={"task_id": task.task_id},
    )
    return response, draft, dev_sid


# ─────────────────────────────────────────────────────────────────────────────
#  Main orchestration entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_agile_workflow(task: AgentTask) -> AgentResponse:
    """Execute the full sequential Agile workflow: PO → SM → Developer → Composer.

    Each agent receives the previous agent's output as enriched context
    (not just the raw original query).  Composer merges all outputs into
    a single structured final response.

    Failure handling
    ────────────────
    Each step is wrapped in try/except.  If a step fails or produces unusable
    output, the workflow continues with what it has and notes the failure in
    warnings.  A completely empty pipeline emits a safe FAILED response.

    Returns
    ───────
    AgentResponse with:
      • agent_role = COMPOSER
      • draft      = final merged text from ComposerAgent.compose()
      • status     = COMPLETED (even if some steps failed, as long as ≥1 succeeded)
      • metadata["workflow"] = step counts, agent list, warnings
    """
    # Determine a stable session ID for this workflow run.
    # Falls back to task_id when the HTTP layer didn't supply a session_id.
    parent_session = _effective_session_id(task)

    logger.info(
        "agile-workflow: starting PO->SM->Developer  task=%s  session=%s",
        task.task_id, parent_session,
    )

    # Register the parent (Composer) session so children can attach to it.
    SESSION_STORE.create_session(
        session_id=parent_session,
        agent_role=AgentRole.COMPOSER.value,
        query=task.masked_input,
        parent_session_id=None,
        metadata={"workflow": "agile_sequential"},
    )

    collected:    list[dict[str, str]] = []   # {"agent": ..., "draft": ...}
    warnings:     list[str]            = []
    child_sessions: list[str]          = []

    # ── Step 1: Product Owner ─────────────────────────────────────────────────
    po_text = ""
    po_sid  = ""
    try:
        _, po_text, po_sid = _step_product_owner(task, parent_session)
        child_sessions.append(po_sid)
        if _usable(po_text):
            collected.append({"agent": AgentRole.PRODUCT_OWNER.value, "draft": po_text})
            logger.info("agile-workflow: PO OK  session=%s  (%d chars)", po_sid, len(po_text))
        else:
            warnings.append("Ürün Sahibi: kullanılabilir çıktı üretilemedi.")
            po_text = ""
            logger.warning("agile-workflow: PO step returned unusable output  session=%s", po_sid)
    except Exception as exc:
        warnings.append(f"Ürün Sahibi adımı başarısız oldu: {exc}")
        po_text = ""
        logger.error("agile-workflow: PO step failed: %s", exc, exc_info=True)

    # ── Step 2: Scrum Master ──────────────────────────────────────────────────
    sm_context = po_text if _usable(po_text) else f"Proje isteği: {task.masked_input}"
    sm_text = ""
    sm_sid  = ""
    try:
        _, sm_text, sm_sid = _step_scrum_master(task, sm_context, parent_session)
        child_sessions.append(sm_sid)
        if _usable(sm_text):
            collected.append({"agent": AgentRole.SCRUM_MASTER.value, "draft": sm_text})
            logger.info("agile-workflow: SM OK  session=%s  (%d chars)", sm_sid, len(sm_text))
        else:
            warnings.append("Scrum Master: kullanılabilir çıktı üretilemedi.")
            sm_text = ""
            logger.warning("agile-workflow: SM step returned unusable output  session=%s", sm_sid)
    except Exception as exc:
        warnings.append(f"Scrum Master adımı başarısız oldu: {exc}")
        sm_text = ""
        logger.error("agile-workflow: SM step failed: %s", exc, exc_info=True)

    # ── Step 3: Developer ─────────────────────────────────────────────────────
    dev_context = sm_text if _usable(sm_text) else sm_context
    dev_sid     = ""
    try:
        _, dev_text, dev_sid = _step_developer(task, dev_context, po_text, parent_session)
        child_sessions.append(dev_sid)
        if _usable(dev_text):
            collected.append({"agent": AgentRole.DEVELOPER.value, "draft": dev_text})
            logger.info("agile-workflow: Dev OK  session=%s  (%d chars)", dev_sid, len(dev_text))
        else:
            warnings.append("Developer: kullanılabilir çıktı üretilemedi.")
            logger.warning("agile-workflow: Developer step returned unusable output  session=%s", dev_sid)
    except Exception as exc:
        warnings.append(f"Developer adımı başarısız oldu: {exc}")
        logger.error("agile-workflow: Developer step failed: %s", exc, exc_info=True)

    # ── Step 4: Composer — merge all outputs ──────────────────────────────────
    composer = ComposerAgent()

    if not collected:
        response_text = composer.format_final_response(
            summary="Agile workflow tüm adımlarda başarısız oldu.",
            key_points=["Hiçbir ajan kullanılabilir çıktı üretemedi."],
            warnings=warnings or ["Bilinmeyen pipeline hatası."],
            final_answer=(
                "Agile workflow tamamlanamadı. "
                "Lütfen isteği yeniden belirtin veya sistem yöneticisiyle iletişime geçin."
            ),
        )
        SESSION_STORE.record_output(
            session_id=parent_session,
            output=response_text,
            status="failed",
            metadata={"child_sessions": child_sessions, "warnings": warnings},
        )
        return AgentResponse(
            task_id=task.task_id,
            agent_role=AgentRole.COMPOSER,
            draft=response_text,
            status=TaskStatus.FAILED,
            metadata={
                "workflow": {
                    "session_id":      parent_session,
                    "child_sessions":  child_sessions,
                    "steps_completed": 0,
                    "steps_attempted": 3,
                    "agents_used":     [],
                    "warnings":        warnings,
                }
            },
        )

    composed = composer.compose(collected)

    workflow_meta: dict = {
        "session_id":      parent_session,
        "child_sessions":  child_sessions,   # ← PO, SM, Developer session IDs
        "steps_completed": len(collected),
        "steps_attempted": 3,
        "agents_used":     [s["agent"] for s in collected],
        "warnings":        warnings,
        "useful_count":    composed.get("useful_count", 0),
        "merged_count":    composed.get("merged_count", 0),
        "conflict_count":  len(composed.get("conflicts", [])),
    }

    # Record the Composer's final merged output back to the parent session.
    SESSION_STORE.record_output(
        session_id=parent_session,
        output=composed["response_text"],
        status="completed",
        metadata=workflow_meta,
    )

    logger.info(
        "agile-workflow: complete  session=%s  %d/3 steps  children=%s",
        parent_session, len(collected), child_sessions,
    )

    return AgentResponse(
        task_id=task.task_id,
        agent_role=AgentRole.COMPOSER,
        draft=composed["response_text"],
        status=TaskStatus.COMPLETED,
        metadata={"workflow": workflow_meta},
    )
