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

def _make_child_task(
    parent: AgentTask,
    role: AgentRole,
    enriched_input: str,
    extra_context: dict | None = None,
) -> AgentTask:
    """Build a child AgentTask that carries parent context forward.

    The child inherits session_id, metadata and existing context keys from
    the parent.  extra_context is merged on top so each step can annotate
    what it contributed without overwriting parent data.
    """
    ctx = {**parent.context}
    if extra_context:
        ctx.update(extra_context)
    return AgentTask(
        session_id=parent.session_id,
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

def _step_product_owner(parent: AgentTask) -> tuple[AgentResponse, str]:
    """Convert the raw project request into structured user stories.

    Wraps the original query in a prompt that explicitly triggers the PO
    agent's `create_story` deterministic intent, ensuring consistent
    backlog output regardless of the original phrasing.
    """
    original = parent.masked_input

    # Include "kullanıcı hikayeleri oluştur" to reliably trigger create_story.
    # The full original query is preserved so PO has project context.
    if re.search(r"hikaye|story|backlog", original, re.I):
        # Query already carries PO vocabulary — use as-is.
        po_input = original
    else:
        po_input = (
            f"Aşağıdaki proje isteği için kullanıcı hikayeleri oluştur "
            f"ve her hikaye için kabul kriterlerini ve öncelik sırasını belirle:\n\n"
            f"{original}"
        )

    task = _make_child_task(
        parent,
        AgentRole.PRODUCT_OWNER,
        po_input,
        extra_context={"workflow_step": "product_owner", "original_request": original},
    )
    response = run_product_owner_pipeline(task)
    return response, _extract_draft(response)


# ─────────────────────────────────────────────────────────────────────────────
#  Step 2 — Scrum Master
# ─────────────────────────────────────────────────────────────────────────────

def _step_scrum_master(
    parent: AgentTask,
    po_output: str,
) -> tuple[AgentResponse, str]:
    """Plan the sprint and assign tasks based on PO output.

    The prompt is crafted to trigger the SM's `sprint_analysis` LLM-augmented
    path (keywords: analiz, değerlendir, öneri) so the SM agent has access
    to both the ontology context and real sprint state when reasoning.
    """
    original = parent.masked_input
    sm_input = (
        f"Proje isteği: {original}\n\n"
        f"Ürün Sahibi (PO) tarafından oluşturulan hikayeler ve iş kalemleri:\n"
        f"{po_output}\n\n"
        f"Yukarıdaki proje gereksinimlerini analiz et ve değerlendir: "
        f"mevcut sprint kapasitesine göre hangi görevler önceliklendirilmeli, "
        f"riskler neler, atamalar nasıl yapılmalı? "
        f"Sprint sağlığını değerlendir ve somut öneriler sun."
    )

    task = _make_child_task(
        parent,
        AgentRole.SCRUM_MASTER,
        sm_input,
        extra_context={
            "workflow_step": "scrum_master",
            "po_output": po_output,
            "original_request": parent.masked_input,
        },
    )
    response = run_scrum_master_pipeline(task)
    return response, _extract_draft(response)


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3 — Developer
# ─────────────────────────────────────────────────────────────────────────────

def _step_developer(
    parent: AgentTask,
    sm_output: str,
    po_output: str,
) -> tuple[AgentResponse, str]:
    """Execute assigned tasks based on the SM sprint plan.

    Passes both the SM plan and the original PO stories as context so
    the Developer agent can align technical work with acceptance criteria.
    """
    original = parent.masked_input
    dev_input = (
        f"Proje isteği: {original}\n\n"
        f"Scrum Master sprint planı ve görev atamaları:\n{sm_output}\n\n"
        f"Referans — Ürün Sahibi hikayeleri:\n{po_output}\n\n"
        f"Yukarıdaki sprint planına göre atanan görevleri teknik açıdan "
        f"değerlendir ve uygulama adımlarını belirt."
    )

    task = _make_child_task(
        parent,
        AgentRole.DEVELOPER,
        dev_input,
        extra_context={
            "workflow_step": "developer",
            "sm_output": sm_output,
            "po_output": po_output,
            "original_request": original,
        },
    )
    # Developer runner requires strategy in metadata.
    task.metadata["strategy"] = task.metadata.get("strategy", "auto")

    response = _process_developer_message(task)
    return response, _extract_draft(response)


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
    logger.info(
        "agile-workflow: starting PO→SM→Developer chain  task=%s", task.task_id
    )

    collected: list[dict[str, str]] = []   # {"agent": ..., "draft": ...}
    warnings:  list[str]            = []

    # ── Step 1: Product Owner ─────────────────────────────────────────────────
    po_text = ""
    try:
        _, po_text = _step_product_owner(task)
        if _usable(po_text):
            collected.append({"agent": AgentRole.PRODUCT_OWNER.value, "draft": po_text})
            logger.info("agile-workflow: PO step OK  (%d chars)", len(po_text))
        else:
            warnings.append("Ürün Sahibi: kullanılabilir çıktı üretilemedi.")
            po_text = ""
            logger.warning("agile-workflow: PO step returned unusable output")
    except Exception as exc:
        warnings.append(f"Ürün Sahibi adımı başarısız oldu: {exc}")
        po_text = ""
        logger.error("agile-workflow: PO step failed: %s", exc, exc_info=True)

    # ── Step 2: Scrum Master ──────────────────────────────────────────────────
    # If PO produced nothing, fall back to the original query as SM context
    # so the sprint planner still has the project description.
    sm_context = po_text if _usable(po_text) else f"Proje isteği: {task.masked_input}"
    sm_text = ""
    try:
        _, sm_text = _step_scrum_master(task, sm_context)
        if _usable(sm_text):
            collected.append({"agent": AgentRole.SCRUM_MASTER.value, "draft": sm_text})
            logger.info("agile-workflow: SM step OK  (%d chars)", len(sm_text))
        else:
            warnings.append("Scrum Master: kullanılabilir çıktı üretilemedi.")
            sm_text = ""
            logger.warning("agile-workflow: SM step returned unusable output")
    except Exception as exc:
        warnings.append(f"Scrum Master adımı başarısız oldu: {exc}")
        sm_text = ""
        logger.error("agile-workflow: SM step failed: %s", exc, exc_info=True)

    # ── Step 3: Developer ─────────────────────────────────────────────────────
    # Developer receives SM output; fall back to sm_context if SM failed.
    dev_context = sm_text if _usable(sm_text) else sm_context
    try:
        _, dev_text = _step_developer(task, dev_context, po_text)
        if _usable(dev_text):
            collected.append({"agent": AgentRole.DEVELOPER.value, "draft": dev_text})
            logger.info("agile-workflow: Developer step OK  (%d chars)", len(dev_text))
        else:
            warnings.append("Developer: kullanılabilir çıktı üretilemedi.")
            logger.warning("agile-workflow: Developer step returned unusable output")
    except Exception as exc:
        warnings.append(f"Developer adımı başarısız oldu: {exc}")
        logger.error("agile-workflow: Developer step failed: %s", exc, exc_info=True)

    # ── Step 4: Composer — merge all outputs ──────────────────────────────────
    composer = ComposerAgent()

    if not collected:
        # Every step failed — safe failure response.
        response_text = composer.format_final_response(
            summary="Agile workflow tüm adımlarda başarısız oldu.",
            key_points=["Hiçbir ajan kullanılabilir çıktı üretemedi."],
            warnings=warnings or ["Bilinmeyen pipeline hatası."],
            final_answer=(
                "Agile workflow tamamlanamadı. "
                "Lütfen isteği yeniden belirtin veya sistem yöneticisiyle iletişime geçin."
            ),
        )
        return AgentResponse(
            task_id=task.task_id,
            agent_role=AgentRole.COMPOSER,
            draft=response_text,
            status=TaskStatus.FAILED,
            metadata={
                "workflow": {
                    "steps_completed": 0,
                    "steps_attempted": 3,
                    "agents_used": [],
                    "warnings": warnings,
                }
            },
        )

    composed = composer.compose(collected)

    workflow_meta: dict = {
        "steps_completed": len(collected),
        "steps_attempted": 3,
        "agents_used": [s["agent"] for s in collected],
        "warnings": warnings,
        "useful_count":    composed.get("useful_count", 0),
        "merged_count":    composed.get("merged_count", 0),
        "conflict_count":  len(composed.get("conflicts", [])),
    }

    logger.info(
        "agile-workflow: complete  %d/3 steps succeeded  conflicts=%d",
        len(collected),
        workflow_meta["conflict_count"],
    )

    return AgentResponse(
        task_id=task.task_id,
        agent_role=AgentRole.COMPOSER,
        draft=composed["response_text"],
        status=TaskStatus.COMPLETED,
        metadata={"workflow": workflow_meta},
    )
