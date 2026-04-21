"""pipeline/composer_runner.py — Composer pipeline path.

Two dispatch modes
──────────────────

SEQUENTIAL WORKFLOW MODE  (agile_workflow.py)
─────────────────────────────────────────────
Triggered when is_workflow_request(query) returns True.
Activates the full PO → SM → Developer → Composer chain where each agent
receives the PREVIOUS agent's output as enriched context.

Use this for:
  • New project requests ("yeni proje başlat", "feature request")
  • End-to-end sprint planning ("sprint planla ve uygula")
  • Full agile lifecycle queries ("uçtan uca", "tam süreç")

PARALLEL DISPATCH MODE  (this file)
────────────────────────────────────
For targeted single-agent queries (e.g. "sprint durumu nedir?",
"blocker'lar neler?").  Scores the query against SM / Developer / PO
vocabulary, selects one or more roles, and dispatches the SAME original
query to each.  Composer merges the outputs.

Routing decision is made once at the top of run_composer_pipeline().
All other logic below is unchanged.
"""

from __future__ import annotations

import logging
import re

from src.agents.composer_agent import ComposerAgent
from src.core.schemas import AgentTask, AgentResponse, AgentRole, TaskStatus
from src.pipeline.developer_runner import _process_developer_message
from src.pipeline.product_owner_runner import run_product_owner_pipeline
from src.pipeline.scrum_master_runner import run_scrum_master_pipeline
from src.pipeline.agile_workflow import is_workflow_request, run_agile_workflow

logger = logging.getLogger(__name__)


_MUTATION_RE = re.compile(
    r"\b(add|update|assign|accept|reject|delete|change|"
    r"ekle|güncelle|guncelle|ata|kabul|reddet|sil|değiştir|degistir)\b",
    re.I,
)

_READ_RE = re.compile(
    r"\b(status|summary|report|overview|list|show|analyze|analysis|compare|"
    r"durum|özet|ozet|rapor|liste|göster|goster|analiz|karşılaştır|karsilastir)\b",
    re.I,
)

_PLACEHOLDER_OUTPUTS = {"ok", "done", "none", "n/a", "null"}


def _extract_response_text(response: AgentResponse) -> str | None:
    """Read candidate text from draft/text/output fields defensively."""
    if response is None:
        return None

    # Primary contract field on AgentResponse.
    draft = getattr(response, "draft", None)
    if draft is not None:
        return str(draft)

    # Defensive fallbacks for mocked/legacy shapes.
    text = getattr(response, "text", None)
    if text is not None:
        return str(text)

    output = getattr(response, "output", None)
    if output is not None:
        return str(output)

    if isinstance(response, dict):
        for key in ("draft", "text", "output"):
            value = response.get(key)
            if value is not None:
                return str(value)
    return None


def _is_usable_output(text: str | None) -> bool:
    if text is None:
        return False
    cleaned = str(text).strip()
    if not cleaned:
        return False
    if cleaned.isspace():
        return False
    if cleaned.lower() in {"none", "null", "n/a"}:
        return False
    if len(cleaned) < 10:
        return False
    normalized = cleaned.lower().strip(" .,!?:;")
    if normalized in _PLACEHOLDER_OUTPUTS:
        return False
    return True


def _term_in_text(text: str, term: str) -> bool:
    """Match terms safely (word-boundary for single tokens)."""
    if " " in term:
        return term in text
    return re.search(rf"\b{re.escape(term)}\b", text, re.I) is not None


def _has_strong_multi_role_signal(query: str) -> bool:
    text = (query or "").lower()
    scrum_process_terms = (
        "sprint", "standup", "retro", "retrospective", "blocker",
        "engel", "daily", "task", "görev",
    )
    po_backlog_terms = (
        "backlog", "story", "acceptance", "criteria",
        "hikaye", "kabul", "öncelik", "oncelik",
    )
    scrum_hit = any(_term_in_text(text, term) for term in scrum_process_terms)
    po_hit = any(_term_in_text(text, term) for term in po_backlog_terms)
    return scrum_hit and po_hit


def _score_roles(query: str) -> dict[AgentRole, int]:
    text = (query or "").lower()
    scores: dict[AgentRole, int] = {
        AgentRole.SCRUM_MASTER: 0,
        AgentRole.DEVELOPER: 0,
        AgentRole.PRODUCT_OWNER: 0,
    }
    scrum_terms = (
        "sprint", "standup", "retro", "retrospective", "blocker",
        "engel", "assigned", "ata", "task", "görev", "daily", "review",
    )
    dev_terms = (
        "code", "bug", "endpoint", "api", "traceback", "exception",
        "refactor", "test", "pipeline", "repo", "architecture", "debug",
        "kod", "hata",
    )
    po_terms = (
        "backlog", "story", "acceptance", "criteria", "priority",
        "product owner", "po", "hikaye", "kabul", "öncelik", "oncelik",
    )

    for token in scrum_terms:
        if _term_in_text(text, token):
            scores[AgentRole.SCRUM_MASTER] += 1
    for token in dev_terms:
        if _term_in_text(text, token):
            scores[AgentRole.DEVELOPER] += 1
    for token in po_terms:
        if _term_in_text(text, token):
            scores[AgentRole.PRODUCT_OWNER] += 1

    # Explicit IDs provide stronger hints.
    if re.search(r"\bS-\d+\b", query, re.I):
        scores[AgentRole.PRODUCT_OWNER] += 2
    if re.search(r"\bT-\d+\b", query, re.I):
        scores[AgentRole.SCRUM_MASTER] += 1
    return scores


def _select_roles(query: str) -> tuple[list[AgentRole], str]:
    scores = _score_roles(query)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_score = ordered[0][1]
    if top_score <= 0:
        return [], "no_reliable_signal"

    is_mutation = bool(_MUTATION_RE.search(query or ""))
    is_read_like = bool(_READ_RE.search(query or ""))

    top_roles = [role for role, score in ordered if score == top_score]
    non_zero_roles = [role for role, score in ordered if score > 0]

    # Mutation queries must dispatch to exactly one role.
    if is_mutation:
        return [top_roles[0]], "single_mutation"

    # Multi-role only for read-like queries with strong explicit evidence
    # across multiple role domains (e.g. sprint/process + backlog/acceptance).
    if is_read_like and len(non_zero_roles) > 1 and _has_strong_multi_role_signal(query):
        return non_zero_roles, "multi_read_like"

    return [top_roles[0]], "single_best_match"


def _dispatch_to_runner(task: AgentTask, role: AgentRole) -> AgentResponse:
    child = AgentTask(
        session_id=task.session_id,
        role=role,
        masked_input=task.masked_input,
        parent_task_id=task.task_id,
        metadata={**task.metadata},
    )
    if role == AgentRole.DEVELOPER:
        child.metadata["strategy"] = child.metadata.get("strategy", "auto")
        return _process_developer_message(child)
    if role == AgentRole.SCRUM_MASTER:
        return run_scrum_master_pipeline(child)
    if role == AgentRole.PRODUCT_OWNER:
        return run_product_owner_pipeline(child)
    return AgentResponse(
        task_id=task.task_id,
        agent_role=AgentRole.COMPOSER,
        draft="Unsupported role dispatch.",
        status=TaskStatus.FAILED,
    )


def _safe_insufficient_response(reason: str) -> dict[str, object]:
    composer = ComposerAgent()
    text = composer.format_final_response(
        summary="Insufficient evidence to select an agent role.",
        key_points=[
            "No reliable intent signal was found in the request.",
            "No downstream agent was dispatched.",
        ],
        warnings=[f"Dispatch skipped: {reason}"],
        final_answer=(
            "I cannot safely determine whether Scrum Master, Developer, or Product Owner "
            "should answer this query."
        ),
    )
    return {
        "response_text": text,
        "useful_count": 0,
        "merged_count": 0,
        "conflicts": [],
    }


def run_composer_pipeline(task: AgentTask) -> AgentResponse:
    """Run Composer orchestration and return AgentResponse.

    Routing decision (evaluated once, at entry):
      is_workflow_request → sequential PO→SM→Developer chain (agile_workflow)
      otherwise           → parallel single-query dispatch (this module)
    """
    # ── Sequential workflow path ──────────────────────────────────────────────
    if is_workflow_request(task.masked_input):
        logger.info(
            "composer: workflow request detected — delegating to agile_workflow  "
            "task=%s", task.task_id
        )
        return run_agile_workflow(task)

    # ── Parallel dispatch path ────────────────────────────────────────────────
    roles, selection_reason = _select_roles(task.masked_input)
    if not roles:
        composed = _safe_insufficient_response(selection_reason)
        return AgentResponse(
            task_id=task.task_id,
            agent_role=AgentRole.COMPOSER,
            draft=composed["response_text"],
            status=TaskStatus.COMPLETED,
            metadata={
                "composer": {
                    "selection_reason": selection_reason,
                    "dispatched_roles": [],
                    "input_count": 0,
                    "useful_count": 0,
                    "merged_count": 0,
                    "conflict_count": 0,
                }
            },
        )

    dispatched: list[AgentResponse] = [_dispatch_to_runner(task, role) for role in roles]

    runner_outputs: list[dict[str, str]] = []
    for response in dispatched:
        if response.status != TaskStatus.COMPLETED:
            continue
        text = _extract_response_text(response)
        if not _is_usable_output(text):
            continue
        runner_outputs.append(
            {
                "agent": response.agent_role.value,
                "draft": str(text).strip(),
            }
        )

    composer = ComposerAgent()
    if runner_outputs:
        composed = composer.compose(runner_outputs)
    else:
        composed = composer.format_final_response(
            summary="No usable outputs. Selected agents did not produce reliable content.",
            key_points=[
                f"Selected roles: {', '.join(role.value for role in roles)}",
                "All downstream outputs were empty or failed.",
            ],
            warnings=["Insufficient downstream evidence."],
            final_answer="Unable to produce a reliable final answer from downstream agents.",
        )
        composed = {
            "response_text": composed,
            "useful_count": 0,
            "merged_count": 0,
            "conflicts": [],
        }

    return AgentResponse(
        task_id=task.task_id,
        agent_role=AgentRole.COMPOSER,
        draft=composed["response_text"],
        status=TaskStatus.COMPLETED,
        metadata={
            "composer": {
                "selection_reason": selection_reason,
                "dispatched_roles": [role.value for role in roles],
                "input_count": len(runner_outputs),
                "useful_count": composed["useful_count"],
                "merged_count": composed["merged_count"],
                "conflict_count": len(composed["conflicts"]),
            }
        },
    )
