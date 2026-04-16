"""pipeline/composer_runner.py — Composer pipeline path.

Collects outputs from multiple agents and produces one structured response.
This path is deterministic and does not call an LLM.
"""

from __future__ import annotations

from typing import Any

from src.agents.composer_agent import ComposerAgent
from src.core.schemas import AgentTask, AgentResponse, AgentRole, TaskStatus


def _extract_agent_outputs(task: AgentTask) -> list[Any]:
    """Extract candidate agent outputs from task context/metadata."""
    # Preferred: explicit payload supplied by callers/tests.
    for key in ("agent_outputs", "outputs"):
        value = task.context.get(key)
        if isinstance(value, list):
            return value

    for key in ("agent_outputs", "outputs"):
        value = task.metadata.get(key)
        if isinstance(value, list):
            return value

    # Fallback: infer from chat history (assistant messages).
    messages = task.metadata.get("messages")
    if isinstance(messages, list):
        inferred: list[dict[str, str]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "assistant":
                continue
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            inferred.append({"agent": "AssistantMessage", "draft": content})
        return inferred

    # Last fallback: treat masked_input itself as a single output if present.
    if task.masked_input.strip():
        return [{"agent": "UserProvidedOutput", "draft": task.masked_input}]

    return []


def run_composer_pipeline(task: AgentTask) -> AgentResponse:
    """Run deterministic composition and return AgentResponse."""
    composer = ComposerAgent()
    raw_outputs = _extract_agent_outputs(task)
    composed = composer.compose(raw_outputs)

    return AgentResponse(
        task_id=task.task_id,
        agent_role=AgentRole.COMPOSER,
        draft=composed["response_text"],
        status=TaskStatus.COMPLETED,
        metadata={
            "composer": {
                "input_count": len(raw_outputs),
                "useful_count": composed["useful_count"],
                "merged_count": composed["merged_count"],
                "conflict_count": len(composed["conflicts"]),
            }
        },
    )

