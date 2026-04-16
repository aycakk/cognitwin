"""core/schemas.py — typed contracts for agent-to-agent communication.

Every layer that dispatches work to an agent or receives results from one
uses these dataclasses.  No agent, runner, or gate module should accept or
return bare dicts or strings at its public boundary — use these instead.

Design decisions
----------------
* Dataclasses (not Pydantic) to keep the dependency footprint minimal.
* AgentRole is a str enum so values can be compared to the string role
  identifiers already used throughout the codebase (GATE_POLICY, permissions).
* sub_tasks on AgentResponse enables the orchestrator's future delegation
  loop: an agent that needs data from another agent emits a sub-task rather
  than importing the other agent directly.
* parent_task_id on AgentTask lets the orchestrator stitch sub-task results
  back to the originating task.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class AgentRole(str, Enum):
    STUDENT       = "StudentAgent"
    DEVELOPER     = "DeveloperAgent"
    SCRUM_MASTER  = "ScrumMasterAgent"
    PRODUCT_OWNER = "ProductOwnerAgent"
    COMPOSER      = "ComposerAgent"

    # Allow comparison with legacy plain-string role identifiers
    # e.g.  AgentRole.STUDENT == "StudentAgent"  → True
    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.value)


class TaskStatus(str, Enum):
    PENDING   = "pending"
    COMPLETED = "completed"
    FAILED    = "failed"
    REDO      = "redo"


@dataclass
class AgentTask:
    """Input contract — what an agent receives from the dispatcher."""

    # Unique identifier for this task (auto-generated if not supplied).
    task_id:        str                  = field(default_factory=lambda: str(uuid.uuid4()))

    # Session that owns this task (set at the API gateway layer).
    session_id:     str                  = ""

    # Which agent should handle this task.
    role:           AgentRole            = AgentRole.STUDENT

    # The user's request, already PII-masked.
    masked_input:   str                  = ""

    # Classified intent label (e.g. "sprint_status", "code_review").
    intent:         str                  = ""

    # Arbitrary context dict (codebase snippets, sprint state, etc.)
    context:        dict[str, Any]       = field(default_factory=dict)

    # Set when this task was spawned by another agent (sub-task pattern).
    parent_task_id: Optional[str]        = None

    # Arbitrary metadata (language, developer_id, strategy, …)
    metadata:       dict[str, Any]       = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Output contract — what an agent returns to the dispatcher."""

    # Echo of the task_id this response satisfies.
    task_id:     str                  = ""

    # Which agent produced this response.
    agent_role:  AgentRole            = AgentRole.STUDENT

    # The generated (and gate-validated) text.
    draft:       str                  = ""

    # Terminal status after gate evaluation + REDO.
    status:      TaskStatus           = TaskStatus.PENDING

    # Gate evaluation results keyed by gate ID.
    gate_results: dict[str, Any]      = field(default_factory=dict)

    # REDO audit entries for this request.
    redo_log:    list[dict]           = field(default_factory=list)

    # Source citations (vector memory doc IDs or ontology triple refs).
    citations:   list[str]            = field(default_factory=list)

    # Sub-tasks emitted by this agent for the orchestrator to dispatch.
    # When non-empty, the orchestrator dispatches them and re-calls the
    # agent with the results injected into AgentTask.context.
    sub_tasks:   list[AgentTask]      = field(default_factory=list)

    # Arbitrary metadata (timing, model used, …)
    metadata:    dict[str, Any]       = field(default_factory=dict)
