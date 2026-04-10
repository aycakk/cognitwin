"""core/exceptions.py — project-wide exception hierarchy.

All COGNITWIN-specific exceptions inherit from CogniTwinError so callers
can catch the entire family with a single except clause when needed.
"""

from __future__ import annotations


class CogniTwinError(Exception):
    """Base class for all COGNITWIN application errors."""


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

class UnknownRoleError(CogniTwinError, ValueError):
    """Raised when a model name cannot be resolved to a known AgentRole."""


# ---------------------------------------------------------------------------
# Ontology
# ---------------------------------------------------------------------------

class OntologyLoadError(CogniTwinError, RuntimeError):
    """Raised when the ontology is required but cannot be loaded."""


class OntologyUnavailableError(CogniTwinError, RuntimeError):
    """Raised at gate evaluation time when the graph is None and required."""


# ---------------------------------------------------------------------------
# Gate / Policy
# ---------------------------------------------------------------------------

class GateFailureError(CogniTwinError, RuntimeError):
    """Raised when a gate fails and no REDO budget remains."""

    def __init__(self, gate_id: str, evidence: str) -> None:
        super().__init__(f"Gate {gate_id} failed: {evidence}")
        self.gate_id  = gate_id
        self.evidence = evidence


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

class OrchestratorDepthError(CogniTwinError, RecursionError):
    """Raised when agent sub-task dispatch exceeds the maximum nesting depth."""
