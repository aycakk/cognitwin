"""governance/policy.py — gate assignment policy.

Single source of truth for which gates run for which agent role.

Rules
-----
- C1 (PII masking) is intentionally absent from every role's gate list here.
  It is applied at the API Gateway layer (openai_routes.py / pipeline.py)
  before any agent sees the input. It is not a post-generation quality gate.

- C2 (memory grounding) does not apply to DeveloperAgent because the developer
  path uses injected codebase/sprint context, not student academic memory.
  Removing it from the DeveloperAgent list replaces the old `exempt=True` flag
  that was hardcoded inside the gate evaluator.

- ScrumMasterAgent runs only C4 (hallucination) because it produces
  deterministic rule-based output.  All other content gates are irrelevant
  for a non-LLM agent.

Adding a new gate
-----------------
Add its key to the relevant roles below. The evaluator reads this table
at call-time; no other file needs to change.

Adding a new role
-----------------
Add an entry to GATE_POLICY.  The evaluator will handle it automatically.
"""

from __future__ import annotations

# role identifier → ordered list of gate IDs to evaluate
GATE_POLICY: dict[str, list[str]] = {
    "StudentAgent": [
        "C2",   # memory grounding (academic namespace)
        "C3",   # ontology compliance
        "C4",   # hallucination markers
        "C5",   # role-permission boundary
        "C6",   # anti-sycophancy
        "C7",   # blindspot completeness
        "A1",   # REDO cycle audit
    ],
    "InstructorAgent": [
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "A1",
    ],
    "ResearcherAgent": [
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "A1",
    ],
    "DeveloperAgent": [
        # C2 (academic memory grounding) intentionally omitted.
        # C2_DEV is the developer-specific variant: checks grounding
        # against injected codebase + sprint context instead.
        "C2_DEV",  # developer codebase/sprint grounding
        "C3",      # ontology compliance still applies
        "C4",      # hallucination markers
        "C5",      # role-permission boundary
        "A1",      # REDO cycle audit
    ],
    "ScrumMasterAgent": [
        # Rule-based deterministic agent — only hallucination guard needed.
        "C4",
    ],
}

# Default gate list for any role not explicitly listed above.
# Keeps the system safe when a new role is added before its policy entry.
DEFAULT_GATE_POLICY: list[str] = ["C4", "A1"]
