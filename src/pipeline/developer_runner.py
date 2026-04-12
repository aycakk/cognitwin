"""pipeline/developer_runner.py — developer pipeline path.

Extracted from src/services/api/pipeline.py (_process_developer_message
and its private helpers). pipeline.py re-imports _process_developer_message
for use in process_user_message.

Path note: this file lives at src/pipeline/, so Path(__file__).resolve()
.parents[2] resolves to the project root (one fewer hop than pipeline.py
which was at src/services/api/).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ollama import chat

from src.agents.developer_orchestrator import DeveloperOrchestrator
from src.agents.scrum_master_agent import ScrumMasterAgent
from src.core.schemas import AgentTask, AgentResponse, AgentRole, TaskStatus
from src.gates.evaluator import evaluate_all_gates
from src.pipeline.redo import run_redo_loop
from src.pipeline.shared import (
    VECTOR_MEM,
    VECTOR_TOP_K,
    CHROMA,
    SYSTEM_PROMPT,
    BLINDSPOT_TRIGGERS,
    build_blindspot_block,
    _safe_chat,
    _sanitize_output,
)

_SCRUM_AGENT = ScrumMasterAgent()

# ─────────────────────────────────────────────────────────────────────────────
#  CODEBASE CONTEXT INJECTION
# ─────────────────────────────────────────────────────────────────────────────

# keyword → source file(s) to include when that keyword appears in the query
_DEV_CONTEXT_MAP: list[tuple[frozenset, str]] = [
    (frozenset({"routing", "route", "api", "endpoint", "librechat", "openai",
                "completions", "models", "v1"}),
     "src/services/api/openai_routes.py"),
    (frozenset({"pipeline", "gate", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8",
                "redo", "blindspot", "grounding", "architecture", "repository",
                "codebase", "this project", "this repo",
                "student path", "developer path", "both paths",
                "student and developer", "compare path",
                "branch", "branches", "worktree", "merge",
                "debug", "bug", "fix issue", "traceback", "exception",
                "stack trace", "error in"}),
     "src/services/api/pipeline.py"),
    (frozenset({"orchestrator", "developer orchestrator", "8-stage", "8 stage",
                "understand", "footprint", "profile"}),
     "src/agents/developer_orchestrator.py"),
    (frozenset({"developer agent", "role packet", "role library", "blueprint"}),
     "src/agents/developer_agent.py"),
    (frozenset({"student agent", "student path", "student pipeline", "studentagent"}),
     "src/agents/student_agent.py"),
    (frozenset({"chroma", "chromadb", "vector memory", "vector store", "memory backend"}),
     "src/database/chroma_manager.py"),
    (frozenset({"masker", "pii", "mask", "anonymi"}),
     "src/utils/masker.py"),
]

# chars read per file — enough for full small files, first section of large ones
_CODE_SNIPPET_CHARS = 4000


def _build_codebase_context(query: str) -> str:
    """
    Build a source-code context block for developer queries about this repo.

    Matches keywords in the query against a file map, reads the relevant
    source files (truncated), and returns a formatted context string.
    Returns an empty string when no codebase keywords are found so that
    the orchestrator's footprint logic is not bypassed unnecessarily.
    """
    text = query.lower()
    # developer_runner.py lives at src/pipeline/ → parents[2] is project root
    project_root = Path(__file__).resolve().parents[2]

    collected: dict[str, str] = {}   # rel_path → content, insertion-ordered dedup
    for keywords, rel_path in _DEV_CONTEXT_MAP:
        if any(kw in text for kw in keywords):
            if rel_path not in collected:
                abs_path = project_root / rel_path
                if abs_path.exists():
                    raw = abs_path.read_text(encoding="utf-8", errors="replace")
                    collected[rel_path] = raw[:_CODE_SNIPPET_CHARS]

    if not collected:
        return ""

    parts = ["=== CODEBASE CONTEXT (live source files) ==="]
    for rel_path, content in collected.items():
        parts.append(f"\n--- {rel_path} ---")
        parts.append(content)
        if len(content) >= _CODE_SNIPPET_CHARS:
            parts.append("... [truncated]")
    parts.append("\n=== END CODEBASE CONTEXT ===")
    return "\n".join(parts)


def _build_sprint_context() -> str:
    """
    Build a labeled sprint context block from the module-level _SCRUM_AGENT instance.

    Reads sprint goal, active assignments, and blocked tasks from the current
    sprint_state.json.  Returns a formatted string using the same labeled-block
    convention as the rest of the codebase (=== VECTOR MEMORY ===, etc.).

    Returns an empty string only when the sprint state is genuinely absent
    (ScrumMasterAgent._load_state falls back to _DEFAULT_SPRINT_STATE, so in
    practice it always returns at least the default goal and empty task lists).
    """
    try:
        sprint_goal  = _SCRUM_AGENT.get_sprint_goal()
        assignments  = _SCRUM_AGENT.get_current_assignments()
        blocked      = _SCRUM_AGENT.get_blocked_tasks()
    except Exception:
        return ""

    lines = ["=== SPRINT CONTEXT ==="]
    lines.append(f"Sprint Goal: {sprint_goal}")

    lines.append("Active Assignments:")
    if assignments:
        for t in assignments:
            lines.append(
                f"  [{t.get('id', '?')}] {t.get('title', '-')}"
                f" \u2192 {t.get('assignee', 'unassigned')}"
                f" ({t.get('status', '?')})"
            )
    else:
        lines.append("  (none)")

    lines.append("Blocked Tasks:")
    if blocked:
        for t in blocked:
            blocker_note = t.get("blocker") or "no description"
            lines.append(
                f"  [{t.get('id', '?')}] {t.get('title', '-')}"
                f" | Blocker: {blocker_note}"
            )
    else:
        lines.append("  (none)")

    lines.append("=== END SPRINT CONTEXT ===")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  DEBUG RESULT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def _validate_debug_result(parsed: dict, project_root: Path) -> dict:
    """
    Machine-verify the files and functions reported by the LLM.

    For every path in files_used: check the file exists under project_root.
    For every name in functions_used: check it appears as an identifier in
    the source of the cited files (simple text search — not AST).

    Returns the input dict extended with a '_validation' key and, if any
    files were invented, sets speculative=True.
    """
    files_used     = [str(f) for f in (parsed.get("files_used") or []) if f]
    functions_used = [str(f) for f in (parsed.get("functions_used") or []) if f]

    invalid_files     : list[str] = []
    unverified_funcs  : list[str] = []

    for rel_path in files_used:
        if not (project_root / rel_path).exists():
            invalid_files.append(rel_path)

    valid_files = [f for f in files_used if (project_root / f).exists()]
    for func_name in functions_used:
        found_in_any = False
        pattern = re.compile(rf"\b{re.escape(func_name)}\b")
        for rel_path in valid_files:
            try:
                content = (project_root / rel_path).read_text(
                    encoding="utf-8", errors="replace"
                )
                if pattern.search(content):
                    found_in_any = True
                    break
            except Exception:
                pass
        if not found_in_any:
            unverified_funcs.append(func_name)

    all_verified = not invalid_files and not unverified_funcs
    result = dict(parsed)
    result["_validation"] = {
        "invalid_files":    invalid_files,
        "unverified_funcs": unverified_funcs,
        "files_ok":         len(invalid_files) == 0,
        "functions_ok":     len(unverified_funcs) == 0,
        "all_verified":     all_verified,
    }
    if invalid_files:
        result["speculative"] = True   # invented paths → force speculative flag
    return result


def _format_debug_result(v: dict) -> str:
    """
    Render a validated structured debug dict as clean, readable text.
    Marks any invented/unverified items explicitly so the user sees them.
    """
    val   = v.get("_validation", {})
    lines = []

    if v.get("entry_point"):
        lines.append(f"Entry Point: {v['entry_point']}")

    if v.get("files_used"):
        lines.append("\nFiles:")
        bad = set(val.get("invalid_files") or [])
        for f in v["files_used"]:
            tag = "  [NOT FOUND IN REPO]" if f in bad else ""
            lines.append(f"  - {f}{tag}")

    if v.get("functions_used"):
        lines.append("\nFunctions:")
        unv = set(val.get("unverified_funcs") or [])
        for fn in v["functions_used"]:
            tag = "  [NOT VERIFIED IN FILES]" if fn in unv else ""
            lines.append(f"  - {fn}{tag}")

    if v.get("execution_path"):
        lines.append("\nExecution Path:")
        for i, step in enumerate(v["execution_path"], 1):
            lines.append(f"  {i}. {step}")

    if v.get("suspected_root_cause"):
        lines.append(f"\nRoot Cause: {v['suspected_root_cause']}")

    if v.get("evidence"):
        lines.append("\nEvidence:")
        for e in v["evidence"]:
            lines.append(f"  - {e}")

    if v.get("fix"):
        lines.append(f"\nFix: {v['fix']}")

    conf = float(v.get("confidence") or 0.0)
    spec = bool(v.get("speculative", False))
    spec_tag = " (SPECULATIVE)" if spec else ""
    lines.append(f"\nConfidence: {conf:.2f}{spec_tag}")

    if not val.get("all_verified", True):
        lines.append(
            "\n[Validation] Some reported files or functions could not be "
            "verified in the repository. Treat this analysis with caution."
        )
    else:
        lines.append("\n[Validation] All reported files and functions verified in repository.")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  DEVELOPER PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def _process_developer_message(task: AgentTask) -> AgentResponse:
    """
    Developer path: DeveloperOrchestrator (8-stage) → C1-C8 gates → REDO.

    The orchestrator handles context retrieval, ontology constraints, profile
    personalisation, and generation.  After the orchestrator returns its
    solution, the same ZT4SWE gate array and REDO loop used by the student
    path are applied to guarantee output integrity.
    """
    user_text    = task.masked_input
    strategy     = task.metadata.get("strategy", "auto")
    developer_id = task.metadata.get("developer_id", "developer-default")
    session_id   = task.session_id

    redo_log: list[dict] = []

    # Stage 1 — Retrieve vector context (used only by gate evaluators).
    # namespace="developer" keeps developer codebase snippets out of the
    # student academic_memory collection.
    vector_context, is_empty = VECTOR_MEM.retrieve(user_text, k=VECTOR_TOP_K, namespace="developer")

    # Stage 2 — DeveloperOrchestrator 8-stage pipeline
    # Inject actual source file snippets so the LLM can answer codebase
    # questions rather than falling back to generic training knowledge.
    codebase_context = _build_codebase_context(user_text)

    # Stage 2a — Sprint context injection (Scrum → Developer read path)
    # Read the current sprint state through the module-level _SCRUM_AGENT instance and
    # format it as a labeled context block.  Combined with codebase_context so
    # the orchestrator LLM is aware of active tasks, the sprint goal, and any
    # blockers when generating its response.
    # NOTE: codebase_context is kept as a separate variable — Stage 2b below
    # gates JSON debug validation on it; that gate must not fire for sprint-only
    # queries where codebase_context is empty.
    sprint_context = _build_sprint_context()
    combined_context = "\n\n".join(filter(None, [codebase_context, sprint_context]))

    orchestrator = DeveloperOrchestrator(
        memory_backend=CHROMA,
        chat_fn=_safe_chat,
        default_model="llama3.2",
    )
    result = orchestrator.run(
        request=user_text,
        developer_id=developer_id,
        strategy=strategy,
        memory_context=combined_context,   # non-empty → used directly in generate()
    )
    draft = _sanitize_output(str(result.get("solution", "")))
    if not draft:
        draft = "Bunu hafızamda bulamadım."

    # Developer → Scrum write-back
    # If the developer's query signals a task status change (update_task intent),
    # update sprint_state.json as a side effect before continuing.
    # Only "update_task" triggers a write-back; other intents (sprint_status,
    # standup, review, …) belong to the Scrum path and must not cause writes here.
    # The return value of handle_query is discarded — the developer response is
    # always the primary output.  Wrapped in try/except so a state I/O error
    # never surfaces to the end user.
    if _SCRUM_AGENT.detect_intent(user_text) == "update_task":
        try:
            _SCRUM_AGENT.handle_query(user_text)
        except Exception:
            pass

    # Stage 2b — Structured debug validation
    # If the orchestrator returned a JSON debug object, validate the cited
    # files and functions against the actual repository before emitting.
    if codebase_context:
        try:
            parsed = json.loads(draft)
            if isinstance(parsed, dict) and "files_used" in parsed:
                # developer_runner.py lives at src/pipeline/ → parents[2] is project root
                project_root = Path(__file__).resolve().parents[2]
                validated    = _validate_debug_result(parsed, project_root)
                draft        = _format_debug_result(validated)
        except (json.JSONDecodeError, ValueError):
            pass  # prose response — skip validation, continue normally

    # Stage 3 — C1-C8 gate array + REDO loop (same as student path)
    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ]

    draft, limit_hit = run_redo_loop(
        draft, base_messages, vector_context, is_empty, redo_log,
        agent_role="DeveloperAgent",
        query=user_text,
        redo_rules=(
            "Answer ONLY from verified developer context. "
            "If not found: \"Bunu hafizamda bulamadim.\""
        ),
        limit_message_template=(
            "Dogrulama basarisiz (Gate {gate}). "
            "Yanit guvenli bicimde teslim edilemiyor.\n"
            "Bunu hafizamda bulamadim."
        ),
        post_process=_sanitize_output,
        gate_fn=evaluate_all_gates,
        chat_fn=chat,
        blindspot_fn=build_blindspot_block,
        session_id=session_id,
    )
    if limit_hit:
        return AgentResponse(
            task_id=task.task_id,
            agent_role=AgentRole.DEVELOPER,
            draft=draft,
            status=TaskStatus.FAILED,
            redo_log=redo_log,
        )

    # Stage 4 — Emission
    if BLINDSPOT_TRIGGERS.search(draft):
        draft = build_blindspot_block(user_text) + draft

    return AgentResponse(
        task_id=task.task_id,
        agent_role=AgentRole.DEVELOPER,
        draft=draft,
        status=TaskStatus.COMPLETED,
        redo_log=redo_log,
    )
