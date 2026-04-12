"""pipeline/student_runner.py — 4-stage ZT4SWE student pipeline.

Extracted from src/services/api/pipeline.py (run_pipeline).
pipeline.py re-imports and re-exports run_pipeline for backward compat.
"""

from __future__ import annotations

from ollama import chat

from src.core.schemas import AgentTask, AgentResponse, AgentRole, TaskStatus
from src.gates.evaluator import evaluate_all_gates
from src.pipeline.redo import run_redo_loop
from src.pipeline.shared import (
    VECTOR_MEM,
    VECTOR_TOP_K,
    SYSTEM_PROMPT,
    BLINDSPOT_TRIGGERS,
    build_ontology_context,
    build_blindspot_block,
)


def run_pipeline(task: AgentTask) -> AgentResponse:
    """
    Execute the 4-stage ZT4SWE verification pipeline.

    Stage 1 — Retrieval & Grounding  : ChromaDB (k=15) + ontology context
    Stage 2 — Draft Synthesis        : LLM via Ollama llama3.2
    Stage 3 — Compliance Verification: C1–C8 gate array + REDO loop (max 2)
    Stage 4 — Emission               : BlindSpot prepended if needed

    redo_log is per-request (thread-safe — no shared mutable state).
    """
    query      = task.masked_input
    agent_role = task.role.value
    session_id = task.session_id

    redo_log: list[dict] = []

    # ── Stage 1 — Retrieval & Grounding ──────────────────────────────────────
    # namespace="academic" isolates student queries from developer/agile data.
    vector_context, is_empty = VECTOR_MEM.retrieve(query, k=VECTOR_TOP_K, namespace="academic")
    ontology_context         = build_ontology_context()

    if is_empty:
        draft = (
            build_blindspot_block(query, "VEKTÖR HAFIZA BOŞ")
            + "Bunu hafızamda bulamadım."
        )
        return AgentResponse(
            task_id=task.task_id,
            agent_role=task.role,
            draft=draft,
            status=TaskStatus.COMPLETED,
        )

    # ── Stage 2 — Draft Synthesis ─────────────────────────────────────────────
    user_message = (
        f"{vector_context}\n\n"
        f"{ontology_context}\n\n"
        f"ROLE: {agent_role}\n\n"
        f"SORU: {query}\n\n"
        "INSTRUCTION: Answer ONLY from the VECTOR MEMORY and ONTOLOGY CONTEXT "
        "above, in Turkish. "
        "Combine VECTOR MEMORY (dates, grades, records) with ONTOLOGY "
        "(course names, exam→course links, agent roles) for a complete answer. "
        "If the answer is NOT in either source: \"Bunu hafızamda bulamadım.\" "
        "Do NOT hallucinate. Do NOT unmask PII tokens. "
        "Prefix ontology-only answers with \"Akademik yapıya göre…\""
    )
    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    resp  = chat(model="llama3.2", messages=base_messages)
    draft = resp.message.content.strip()

    # ── Stage 3 — Compliance Verification ────────────────────────────────────
    draft, limit_hit = run_redo_loop(
        draft, base_messages, vector_context, is_empty, redo_log,
        agent_role=agent_role,
        query=query,
        redo_rules=(
            "Answer ONLY from VECTOR MEMORY and ONTOLOGY CONTEXT. "
            "If not found: \"Bunu hafızamda bulamadım.\""
        ),
        limit_message_template=(
            "⚠ Doğrulama başarısız (Gate {gate}). "
            "Yanıt güvenli biçimde teslim edilemiyor.\n"
            "Bunu hafızamda bulamadım."
        ),
        post_process=lambda s: s,
        gate_fn=evaluate_all_gates,
        chat_fn=chat,
        blindspot_fn=build_blindspot_block,
        session_id=session_id,
    )
    if limit_hit:
        return AgentResponse(
            task_id=task.task_id,
            agent_role=task.role,
            draft=draft,
            status=TaskStatus.FAILED,
            redo_log=redo_log,
        )

    # ── Stage 4 — Emission ────────────────────────────────────────────────────
    # Only prepend a blindspot block when retrieval was genuinely empty.
    # Checking draft text alone is wrong: the LLM may say "Bunu hafızamda
    # bulamadım" for a sub-topic it couldn't resolve even though memory docs
    # *were* retrieved, which should not produce the blindspot wrapper.
    if is_empty and BLINDSPOT_TRIGGERS.search(draft):
        draft = build_blindspot_block(query) + draft

    return AgentResponse(
        task_id=task.task_id,
        agent_role=task.role,
        draft=draft,
        status=TaskStatus.COMPLETED,
        redo_log=redo_log,
    )
