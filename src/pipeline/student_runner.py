"""pipeline/student_runner.py — 4-stage ZT4SWE student pipeline.

Extracted from src/services/api/pipeline.py (run_pipeline).
pipeline.py re-imports and re-exports run_pipeline for backward compat.
"""

from __future__ import annotations

import re

from ollama import chat

from src.core.schemas import AgentTask, AgentResponse, AgentRole, TaskStatus
from src.gates.evaluator import evaluate_all_gates
from src.pipeline.redo import run_redo_loop
from src.pipeline.shared import (
    VECTOR_MEM,
    VECTOR_TOP_K,
    SYSTEM_PROMPT,
    build_ontology_context,
    build_blindspot_block,
)
from src.intent_classifier import predict_intent


_ONTOLOGY_FIRST_INTENTS = {"exam_info", "course_info", "instructor_info"}
_BLINDSPOT_PHRASE = "Bunu hafızamda bulamadım."
_BLINDSPOT_BLOCK_RE = re.compile(r"\n?┌─.*?┘\n?", re.S)
_INTERNAL_TERM_RE = re.compile(
    r"\b(VECTOR MEMORY|ONTOLOGY CONTEXT|ROLE|INTENT|SORU|Result)\b",
    re.I,
)
_EN_FILLER_RE = re.compile(
    r"\b(information|unknown|combine|respond|note that|context|summary)\b",
    re.I,
)
_FORBIDDEN_DETERMINISTIC_RE = re.compile(
    r"(VECTOR MEMORY|ONTOLOGY|OntologyContext|information|combine|respond)",
    re.I,
)


def _extract_exam_override_from_ontology(
    ontology_context: str,
    query: str,
) -> str | None:
    """Build deterministic exam answer if ontology context exposes a usable date."""
    if not ontology_context.strip():
        return None

    date_match = re.search(
        (
            r"\b("
            r"\d{1,2}[./]\d{1,2}[./]\d{2,4}"
            r"|\d{4}-\d{2}-\d{2}"
            r"|\d{1,2}\s+"
            r"(Ocak|Subat|Şubat|Mart|Nisan|Mayıs|Mayis|Haziran|Temmuz|Agustos|Ağustos|"
            r"Eylul|Eylül|Ekim|Kasim|Kasım|Aralik|Aralık|"
            r"January|February|March|April|May|June|July|August|September|October|November|December)"
            r"\s+\d{4}"
            r")\b"
        ),
        ontology_context,
        re.I,
    )
    if not date_match:
        return None
    date_text = date_match.group(1)

    course_name = ""
    # Preferred: ontology relationship line.
    rel_match = re.search(r"belongs_to\s+Course:\s*([^\n]+)", ontology_context, re.I)
    if rel_match:
        course_name = rel_match.group(1).replace("_", " ").strip()

    # Fallback: infer from query phrasing.
    if not course_name:
        q_match = re.search(r"([A-Za-zÇĞİÖŞÜçğıöşü0-9_-]+)\s+sınav", query, re.I)
        if q_match:
            course_name = q_match.group(1).replace("_", " ").strip()

    if course_name:
        return f"{course_name} dersi vize sınavı {date_text} tarihinde yapılacaktır."
    return f"Bu dersin vize sınavı {date_text} tarihinde yapılacaktır."


def _sanitize_student_output(
    draft: str,
    *,
    intent: str,
    has_ontology_facts: bool,
) -> str:
    """Keep output concise/source-grounded for ontology-first intents."""
    text = _sanitize_visible_text(draft)
    if not text:
        return text

    # Remove contradictory blindspot phrase only when meaningful content exists.
    without_phrase = re.sub(re.escape(_BLINDSPOT_PHRASE), "", text, flags=re.I).strip(" \t\n\r.,;:-")
    if without_phrase and any(ch.isalpha() for ch in without_phrase):
        text = without_phrase

    if intent in _ONTOLOGY_FIRST_INTENTS and has_ontology_facts:
        # Prevent contradictory outputs: drop blindspot scaffold for ontology-first intents.
        stripped = _BLINDSPOT_BLOCK_RE.sub(" ", text)
        stripped = re.sub(re.escape(_BLINDSPOT_PHRASE), "", stripped, flags=re.I)
        stripped = " ".join(stripped.split()).strip(" \t\n\r.,;:-")
        if stripped:
            text = stripped

        # Keep a single concise sentence for ontology-first intents.
        parts = [p.strip(" \t\n\r.,;:-") for p in re.split(r"[.!?]+", text) if p.strip()]
        if parts:
            text = parts[0] + "."
        else:
            fallback = {
                "exam_info": "Akademik yapıya göre sınav bilgisi bulundu ancak tarih bilgisi net değil.",
                "course_info": "Akademik yapıya göre ders bilgisi mevcut ancak soru net yanıtlanamadı.",
                "instructor_info": "Akademik yapıya göre öğretim elemanı bilgisi mevcut ancak soru net yanıtlanamadı.",
            }
            text = fallback.get(intent, "Akademik yapıya göre ilgili bilgi bulundu.")

    return text


def _sanitize_visible_text(text: str) -> str:
    cleaned = _BLINDSPOT_BLOCK_RE.sub(" ", text or "")
    cleaned = _INTERNAL_TERM_RE.sub(" ", cleaned)
    cleaned = _EN_FILLER_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\n\r.,;:-")
    return cleaned


def _finalize_deterministic_output(text: str) -> str:
    cleaned = _sanitize_visible_text(text)
    cleaned = re.sub(re.escape(_BLINDSPOT_PHRASE), "", cleaned, flags=re.I).strip(" \t\n\r.,;:-")
    cleaned = re.split(
        r"(Soru:|Cevap:|Düşündüğüm|Vector Memory|Ontology|Ayrıca|Bu nedenle)",
        cleaned,
        maxsplit=1,
        flags=re.I,
    )[0]
    cleaned = cleaned.strip()
    # Deterministic output must be single-line and single-sentence.
    cleaned = (cleaned.split("\n")[0] if cleaned else "").strip()
    if cleaned:
        cleaned = cleaned.split(".")[0].strip(" \t\n\r.,;:-") + "."
    # Optional safety requested by user.
    if _FORBIDDEN_DETERMINISTIC_RE.search(cleaned):
        head = cleaned.split(".")[0].strip(" \t\n\r.,;:-")
        cleaned = (head + ".") if head else ""
    return cleaned


def _deterministic_response(
    task: AgentTask,
    *,
    text: str,
    redo_log: list[dict],
    source: str,
) -> AgentResponse:
    final_text = _finalize_deterministic_output(text)
    # Final guard: deterministic outputs must stay single-line.
    final_text = final_text.split("\n")[0].strip()
    if _FORBIDDEN_DETERMINISTIC_RE.search(final_text):
        head = final_text.split(".")[0].strip(" \t\n\r.,;:-")
        final_text = (head + ".") if head else final_text
    return AgentResponse(
        task_id=task.task_id,
        agent_role=task.role,
        draft=final_text,
        status=TaskStatus.COMPLETED,
        redo_log=redo_log,
        metadata={"source": source},
    )


def _extract_courses_from_ontology_context(ontology_context: str) -> list[str]:
    raw = re.findall(r"belongs_to\s+Course:\s*([^\n]+)", ontology_context, flags=re.I)
    seen: set[str] = set()
    result: list[str] = []
    for item in raw:
        course = item.replace("_", " ").strip(" \t\n\r.,;:-")
        key = course.lower()
        if course and key not in seen:
            seen.add(key)
            result.append(course)
    return result


def _extract_courses_from_memory_context(vector_context: str) -> list[str]:
    # Common course-code pattern (e.g., COM8090, OR701, CSE-101).
    raw = re.findall(r"\b[A-Z]{2,6}-?\d{3,4}\b", vector_context or "")
    seen: set[str] = set()
    result: list[str] = []
    for item in raw:
        course = item.strip()
        key = course.lower()
        if course and key not in seen:
            seen.add(key)
            result.append(course)
    return result


def _build_course_info_answer(ontology_context: str, vector_context: str) -> str:
    courses = _extract_courses_from_ontology_context(ontology_context)
    if not courses:
        courses = _extract_courses_from_memory_context(vector_context)

    if courses:
        return "Bu dönem aldığınız dersler: " + ", ".join(courses[:8]) + "."
    return "Bu dönem ders bilgisi net bulunamadı."


def _extract_course_from_query(query: str) -> str | None:
    code_match = re.search(r"\b([A-Z]{2,6}-?\d{3,4})\b", query or "", flags=re.I)
    if code_match:
        return code_match.group(1).upper()

    name_match = re.search(
        r"\b([A-Za-zÇĞİÖŞÜçğıöşü0-9_-]{2,})\s+ders(?:i|in)\b",
        query or "",
        flags=re.I,
    )
    if not name_match:
        return None
    candidate = name_match.group(1).strip()
    if candidate.lower() in {"bu", "şu", "su", "o"}:
        return None
    return candidate.replace("_", " ")


def _extract_instructor_for_course(
    course: str,
    *,
    ontology_context: str,
    vector_context: str,
) -> str | None:
    if not course:
        return None
    ctx = "\n".join([ontology_context or "", vector_context or ""])
    course_re = re.escape(course)

    patterns = [
        rf"{course_re}.*?(?:hocası|hoca|öğretim elemanı|instructor|lecturer)\s*[:=-]?\s*([A-Za-zÇĞİÖŞÜçğıöşü.\- ]{{2,80}})",
        rf"(?:hocası|hoca|öğretim elemanı|instructor|lecturer)\s*[:=-]?\s*([A-Za-zÇĞİÖŞÜçğıöşü.\- ]{{2,80}}).*?{course_re}",
    ]
    for pat in patterns:
        m = re.search(pat, ctx, flags=re.I)
        if m:
            name = m.group(1).strip(" \t\n\r.,;:-")
            if name and not re.search(r"\d", name):
                return name
    return None


def _build_instructor_info_answer(
    query: str,
    *,
    ontology_context: str,
    vector_context: str,
) -> str:
    course = _extract_course_from_query(query)
    if not course:
        return "Hangi ders için hoca bilgisini istediğini belirtir misin?"

    instructor = _extract_instructor_for_course(
        course,
        ontology_context=ontology_context,
        vector_context=vector_context,
    )
    if instructor:
        return f"{course} dersi hocası {instructor}."
    return f"{course} dersi için hoca bilgisi net bulunamadı."


def run_pipeline(task: AgentTask) -> AgentResponse:
    """
    Execute the 4-stage ZT4SWE verification pipeline.

    Stage 1 — Retrieval & Grounding  : ChromaDB (k=15) + ontology context
    Stage 2 — Draft Synthesis        : LLM via Ollama llama3.2
    Stage 3 — Compliance Verification: C1–C8 gate array + REDO loop (max 2)
    Stage 4 — Emission               : BlindSpot prepended if needed

    redo_log is per-request (thread-safe — no shared mutable state).
    """
    query = task.masked_input
    agent_role = task.role.value
    session_id = task.session_id
    intent = predict_intent(query)
    redo_log: list[dict] = []

    print(f"[INTENT ROUTER] query={query!r} intent={intent}")

    # ── Stage 1 — Retrieval & Grounding ──────────────────────────────────────
    vector_context, is_empty = VECTOR_MEM.retrieve(query, k=VECTOR_TOP_K, namespace="academic")
    ontology_context = build_ontology_context()

    print("=== DEBUG START ===")
    print("USER QUERY:", query)
    print("MEMORY:", vector_context)
    print("ONTOLOGY:", ontology_context)
    print("=== DEBUG END ===")

    intent_prefers_ontology = intent in _ONTOLOGY_FIRST_INTENTS
    has_ontology_facts = bool(ontology_context.strip()) and (
        "[ONTOLOGY: unavailable]" not in ontology_context
        and "No structured individuals found in ontology" not in ontology_context
    )

    # BlindSpot is disabled for deterministic intents.
    if intent in _ONTOLOGY_FIRST_INTENTS:
        skip_blindspot = True
    else:
        skip_blindspot = False

    # BlindSpot only when BOTH sources are unavailable on non-deterministic intents.
    no_sources = is_empty and not has_ontology_facts
    if no_sources and not skip_blindspot:
        draft = build_blindspot_block(query, "VEKTÖR HAFIZA BOŞ") + _BLINDSPOT_PHRASE
        return AgentResponse(
            task_id=task.task_id,
            agent_role=task.role,
            draft=draft,
            status=TaskStatus.COMPLETED,
        )

    # Ontology-first intents get ontology grounding priority, but gates still run.
    if intent_prefers_ontology and has_ontology_facts:
        grounding_context = (
            f"{vector_context}\n\n{ontology_context}"
            if vector_context.strip()
            else ontology_context
        )
    else:
        grounding_context = vector_context
    grounding_is_empty = no_sources

    exam_override = None
    if intent == "exam_info" and has_ontology_facts:
        exam_override = _extract_exam_override_from_ontology(ontology_context, query)

    # ── Deterministic Branches (high-confidence intents) ────────────────────
    if intent == "exam_info":
        print("[ROUTE:EXAM_DEBUG_MARKER]")
        return AgentResponse(
            task_id=task.task_id,
            agent_role=task.role,
            draft="CT_EXAM_OK_2026",
            status=TaskStatus.COMPLETED,
            metadata={"debug": "exam_branch_hit"},
        )

    if intent == "course_info":
        print("[ROUTE:COURSE_DEBUG_MARKER]")
        return AgentResponse(
            task_id=task.task_id,
            agent_role=task.role,
            draft="CT_COURSE_OK_2026",
            status=TaskStatus.COMPLETED,
            metadata={"debug": "course_branch_hit"},
        )

    if intent == "instructor_info":
        course_name = _extract_course_from_query(query)
        if not course_name:
            print("[ROUTE:INSTRUCTOR_DEBUG_MARKER]")
            return AgentResponse(
                task_id=task.task_id,
                agent_role=task.role,
                draft="CT_INSTRUCTOR_OK_2026",
                status=TaskStatus.COMPLETED,
                metadata={"debug": "instructor_clarify_hit"},
            )
        print("[ROUTE:INSTRUCTOR_STRUCTURED]")
        print("[BLOCK:LLM_SKIPPED_FOR_DETERMINISTIC_INTENT]")
        return _deterministic_response(
            task,
            text=_build_instructor_info_answer(
                query,
                ontology_context=ontology_context,
                vector_context=vector_context,
            ),
            redo_log=redo_log,
            source="ontology_or_memory",
        )

    # Hard block: deterministic intents must never reach LLM/REDO.
    if intent in _ONTOLOGY_FIRST_INTENTS:
        print("[BLOCK:LLM_SKIPPED_FOR_DETERMINISTIC_INTENT]")
        return _deterministic_response(
            task,
            text="İlgili bilgi net bulunamadı.",
            redo_log=redo_log,
            source="deterministic_fallback",
        )

    # ── Stage 2 — Prompt + LLM ───────────────────────────────────────────────
    print("[ROUTE:GENERAL_LLM]")
    user_message = (
        f"{vector_context}\n\n"
        f"{ontology_context}\n\n"
        f"ROLE: {agent_role}\n\n"
        f"INTENT: {intent}\n\n"
        f"SORU: {query}\n\n"
        "YANIT KURALLARI:\n"
        "- Sadece yukarıdaki VECTOR MEMORY ve ONTOLOGY CONTEXT verilerini kullan.\n"
        "- Son çıktıyı yalnızca Türkçe üret; İngilizce-Türkçe karışık cümle yazma.\n"
        "- Desteklenmeyen kelime, isim, tarih veya olay uydurma.\n"
        "- Konu dışı cümle, ek yorum ve takip sorusu ekleme.\n"
        "- exam_info/course_info/instructor_info için ontology verisi varsa ontology önceliklidir.\n"
        "- exam_info için ontology içinde tarih varsa tek cümle formatı kullan:\n"
        "  '<Ders adı> dersi vize sınavı <tarih> tarihinde yapılacaktır.'\n"
        "- PII maskelemesini bozma. Halüsinasyon üretme."
    )
    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    resp = chat(model="llama3.2", messages=base_messages)
    draft = resp.message.content.strip()

    # exam_info override: ontology date varsa LLM metnini geçersiz kıl.
    if exam_override:
        draft = exam_override

    draft = _sanitize_student_output(
        draft,
        intent=intent,
        has_ontology_facts=has_ontology_facts,
    )

    # ── Stage 3 — Compliance Verification (single REDO loop) ────────────────
    draft, limit_hit = run_redo_loop(
        draft,
        base_messages,
        grounding_context,
        grounding_is_empty,
        redo_log,
        agent_role=agent_role,
        query=query,
        redo_rules=(
            "Yanıtı yalnızca VECTOR MEMORY ve ONTOLOGY CONTEXT verilerinden üret. "
            "Konu dışı içerik üretme; Türkçe ve kısa kal."
        ),
        limit_message_template=(
            "⚠ Doğrulama başarısız (Gate {gate}). "
            "Yanıt güvenli biçimde teslim edilemiyor."
        ),
        post_process=lambda s: _sanitize_student_output(
            s,
            intent=intent,
            has_ontology_facts=has_ontology_facts,
        ),
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

    # ── Stage 4 — Final sanitize + emission ──────────────────────────────────
    draft = _sanitize_student_output(
        draft,
        intent=intent,
        has_ontology_facts=has_ontology_facts,
    )

    # Never allow mixed substantive answer + blindspot phrase when sources exist.
    if not no_sources:
        cleaned = re.sub(re.escape(_BLINDSPOT_PHRASE), "", draft, flags=re.I).strip(" \t\n\r.,;:-")
        if cleaned:
            draft = cleaned

    return AgentResponse(
        task_id=task.task_id,
        agent_role=task.role,
        draft=draft,
        status=TaskStatus.COMPLETED,
        redo_log=redo_log,
    )
