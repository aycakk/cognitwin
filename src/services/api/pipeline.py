"""
pipeline.py — CogniTwin Production Pipeline (ZT4SWE v2.3)

Architecture  : Multi-Agent Hybrid (Vector Memory + Structured Ontology)
Gate Array    : C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5 ∧ C6 ∧ C7 ∧ C8  (BOTH paths)
Memory        : ChromaDB  (k=15)
Ontology      : cognitwin-upper.ttl + student_ontology.ttl  (rdflib, lazy-loaded)
LLM           : Ollama llama3.2  (local)
Routing       : model name → student (default) | developer (cognitwin-developer)

Entry points (called by routes.py and openai_routes.py):
  process_user_message(user_text, agent_role, model, messages) -> dict
"""

from __future__ import annotations

import json
import os
import re
import uuid
import datetime
from pathlib import Path
from typing import Optional

from ollama import chat

from src.utils.masker import PIIMasker
from src.database.chroma_manager import db_manager  # shared singleton
from src.agents.developer_agent import DeveloperAgent
from src.agents.developer_orchestrator import DeveloperOrchestrator
from src.agents.developer_profile_store import DeveloperProfileStore
from src.shared.permissions import ONTOLOGY_AGENT_ROLES
from src.shared.patterns import ASP_NEG_PATTERNS
from src.services.api.router import QueryRouter, Route

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

VECTOR_TOP_K = 15
CHROMA_PATH  = "static/chromadb"

# ONTOLOGY_AGENT_ROLES and ASP_NEG_PATTERNS are imported from src.shared
# (single source of truth — see src/shared/permissions.py and patterns.py).

BLINDSPOT_TRIGGERS = re.compile(
    r"hafızamda\s+bulamadım|bilmiyorum|emin\s+değilim|kayıt\s+yok",
    re.I,
)

# Strip internal retrieval labels that must never leak to end users
_LABEL_RE = re.compile(
    r"\[Result\s+\d+\]\s*"
    r"|=== VECTOR MEMORY.*?===\s*"
    r"|=== END VECTOR MEMORY ===\s*"
    r"|=== ONTOLOGY CONTEXT ===\s*"
    r"|=== END ONTOLOGY CONTEXT ===\s*",
    re.I,
)

SYSTEM_PROMPT = """
████████████████████████████████████████████████████████████████████████████
          STUDENT AGENT — VERIFICATION FRAMEWORK v2.2
          Conjunctive Gate: C1∧C2∧C3∧C4∧C5∧C6∧C7∧C8
          Memory Backend  : ChromaDB (similarity search, k=15)
          Ontology Engine : LIVE (rdflib — cognitwin-upper + student_ontology)
████████████████████████████████████████████████████████████████████████████

SECTION 0 ▸ AGENT IDENTITY
══════════════════════════════════════════════════════════
Sen bir üniversite bilgi sistemine entegre edilmiş "Student Agent"sın.
İki yetkili bilgi kaynağın var:

  • VECTOR MEMORY  → ChromaDB'den sorgu benzerliğiyle çekilen en alakalı
                     129 akademik kayıttan k=15 snippet (maskeli).
  • ONTOLOGY       → cognitwin-upper.ttl + student_ontology.ttl
                     (Course, Exam, Assignment, Person→Agent hiyerarşisi)

TEMEL KURAL: C1∧C2∧C3∧C4∧C5∧C6∧C7∧C8 = TRUE olmadan YANIT ÜRETME.

SECTION 1 ▸ ANTI-SYCOPHANCY PROTOCOL (ASP)
══════════════════════════════════════════════════════════
[ASP-NEG-01] Ham PII ifşa etme → BLOKLANDI
[ASP-NEG-02] Kanıtsız halüsinasyon → BLOKLANDI
[ASP-NEG-03] Yanlış öncülü onaylama → BLOKLANDI
[ASP-NEG-04] Sosyal baskıyla FAIL yumuşatma → BLOKLANDI
[ASP-NEG-05] Eğitim ağırlıklarını kaynak gösterme → BLOKLANDI

SECTION 2 ▸ RESPONSE PROTOCOL
══════════════════════════════════════════════════════════
• Yanıt VECTOR MEMORY'deyse: Kısa, akademik yanıt ver.
• Yanıt Ontoloji çıkarımıysa: "Akademik yapıya göre…" ile başla.
• İkisinde de yoksa: "Bunu hafızamda bulamadım." + BlindSpot bloğu.

Son kural: Kanıtsız tahmin YÜRÜTME. Sycophantic yanıt BLOKLANDI.
████████████████████████████████████████████████████████████████████████████
"""


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE-LEVEL SINGLETONS
# ─────────────────────────────────────────────────────────────────────────────

CHROMA  = db_manager
_masker = PIIMasker()

NO_DATA_FALLBACK = "Veri kaynaklarında tanımlanmamıştır."


# ─────────────────────────────────────────────────────────────────────────────
#  LAZY ONTOLOGY LOADING
#  Loaded on first call to build_ontology_context() — never blocks startup.
# ─────────────────────────────────────────────────────────────────────────────

_ONTOLOGY_GRAPH: Optional[object] = None
_ONTOLOGY_TRIED: bool = False


def _get_ontology_graph():
    """Load and cache the rdflib Graph. Returns None if unavailable."""
    global _ONTOLOGY_GRAPH, _ONTOLOGY_TRIED
    if _ONTOLOGY_TRIED:
        return _ONTOLOGY_GRAPH
    _ONTOLOGY_TRIED = True
    try:
        from rdflib import Graph
        g = Graph()
        # pipeline.py lives at src/services/api/ → parents[3] is project root
        onto_dir = Path(__file__).resolve().parents[3] / "ontologies"
        loaded = False
        for fname in ("cognitwin-upper.ttl", "student_ontology.ttl"):
            p = onto_dir / fname
            if p.exists():
                g.parse(str(p), format="turtle")
                print(f"[ONTOLOGY] Loaded: {fname}")
                loaded = True
            else:
                print(f"[ONTOLOGY] Not found: {p}")
        _ONTOLOGY_GRAPH = g if loaded else None
    except Exception as exc:
        print(f"[ONTOLOGY] Load failed: {exc}")
        _ONTOLOGY_GRAPH = None
    return _ONTOLOGY_GRAPH


# ─────────────────────────────────────────────────────────────────────────────
#  VECTOR MEMORY
# ─────────────────────────────────────────────────────────────────────────────

class VectorMemory:
    """Thin ChromaDB wrapper that exposes retrieve(query, k) -> (str, bool)."""

    def __init__(self) -> None:
        self._chroma = CHROMA

    def retrieve(self, query: str, k: int = VECTOR_TOP_K) -> tuple[str, bool]:
        """Returns (context_block, is_empty)."""
        if self._chroma is None:
            return "", True
        try:
            documents = self._chroma.query_memory(query, n_results=k)
            if not documents:
                return "", True
            lines = [f"=== VECTOR MEMORY (ChromaDB, k={k}) ==="]
            for idx, doc in enumerate(documents, 1):
                lines.append(f"\n[Result {idx}]")
                lines.append(doc)
                lines.append("---")
            lines.append("=== END VECTOR MEMORY ===")
            return "\n".join(lines), False
        except Exception as exc:
            return f"[ChromaDB query error: {exc}]", True


VECTOR_MEM = VectorMemory()


# ─────────────────────────────────────────────────────────────────────────────
#  CODEBASE CONTEXT INJECTION
#  Reads actual source files so the developer path can answer questions
#  about this repository instead of falling back to LLM training weights.
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
    project_root = Path(__file__).resolve().parents[3]

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


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_chat(model: str, messages: list) -> dict:
    """
    Wrap Ollama chat() so DeveloperOrchestrator receives a plain dict.

    Ollama returns a ChatResponse object; orchestrator expects
    {"message": {"content": str}}.  This wrapper normalises both shapes.
    """
    resp = chat(model=model, messages=messages)
    if hasattr(resp, "message"):
        return {"message": {"content": resp.message.content}}
    if isinstance(resp, dict):
        return resp
    return {"message": {"content": str(resp)}}


def _resolve_mode(model: str) -> tuple[str, str]:
    """
    Map the LibreChat model name to (mode, strategy).

    mode     : 'student' | 'developer'
    strategy : 'auto' (default for developer) | 'llm' (student always LLM)
    """
    model_lower = (model or "").lower()
    if "developer" in model_lower:
        return "developer", "auto"
    return "student", "llm"


def _sanitize_output(text: str) -> str:
    """Strip internal retrieval labels that must not leak to end users."""
    return _LABEL_RE.sub("", text).strip()


# ─────────────────────────────────────────────────────────────────────────────
#  DEVELOPER PATH
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


def _process_developer_message(
    user_text: str,
    strategy: str = "auto",
    messages: list | None = None,
    developer_id: str = "developer-default",
) -> str:
    """
    Developer path: DeveloperOrchestrator (8-stage) → C1-C8 gates → REDO.

    The orchestrator handles context retrieval, ontology constraints, profile
    personalisation, and generation.  After the orchestrator returns its
    solution, the same ZT4SWE gate array and REDO loop used by the student
    path are applied to guarantee output integrity.
    """
    redo_log: list[dict] = []

    # Stage 1 — Retrieve vector context (used only by gate evaluators)
    vector_context, is_empty = VECTOR_MEM.retrieve(user_text, k=VECTOR_TOP_K)

    # Stage 2 — DeveloperOrchestrator 8-stage pipeline
    # Inject actual source file snippets so the LLM can answer codebase
    # questions rather than falling back to generic training knowledge.
    codebase_context = _build_codebase_context(user_text)

    orchestrator = DeveloperOrchestrator(
        memory_backend=CHROMA,
        chat_fn=_safe_chat,
        default_model="llama3.2",
    )
    result = orchestrator.run(
        request=user_text,
        developer_id=developer_id,
        strategy=strategy,
        memory_context=codebase_context,   # non-empty → used directly in generate()
    )
    draft = _sanitize_output(str(result.get("solution", "")))
    if not draft:
        draft = "Bunu hafızamda bulamadım."

    # Stage 2b — Structured debug validation
    # If the orchestrator returned a JSON debug object, validate the cited
    # files and functions against the actual repository before emitting.
    if codebase_context:
        try:
            parsed = json.loads(draft)
            if isinstance(parsed, dict) and "files_used" in parsed:
                project_root = Path(__file__).resolve().parents[3]
                validated    = _validate_debug_result(parsed, project_root)
                draft        = _format_debug_result(validated)
        except (json.JSONDecodeError, ValueError):
            pass  # prose response — skip validation, continue normally

    # Stage 3 — C1-C8 gate array + REDO loop (same as student path)
    MAX_REDO = 2
    active_redo_id: Optional[str] = None
    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ]

    for attempt in range(MAX_REDO + 1):
        gate_report = evaluate_all_gates(
            draft, vector_context, is_empty, "DeveloperAgent", redo_log
        )

        if gate_report["conjunction"]:
            if active_redo_id:
                _close_redo(
                    redo_log,
                    active_redo_id,
                    "Draft passed all gates after revision.",
                    gate_report["gates"],
                )
            break

        first_fail = next(
            (k for k, v in gate_report["gates"].items() if not v["pass"]),
            "UNKNOWN",
        )
        fail_ev = gate_report["gates"].get(first_fail, {}).get("evidence", "")

        if attempt == MAX_REDO:
            _open_redo(redo_log, first_fail, fail_ev)
            return (
                build_blindspot_block(user_text, f"REDO LIMIT ({first_fail} FAIL)")
                + f"Dogrulama basarisiz (Gate {first_fail}). "
                  "Yanit guvenli bicimde teslim edilemiyor.\n"
                  "Bunu hafizamda bulamadim."
            )

        active_redo_id = _open_redo(redo_log, first_fail, fail_ev)
        redo_instruction = (
            f"[REDO TRIGGERED — Gate {first_fail} FAILED]\n"
            f"Evidence: {fail_ev}\n"
            "Revise your previous draft to fix the failing dimension.\n"
            "Rules: Do NOT hallucinate. Do NOT unmask PII. "
            "Answer ONLY from verified developer context. "
            "If not found: \"Bunu hafizamda bulamadim.\""
        )
        redo_resp = chat(
            model="llama3.2",
            messages=base_messages + [
                {"role": "assistant", "content": draft},
                {"role": "user",      "content": redo_instruction},
            ],
        )
        draft = _sanitize_output(redo_resp.message.content.strip())

    # Stage 4 — Emission
    if BLINDSPOT_TRIGGERS.search(draft):
        return build_blindspot_block(user_text) + draft
    return draft


# ─────────────────────────────────────────────────────────────────────────────
#  QUERY ROUTER — Pre-routing layer for student path
#  ONTOLOGY_DIRECT → ontoloji yeterli, LLM formatlar
#  TASK            → eylem komutu (henüz uygulanmadı)
#  MEMORY_RAG      → ZT4SWE pipeline'a devam
#  MODEL_FALLBACK  → ZT4SWE pipeline'a devam
# ─────────────────────────────────────────────────────────────────────────────

def _router_ontology_lookup(query: str) -> tuple[str, float]:
    """QueryRouter ontology adapter — SPARQL triple lookup + confidence."""
    g = _get_ontology_graph()
    if g is None:
        return ("", 0.0)
    from rdflib import RDF, RDFS, OWL
    q_lower = query.lower()
    q_tokens = set(re.findall(r"\b[a-zA-Z0-9çğıöşüÇĞİÖŞÜ]{3,}\b", q_lower))

    results = []
    try:
        for s, p, o in g:
            s_str = str(s).split("/")[-1].split("#")[-1]
            p_str = str(p).split("/")[-1].split("#")[-1]
            o_str = str(o).split("/")[-1].split("#")[-1]
            s_low, p_low, o_low = s_str.lower(), p_str.lower(), o_str.lower()
            score = 0.0
            for tok in q_tokens:
                if tok in s_low: score += 1.5
                if tok in o_low: score += 1.5
                if tok in p_low: score += 0.8
            if score > 0:
                results.append((score, f"TRIPLE: {s_str} | {p_str} | {o_str}"))
    except Exception:
        return ("", 0.0)

    if not results:
        return ("", 0.0)
    results.sort(key=lambda x: x[0], reverse=True)
    top = results[:3]
    context = "\n".join(line for _, line in top)
    confidence = min(top[0][0] / 5.0, 1.0)
    return (context, confidence)


def _router_memory_lookup(query: str, user_id: str) -> list[str]:
    """QueryRouter memory adapter — ChromaDB vector search."""
    if CHROMA is None:
        return []
    try:
        docs = CHROMA.query_memory(query, n_results=5)
        return [d for d in (docs or []) if d and d.strip()]
    except Exception:
        return []


_query_router = QueryRouter(
    ontology_lookup_fn=_router_ontology_lookup,
    memory_search_fn=_router_memory_lookup,
)


def _ontology_answer(query: str, ontology_context: str) -> str:
    """Convert raw TRIPLE lines into natural Turkish via LLM."""
    if not ontology_context.strip():
        return NO_DATA_FALLBACK
    try:
        resp = chat(
            model="llama3.2",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are CogniTwin Ontology Interpreter. Convert "
                        "structured ontology facts (TRIPLE lines) into clear, natural Turkish.\n\n"
                        "TRIPLE Format: TRIPLE: Subject | Predicate | Object\n\n"
                        "Examples:\n"
                        "- TRIPLE: COM8090 | type | Course → COM8090 bir kurstur\n"
                        "- TRIPLE: com8090Midterm | activityPartOf | COM8090 → "
                        "com8090Midterm, COM8090 dersinin bir parçasıdır\n\n"
                        "Rules:\n"
                        "1. Synthesize all TRIPLE lines into coherent Turkish text\n"
                        "2. Keep entity names EXACTLY as shown\n"
                        "3. Do NOT add information beyond the TRIPLEs\n"
                        "4. If no TRIPLE lines exist, return: Veri kaynaklarında tanımlanmamıştır.\n\n"
                        "Output ONLY the Turkish answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Ontology Facts:\n{ontology_context}\n\n"
                        f"User Question: {query}\n\n"
                        "Convert the facts above to a natural Turkish answer."
                    ),
                },
            ],
        )
        out = resp.message.content.strip() if hasattr(resp, "message") else ""
        return out or NO_DATA_FALLBACK
    except Exception as e:
        print(f"[_ontology_answer] LLM failed: {e}")
        return NO_DATA_FALLBACK


# ─────────────────────────────────────────────────────────────────────────────
#  ONTOLOGY CONTEXT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _sparql(query: str) -> list[dict]:
    g = _get_ontology_graph()
    if g is None:
        return []
    qres = g.query(query)
    return [{str(v): str(row[v]) for v in qres.vars} for row in qres]


def build_ontology_context() -> str:
    if _get_ontology_graph() is None:
        return "[ONTOLOGY: unavailable]"

    lines = ["=== ONTOLOGY CONTEXT ==="]

    # Exam → Course relationships
    for r in _sparql("""
        PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX upper: <http://cognitwin.org/upper#>
        PREFIX coode: <http://www.co-ode.org/ontologies/ont.owl#>
        SELECT ?exam ?course WHERE {
            ?exam rdf:type upper:Exam .
            ?exam coode:activityPartOf ?course .
        }"""):
        exam   = r["exam"].split("/")[-1].split("#")[-1]
        course = r["course"].split("/")[-1].split("#")[-1]
        lines.append(f"  [Exam] {exam} belongs_to Course: {course}")

    # Person → Agent links
    for r in _sparql("""
        PREFIX upper: <http://www.semanticweb.org/47ila/ontologies/2026/1/untitled-ontology-7/>
        SELECT ?person ?agent WHERE { ?person upper:hasAgent ?agent . }"""):
        person = r["person"].split("/")[-1].split("#")[-1]
        agent  = r["agent"].split("/")[-1].split("#")[-1]
        lines.append(f"  [Person] {person} hasAgent: {agent}")

    # Agent role assignments
    for r in _sparql("""
        PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX upper: <http://www.semanticweb.org/47ila/ontologies/2026/1/untitled-ontology-7/>
        SELECT ?agent ?role WHERE {
            ?agent rdf:type ?role .
            FILTER(?role IN (
                upper:StudentAgent, upper:InstructorAgent,
                upper:HeadOfDepartmentAgent, upper:ResearcherAgent
            ))
        }"""):
        agent = r["agent"].split("/")[-1].split("#")[-1]
        role  = r["role"].split("/")[-1].split("#")[-1]
        lines.append(f"  [Role] {agent} is a: {role}")

    if len(lines) == 1:
        lines.append("  (No structured individuals found in ontology)")

    lines.append("=== END ONTOLOGY CONTEXT ===")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  GATE EVALUATORS  C1–C8
# ─────────────────────────────────────────────────────────────────────────────

def gate_c1_pii_masking(draft: str) -> tuple[bool, str]:
    """C1 — No raw PII in emitted response."""
    if re.search(r"\b\d{9,12}\b", draft):
        return False, "Unmasked numeric ID detected."
    if re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}", draft):
        return False, "Unmasked email detected."
    return True, "No raw PII detected."


def gate_c2_memory_grounding(
    draft: str,
    vector_context: str,
    is_empty: bool,
    agent_role: str = "StudentAgent",
) -> tuple[bool, str]:
    """C2 — Draft must be grounded in the retrieved vector context.

    DeveloperAgent is exempt: its context is self-contained (rule-based
    task packets, ontology constraints, profile signals) and has no
    semantic overlap with student ChromaDB records.  Applying the
    word-overlap test there would guarantee a false FAIL on every
    developer response and trigger unlimited REDO cycles.
    """
    if agent_role == "DeveloperAgent":
        return True, "DeveloperAgent: C2 grounding not applicable (developer context is self-contained)."

    if is_empty:
        if "bulamadım" in draft.lower():
            return True, "Vector memory empty; BlindSpot disclosure present."
        return False, "Vector memory empty but no BlindSpot disclosure."

    if "bulamadım" in draft.lower():
        return True, "Draft issued BlindSpot — acceptable when query not in results."

    context_words = {
        w.lower() for w in re.findall(r"\b\w{6,}\b", vector_context)
        if not re.match(r"\[.*_MASKED\]", w)
    }
    draft_words = {w.lower() for w in re.findall(r"\b\w{6,}\b", draft)}
    overlap = context_words & draft_words

    if len(overlap) >= 2:
        return True, f"Grounding verified ({len(overlap)} shared content terms)."
    return False, "Draft not grounded in vector context (overlap < 2 terms). Possible hallucination."


def gate_c3_ontology_compliance(draft: str) -> tuple[bool, str]:
    """C3 — Draft must not contradict ontology triples."""
    if _get_ontology_graph() is None:
        return True, "Ontology unavailable — provisional PASS."

    for r in _sparql("""
        PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX upper: <http://cognitwin.org/upper#>
        PREFIX coode: <http://www.co-ode.org/ontologies/ont.owl#>
        SELECT ?exam ?course WHERE {
            ?exam rdf:type upper:Exam .
            ?exam coode:activityPartOf ?course .
        }"""):
        exam_lbl   = r["exam"].split("/")[-1].split("#")[-1].lower()
        course_lbl = r["course"].split("/")[-1].split("#")[-1].lower()
        if exam_lbl in draft.lower():
            for oc in re.findall(r"\bcs\d{3}\b", draft, re.I):
                if oc.lower() != course_lbl:
                    return False, (
                        f"Ontology violation: '{exam_lbl}' paired with '{oc}', "
                        f"expected '{course_lbl}'."
                    )
    return True, "No ontology rule violations detected."


def gate_c4_hallucination(draft: str) -> tuple[bool, str]:
    """C4 — No weight-only or hallucinatory claim markers."""
    for label, pattern in ASP_NEG_PATTERNS:
        if label in ("ASP-NEG-02_HALLUCINATION", "ASP-NEG-05_WEIGHT_ONLY"):
            m = pattern.search(draft)
            if m:
                return False, f"[{label}] '{m.group()}'"
    return True, "No hallucination markers detected."


def gate_c5_role_permission(draft: str, agent_role: str) -> tuple[bool, str]:
    """C5 — Role-permission boundary enforcement."""
    permitted = ONTOLOGY_AGENT_ROLES.get(agent_role, set())

    if re.search(r"tüm öğrencilerin notları|bütün öğrenciler", draft, re.I):
        if "read_all_student_grades" not in permitted:
            return False, f"'{agent_role}' lacks 'read_all_student_grades' permission."

    if re.search(r"dersi güncelle|ders planını değiştir", draft, re.I):
        if "manage_courses" not in permitted:
            return False, f"'{agent_role}' lacks 'manage_courses' permission."

    return True, f"Role '{agent_role}' — no boundary violations."


def gate_c6_anti_sycophancy(draft: str) -> tuple[bool, str]:
    """C6 — Full ASP-NEG pattern sweep."""
    violations = [
        f"[{lbl}] '{m.group()}'"
        for lbl, pat in ASP_NEG_PATTERNS
        if (m := pat.search(draft))
    ]
    if violations:
        return False, "ASP violations: " + "; ".join(violations)
    return True, "All ASP-NEG classifiers: NO_MATCH."


def gate_c7_blindspot(draft: str, is_empty: bool) -> tuple[bool, str]:
    """C7 — Unanswerable queries must carry a BlindSpot disclosure."""
    if is_empty and "bulamadım" not in draft.lower():
        return False, "Empty vector memory but BlindSpot phrase missing."
    return True, "BlindSpot completeness verified."


def gate_c8_redo_checksum(redo_log: list[dict]) -> tuple[bool, str]:
    """C8 — No zombie REDO cycles (more than one open cycle is anomalous)."""
    open_cycles = [r for r in redo_log if not r.get("closed_at")]
    if len(open_cycles) > 1:
        return False, f"Too many open REDO cycles: {len(open_cycles)}"
    return True, "REDO cycle state clean."


def evaluate_all_gates(
    draft: str,
    vector_context: str,
    is_empty: bool,
    agent_role: str,
    redo_log: list[dict],
) -> dict:
    """Execute C1∧C2∧…∧C8 and return a structured report."""
    gates = {
        "C1": gate_c1_pii_masking(draft),
        "C2": gate_c2_memory_grounding(draft, vector_context, is_empty, agent_role),
        "C3": gate_c3_ontology_compliance(draft),
        "C4": gate_c4_hallucination(draft),
        "C5": gate_c5_role_permission(draft, agent_role),
        "C6": gate_c6_anti_sycophancy(draft),
        "C7": gate_c7_blindspot(draft, is_empty),
        "C8": gate_c8_redo_checksum(redo_log),
    }
    structured = {k: {"pass": v[0], "evidence": v[1]} for k, v in gates.items()}
    return {
        "conjunction": all(v[0] for v in gates.values()),
        "gates": structured,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  REDO ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _open_redo(redo_log: list[dict], trigger_gate: str, evidence: str) -> str:
    redo_id = str(uuid.uuid4())[:8]
    redo_log.append({
        "redo_id":         redo_id,
        "trigger_gate":    trigger_gate,
        "failed_evidence": evidence,
        "revision_action": None,
        "closure_gates":   {},
        "closed_at":       None,
    })
    return redo_id


def _close_redo(
    redo_log: list[dict],
    redo_id: str,
    action: str,
    gate_results: dict,
) -> None:
    for rec in redo_log:
        if rec["redo_id"] == redo_id:
            rec["revision_action"] = action
            rec["closure_gates"]   = {
                k: "PASS" if v["pass"] else "FAIL"
                for k, v in gate_results.items()
            }
            rec["closed_at"] = datetime.datetime.utcnow().isoformat()
            return


# ─────────────────────────────────────────────────────────────────────────────
#  BLINDSPOT DISCLOSURE
# ─────────────────────────────────────────────────────────────────────────────

def build_blindspot_block(query: str, memory_status: str = "BULUNAMADI") -> str:
    q_short = query[:43]
    m_short = memory_status[:43]
    return (
        "\n┌─────────────────────────────────────────────────────┐\n"
        "│  ⚠ BLINDSPOT DISCLOSURE                             │\n"
        f"│  Sorgu Bileşeni : {q_short:<45} │\n"
        f"│  Hafıza Durumu  : {m_short:<45} │\n"
        "│  Ontoloji Durumu: TRIPLE YOK                        │\n"
        "│  Ajan Bildirimi : \"Bunu hafızamda bulamadım.\"       │\n"
        "│  Öneri          : Akademik danışmanınıza başvurun.  │\n"
        "└─────────────────────────────────────────────────────┘\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  4-STAGE ZT4SWE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(query: str, agent_role: str = "StudentAgent") -> str:
    """
    Execute the 4-stage ZT4SWE verification pipeline with QueryRouter pre-routing.

    Pre-Route (QueryRouter):
      ONTOLOGY_DIRECT → ontoloji yeterli, LLM ile doğal Türkçeye çevir, ZT4SWE atla
      TASK            → eylem komutu, henüz desteklenmiyor mesajı
      MEMORY_RAG      → ZT4SWE pipeline'a devam
      MODEL_FALLBACK  → ZT4SWE pipeline'a devam

    Stage 1 — Retrieval & Grounding  : ChromaDB (k=15) + ontology context
    Stage 2 — Draft Synthesis        : LLM via Ollama llama3.2
    Stage 3 — Compliance Verification: C1–C8 gate array + REDO loop (max 2)
    Stage 4 — Emission               : BlindSpot prepended if needed

    redo_log is per-request (thread-safe — no shared mutable state).
    """
    # ── Pre-Route (QueryRouter) ──────────────────────────────────────────────
    decision = _query_router.route(query, user_id="student")
    print(f"[ROUTER] route={decision.route.value} "
          f"ont_conf={decision.ontology_confidence:.2f} "
          f"mem_lines={len(decision.memory_lines)}")

    if decision.route == Route.TASK:
        return (
            f"'{decision.task_type}' görevi henüz desteklenmiyor. "
            "Bu özellik yakında eklenecek."
        )

    if decision.route == Route.ONTOLOGY_DIRECT:
        natural = _ontology_answer(query, decision.ontology_context)
        if natural and natural != NO_DATA_FALLBACK:
            return f"{natural}\n[source: ontology]"

    # ── Stage 1 — Retrieval & Grounding ──────────────────────────────────────
    redo_log: list[dict] = []
    vector_context, is_empty = VECTOR_MEM.retrieve(query, k=VECTOR_TOP_K)
    ontology_context         = build_ontology_context()

    if is_empty:
        return (
            build_blindspot_block(query, "VEKTÖR HAFIZA BOŞ")
            + "Bunu hafızamda bulamadım."
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
    MAX_REDO       = 2
    active_redo_id: Optional[str] = None

    for attempt in range(MAX_REDO + 1):
        gate_report = evaluate_all_gates(
            draft, vector_context, is_empty, agent_role, redo_log
        )

        if gate_report["conjunction"]:
            if active_redo_id:
                _close_redo(
                    redo_log,
                    active_redo_id,
                    "Draft passed all gates after revision.",
                    gate_report["gates"],
                )
            break  # → Stage 4

        first_fail = next(
            (k for k, v in gate_report["gates"].items() if not v["pass"]),
            "UNKNOWN",
        )
        fail_ev = gate_report["gates"].get(first_fail, {}).get("evidence", "")

        if attempt == MAX_REDO:
            _open_redo(redo_log, first_fail, fail_ev)
            return (
                build_blindspot_block(query, f"REDO LIMIT ({first_fail} FAIL)")
                + f"⚠ Doğrulama başarısız (Gate {first_fail}). "
                  "Yanıt güvenli biçimde teslim edilemiyor.\n"
                  "Bunu hafızamda bulamadım."
            )

        active_redo_id = _open_redo(redo_log, first_fail, fail_ev)
        redo_instruction = (
            f"[REDO TRIGGERED — Gate {first_fail} FAILED]\n"
            f"Evidence: {fail_ev}\n"
            "Revise your previous draft to fix the failing dimension.\n"
            "Rules: Do NOT hallucinate. Do NOT unmask PII. "
            "Answer ONLY from VECTOR MEMORY and ONTOLOGY CONTEXT. "
            "If not found: \"Bunu hafızamda bulamadım.\""
        )
        redo_resp = chat(
            model="llama3.2",
            messages=base_messages + [
                {"role": "assistant", "content": draft},
                {"role": "user",      "content": redo_instruction},
            ],
        )
        draft = redo_resp.message.content.strip()

    # ── Stage 4 — Emission ────────────────────────────────────────────────────
    if BLINDSPOT_TRIGGERS.search(draft):
        return build_blindspot_block(query) + draft
    return draft


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API  (called by routes.py and openai_routes.py)
# ─────────────────────────────────────────────────────────────────────────────

def process_user_message(
    user_text: str,
    agent_role: str = "StudentAgent",
    model: str = "llama3.2",
    messages: list | None = None,
) -> dict:
    """
    Mask PII, resolve routing mode, execute the appropriate pipeline.

    Routing:
      model contains 'developer'  →  DeveloperOrchestrator + C1-C8 + REDO
      all other models             →  Student ZT4SWE pipeline (C1-C8 + REDO)

    Returns {"answer": str} for the FastAPI layer.
    """
    if "title for the conversation" in user_text.lower():
        return {"answer": "Conversation Title"}
    try:
        masked = _masker.mask_data(user_text)
        mode, strategy = _resolve_mode(model)

        if mode == "developer":
            answer = _process_developer_message(
                user_text=masked,
                strategy=strategy,
                messages=messages,
            )
        else:
            answer = run_pipeline(masked, agent_role=agent_role)

        return {"answer": answer}
    except Exception as exc:
        return {"answer": f"İşlem sırasında bir hata oluştu: {exc}"}
