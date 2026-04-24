"""pipeline/shared.py — shared infrastructure for both pipeline runners.

Contains the singletons, constants, and utility functions needed by
both student_runner.py and developer_runner.py. Extracted from
src/services/api/pipeline.py so those runner modules can import
without creating a circular dependency back into pipeline.py.

Exported names used by pipeline.py and main_cli.py are re-imported
in pipeline.py at module level to preserve all existing call-sites.
"""

from __future__ import annotations

import re

from ollama import chat

from src.database.chroma_manager import db_manager
from src.ontology.loader import _get_ontology_graph, _sparql

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

VECTOR_TOP_K = 15

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
#  SINGLETONS
# ─────────────────────────────────────────────────────────────────────────────

CHROMA = db_manager


class VectorMemory:
    """
    Namespace-aware ChromaDB wrapper.

    retrieve(query, k, namespace) returns (context_block, is_empty).

    namespace must be one of the keys in chroma_manager.NAMESPACE_MAP:
      "academic"  — student path (default; backward compatible)
      "developer" — developer path
      "agile"     — scrum/agile path
    """

    def __init__(self) -> None:
        self._chroma = CHROMA

    def retrieve(
        self,
        query: str,
        k: int = VECTOR_TOP_K,
        namespace: str = "academic",
    ) -> tuple[str, bool]:
        """Returns (context_block, is_empty).

        Uses the namespace-specific collection so that student queries
        never see developer codebase snippets and vice-versa.
        """
        if self._chroma is None:
            return "", True
        try:
            documents = self._chroma.query_by_namespace(
                question=query, namespace=namespace, n_results=k
            )
            if not documents:
                return "", True
            lines = [f"=== VECTOR MEMORY (ChromaDB, ns={namespace!r}, k={k}) ==="]
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
#  UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize_output(text: str) -> str:
    """Strip internal retrieval labels that must not leak to end users."""
    return _LABEL_RE.sub("", text).strip()


def _safe_chat(model: str, messages: list) -> dict:
    """
    Wrap Ollama chat() so DeveloperOrchestrator receives a plain dict.

    Ollama returns a ChatResponse object; orchestrator expects
    {"message": {"content": str}}.  This wrapper normalises both shapes.
    """
    resp = chat(
        model=model,
        messages=messages,
        options={"temperature": 0.15, "top_p": 0.9},
    )
    if hasattr(resp, "message"):
        return {"message": {"content": resp.message.content}}
    if isinstance(resp, dict):
        return resp
    return {"message": {"content": str(resp)}}


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


def build_ontology_context() -> str:
    graph = _get_ontology_graph()
    if graph is None:
        print("[ONTOLOGY_CONTEXT] graph unavailable")
        return "[ONTOLOGY: unavailable]"
    print(f"[ONTOLOGY_CONTEXT] graph loaded triples={len(graph)}")

    lines = ["=== ONTOLOGY CONTEXT ==="]
    fact_count = 0

    def _safe_rows(label: str, query: str) -> list[dict]:
        try:
            rows = _sparql(query)
            print(f"[ONTOLOGY_QUERY] {label} rows={len(rows)}")
            return rows
        except Exception as exc:
            print(f"[ONTOLOGY_QUERY_ERROR] {label}: {exc}")
            return []

    # Exam → Course relationships
    exam_course_rows = _safe_rows(
        "exam_course",
        """
        PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX coode: <http://www.co-ode.org/ontologies/ont.owl#>
        SELECT ?exam ?course WHERE {
            ?exam coode:activityPartOf ?course .
        }""",
    )
    for r in exam_course_rows:
        exam   = r["exam"].split("/")[-1].split("#")[-1]
        course = r["course"].split("/")[-1].split("#")[-1]
        lines.append(f"  [Exam] {exam} belongs_to Course: {course}")
        fact_count += 1

    # Exam date facts (if provided in ontology).
    for r in _safe_rows(
        "exam_date",
        """
        PREFIX student: <http://cognitwin.org/student#>
        SELECT ?exam ?date WHERE {
            ?exam student:hasExamDate ?date .
        }""",
    ):
        exam = r["exam"].split("/")[-1].split("#")[-1]
        date = r["date"]
        lines.append(f"  [ExamDate] {exam} hasExamDate: {date}")
        fact_count += 1

    # Course → Instructor relationships.
    for r in _safe_rows(
        "course_instructor",
        """
        PREFIX student: <http://cognitwin.org/student#>
        SELECT ?course ?instructor WHERE {
            ?course student:hasInstructor ?instructor .
        }""",
    ):
        course = r["course"].split("/")[-1].split("#")[-1]
        instructor = r["instructor"].split("/")[-1].split("#")[-1]
        lines.append(f"  [Instructor] Course: {course} hasInstructor: {instructor}")
        fact_count += 1

    # Student → Course enrollments.
    for r in _safe_rows(
        "takes_course",
        """
        PREFIX student: <http://cognitwin.org/student#>
        SELECT ?student ?course WHERE {
            ?student student:takesCourse ?course .
        }""",
    ):
        student = r["student"].split("/")[-1].split("#")[-1]
        course = r["course"].split("/")[-1].split("#")[-1]
        lines.append(f"  [Enrollment] {student} takesCourse: {course}")
        fact_count += 1

    # Person → Agent links
    for r in _safe_rows(
        "person_agent",
        """
        PREFIX upper: <http://www.semanticweb.org/47ila/ontologies/2026/1/untitled-ontology-7/>
        SELECT ?person ?agent WHERE { ?person upper:hasAgent ?agent . }""",
    ):
        person = r["person"].split("/")[-1].split("#")[-1]
        agent  = r["agent"].split("/")[-1].split("#")[-1]
        lines.append(f"  [Person] {person} hasAgent: {agent}")
        fact_count += 1

    # Agent role assignments (namespace-agnostic filter by role name).
    for r in _safe_rows(
        "agent_roles",
        """
        PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?agent ?role WHERE {
            ?agent rdf:type ?role .
            FILTER(REGEX(STR(?role), "(StudentAgent|InstructorAgent|HeadOfDepartmentAgent|ResearcherAgent)$"))
        }""",
    ):
        agent = r["agent"].split("/")[-1].split("#")[-1]
        role  = r["role"].split("/")[-1].split("#")[-1]
        lines.append(f"  [Role] {agent} is a: {role}")
        fact_count += 1

    if fact_count == 0:
        lines.append("  (No structured individuals found in ontology)")

    lines.append("=== END ONTOLOGY CONTEXT ===")
    context = "\n".join(lines)
    print(f"[ONTOLOGY_CONTEXT_FACTS] count={fact_count}")
    print(f"[ONTOLOGY_CONTEXT_OUTPUT]\n{context}")
    return context
