"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   STUDENT AGENT — main_cli.py  (Verification Framework v2.2)               ║
║   Conjunctive Gate Array : C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5 ∧ C6 ∧ C7 ∧ C8         ║
║   Ontology Engine        : rdflib  (LIVE — C3 & C5)                        ║
║   Memory Backend         : ChromaDB  (LIVE — similarity search, k=15)      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Directory expectations
──────────────────────
  static/chromadb/          ← your pre-ingested ChromaDB instance (129 records)
  data/cognitwin-upper.ttl  ← upper ontology
  data/student_ontology.ttl ← student ontology
  src/database/vector_store.py  ← ChromaManager class
"""

from __future__ import annotations  # BU HER ZAMAN EN ÜSTTE OLMALI

import sys
import os

# Proje kök dizinini tanıtma (Import hatasını çözen kısım)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # src'nin bir üstü
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Şimdi importlar gelebilir
from src.database.chroma_manager import db_manager
import re
import uuid
import datetime
from typing import Optional

from ollama import chat

# ── Optional: masker for legacy raw-file fallback ────────────────────────────
try:
    from utils.masker import PIIMasker
    _MASKER_AVAILABLE = True
except ImportError:
    _MASKER_AVAILABLE = False

# ── rdflib ───────────────────────────────────────────────────────────────────
try:
    from rdflib import Graph, Namespace, RDF
    _RDFLIB_AVAILABLE = True
except ImportError:
    _RDFLIB_AVAILABLE = False

# ── ChromaDB / ChromaManager ──────────────────────────────────────────────────
from src.database.chroma_manager import db_manager


# ─────────────────────────────────────────────────────────────────────────────
#  NAMESPACE DECLARATIONS  (mirrors the two .ttl files)
# ─────────────────────────────────────────────────────────────────────────────
UPPER_NS   = Namespace("http://www.semanticweb.org/47ila/ontologies/2026/1/untitled-ontology-7/")
STUDENT_NS = Namespace("http://cognitwin.org/student#")
UPPER_BASE = Namespace("http://cognitwin.org/upper#")
COODE_NS   = Namespace("http://www.co-ode.org/ontologies/ont.owl#")

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

VECTOR_TOP_K = 15          # similarity search budget
CHROMA_PATH  = "static/chromadb"   # default persistence path
COLLECTION_NAME = "academic_memory"  # change if your collection has a different name

ONTOLOGY_AGENT_ROLES: dict[str, set[str]] = {
    "StudentAgent":          {"read_own_grades", "read_own_courses",
                              "read_exam_dates", "read_assignment_deadlines"},
    "InstructorAgent":       {"read_own_grades", "read_own_courses",
                              "read_exam_dates", "read_assignment_deadlines",
                              "read_all_student_grades", "manage_courses"},
    "HeadOfDepartmentAgent": {"read_own_grades", "read_own_courses",
                              "read_exam_dates", "read_assignment_deadlines",
                              "read_all_student_grades", "manage_courses",
                              "manage_department"},
    "ResearcherAgent":       {"read_own_courses", "read_exam_dates",
                              "read_assignment_deadlines"},
}

ASP_NEG_PATTERNS = [
    ("ASP-NEG-01_PII_UNMASK",    re.compile(r"\b\d{8,11}\b")),
    ("ASP-NEG-02_HALLUCINATION", re.compile(r"tahminim|sanırım|galiba|muhtemelen", re.I)),
    ("ASP-NEG-03_FALSE_PREMISE", re.compile(r"haklısınız|evet, öyle söylemiştim", re.I)),
    ("ASP-NEG-04_SOFTENED_FAIL", re.compile(r"yine de cevaplamaya çalışayım|bence şöyle olabilir", re.I)),
    ("ASP-NEG-05_WEIGHT_ONLY",   re.compile(r"genel bilgime göre|eğitim verilerime göre", re.I)),
]

BLINDSPOT_TRIGGERS = re.compile(
    r"hafızamda\s+bulamadım|bilmiyorum|emin\s+değilim|kayıt\s+yok",
    re.I,
)

REDO_LOG: list[dict] = []   # in-memory audit trail for this session


# ─────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT  (v2.2 — Vector Memory aware)
# ─────────────────────────────────────────────────────────────────────────────

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
#  MODULE-LEVEL SINGLETONS  (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────

CHROMA = db_manager


def _init_ontology_graph() -> Optional["Graph"]:
    if not _RDFLIB_AVAILABLE:
        return None
    g = Graph()
    
    # Dosyaların bulunduğu tam yolu senin ekran görüntüne göre ayarlıyoruz
    current_dir = os.path.dirname(os.path.abspath(__file__)) # src klasörü
    project_root = os.path.dirname(current_dir) # CogniTwin ana klasörü
    
    # Ekran görüntüsündeki 'ontologies' klasörüne gidiyoruz
    ontology_folder = os.path.join(project_root, "ontologies")

    files = ["cognitwin-upper.ttl", "student_ontology.ttl"]
    found_any = False

    for fname in files:
        target_path = os.path.join(ontology_folder, fname)
        if os.path.exists(target_path):
            try:
                g.parse(target_path, format="turtle")
                print(f"✅ [ONTOLOGY] Yüklendi: {fname}")
                found_any = True
            except Exception as e:
                print(f"❌ [ONTOLOGY] Hata ({fname}): {e}")

    return g if found_any else None


# Singletons
ONTOLOGY_GRAPH: Optional["Graph"] = _init_ontology_graph()


# ─────────────────────────────────────────────────────────────────────────────
#  VECTOR MEMORY RETRIEVAL  (replaces load_footprints flat-file read)
# ─────────────────────────────────────────────────────────────────────────────

class VectorMemory:
    """
    Thin wrapper that exposes a single `retrieve(query, k)` method.
    Falls back to the flat footprints.txt file if ChromaDB is unavailable.
    """

    def __init__(self, chroma: Optional["ChromaManager"]) -> None:
        self._chroma = chroma
        self._fallback_text: str = ""
        self._fallback_loaded: bool = False

        if chroma is None:
            self._fallback_text, _ = self._load_flat_file()
            self._fallback_loaded  = True

    # ── public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = VECTOR_TOP_K) -> tuple[str, bool]:
        """
        Returns (context_block: str, is_empty: bool).

        context_block contains k snippets formatted as:
          [Result 1 | score=0.87]
          <text>
          ---
        """
        if self._chroma is not None:
            return self._vector_retrieve(query, k)
        # graceful degradation: return full flat file as context
        is_empty = len(self._fallback_text.strip()) == 0
        return self._fallback_text, is_empty

    def source_label(self) -> str:
        return "ChromaDB (vector)" if self._chroma is not None else "footprints.txt (flat)"

    # ── private helpers ───────────────────────────────────────────────────────

    def _vector_retrieve(self, query: str, k: int) -> tuple[str, bool]:
        """Senin ChromaManager (query_memory) yapına göre güncellendi."""
        try:
            # Senin manager'ındaki fonksiyonu çağırıyoruz:
            documents = self._chroma.query_memory(query, n_results=k)

            if not documents or len(documents) == 0:
                return "", True

            lines: list[str] = ["=== VECTOR MEMORY (ChromaDB, k={}) ===".format(k)]
            for idx, doc in enumerate(documents, start=1):
                lines.append(f"\n[Result {idx}]")
                lines.append(doc)
                lines.append("---")

            lines.append("=== END VECTOR MEMORY ===")
            return "\n".join(lines), False
            
        except Exception as exc:
            return f"[ChromaDB query error: {exc}]", True

    @staticmethod
    def _load_flat_file() -> tuple[str, bool]:
        """Legacy flat-file fallback (footprints.txt)."""
        base        = os.path.dirname(__file__)
        masked_path = os.path.join(base, "data", "masked", "footprints_masked.txt")
        raw_path    = os.path.join(base, "data", "footprints.txt")

        if os.path.exists(masked_path):
            with open(masked_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            return content, len(content) == 0

        if os.path.exists(raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if not raw:
                return "", True
            if _MASKER_AVAILABLE:
                masker = PIIMasker()
                return masker.mask_data(raw), False
            return raw, False

        return "", True


# Module-level instance
VECTOR_MEM = VectorMemory(CHROMA)


# ─────────────────────────────────────────────────────────────────────────────
#  ONTOLOGY CONTEXT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _sparql(query: str) -> list[dict]:
    if ONTOLOGY_GRAPH is None:
        return []
    rows = []
    # rdflib'in yeni versiyonuna uygun sorgu işleme:
    qres = ONTOLOGY_GRAPH.query(query)
    for row in qres:
        # row.vars yerine qres.vars kullanıyoruz
        rows.append({str(v): str(row[v]) for v in qres.vars})
    return rows


def build_ontology_context() -> str:
    if ONTOLOGY_GRAPH is None:
        return "[ONTOLOGY: unavailable — rdflib not installed or TTL files not found]"

    lines = ["=== ONTOLOGY CONTEXT ==="]

    # Exam → Course
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

    # Person → Agent
    for r in _sparql("""
        PREFIX upper: <http://www.semanticweb.org/47ila/ontologies/2026/1/untitled-ontology-7/>
        SELECT ?person ?agent WHERE { ?person upper:hasAgent ?agent . }"""):
        person = r["person"].split("/")[-1].split("#")[-1]
        agent  = r["agent"].split("/")[-1].split("#")[-1]
        lines.append(f"  [Person] {person} hasAgent: {agent}")

    # Agent roles
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

    # PII mask tokens
    mask_texts = [
        r["text"] for r in _sparql("""
        PREFIX upper: <http://www.semanticweb.org/47ila/ontologies/2026/1/untitled-ontology-7/>
        SELECT ?text WHERE { ?label upper:maskText ?text . }""")
    ]
    if mask_texts:
        lines.append(f"  [PII Tokens] {', '.join(mask_texts)}")

    if len(lines) == 1:
        lines.append("  (No structured individuals found in ontology graph)")

    lines.append("=== END ONTOLOGY CONTEXT ===")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  GATE EVALUATORS  (C1–C8)
# ─────────────────────────────────────────────────────────────────────────────

def gate_c1_pii_masking(draft: str) -> tuple[bool, str]:
    """D1 — Scan draft for unmasked raw PII."""
    raw_id = re.search(r"\b\d{9,12}\b", draft)
    if raw_id:
        return False, f"Unmasked numeric ID at pos {raw_id.start()}: '{raw_id.group()}'"
    raw_email = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}", draft)
    if raw_email:
        return False, f"Unmasked email: '{raw_email.group()}'"
    return True, "No raw PII detected."


def gate_c2_memory_grounding(
    draft: str,
    vector_context: str,
    is_empty: bool,
) -> tuple[bool, str]:
    """
    D2 — Verify that the draft is grounded in the retrieved vector context.

    PASS conditions:
      • Vector context is non-empty (retrieval succeeded) — the LLM was
        instructed to answer only from it; hallucination absence is handled
        by C4.
      • Vector context is empty AND the draft contains the mandatory
        BlindSpot phrase "bulamadım".
    """
    if is_empty:
        if "bulamadım" in draft.lower():
            return True, "Vector memory empty; BlindSpot disclosure present."
        return False, "Vector memory empty but no BlindSpot disclosure in draft."

    # Non-empty: check at least one content word from the context appears in draft,
    # OR that the draft correctly issues a BlindSpot (query genuinely not in DB).
    if "bulamadım" in draft.lower():
        return True, "Draft issued BlindSpot — acceptable when query not in vector results."

    # Lightweight overlap check: pick a few words (>5 chars) from context
    context_words = {
        w.lower() for w in re.findall(r"\b\w{6,}\b", vector_context)
        if not re.match(r"\[.*_MASKED\]", w)      # exclude mask tokens
    }
    draft_words = {w.lower() for w in re.findall(r"\b\w{6,}\b", draft)}
    overlap = context_words & draft_words

    if len(overlap) >= 2:
        sample = list(overlap)[:4]
        return True, f"Draft shares {len(overlap)} content words with vector context (e.g. {sample})."

    return (
        False,
        "Draft does not appear grounded in retrieved vector context "
        "(overlap < 2 content words). Possible hallucination.",
    )


def gate_c3_ontology_compliance(draft: str) -> tuple[bool, str]:
    """D3 — Live SPARQL check: draft must not contradict ontology triples."""
    if ONTOLOGY_GRAPH is None:
        return True, "Ontology engine unavailable — provisional PASS."

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
            other = re.findall(r"\bcs\d{3}\b", draft, re.I)
            for oc in other:
                if oc.lower() != course_lbl.lower():
                    return False, (
                        f"Ontology violation: draft pairs '{exam_lbl}' with '{oc}' "
                        f"but ontology states it belongs to '{course_lbl}'."
                    )
    return True, "No ontology rule violations detected."


def gate_c4_hallucination(draft: str) -> tuple[bool, str]:
    """D4 — Detect weight-only / hallucinatory claim markers."""
    for label, pattern in ASP_NEG_PATTERNS:
        if label in ("ASP-NEG-02_HALLUCINATION", "ASP-NEG-05_WEIGHT_ONLY"):
            m = pattern.search(draft)
            if m:
                return False, f"[{label}] '{m.group()}'"
    return True, "No hallucination markers detected."


def gate_c5_role_permission(draft: str, agent_role: str) -> tuple[bool, str]:
    """D5 — Role-permission boundary (live lookup from ONTOLOGY_AGENT_ROLES)."""
    permitted = ONTOLOGY_AGENT_ROLES.get(agent_role, set())

    if re.search(r"tüm öğrencilerin notları|bütün öğrenciler", draft, re.I):
        if "read_all_student_grades" not in permitted:
            return False, f"'{agent_role}' lacks 'read_all_student_grades' permission."

    if re.search(r"dersi güncelle|ders planını değiştir", draft, re.I):
        if "manage_courses" not in permitted:
            return False, f"'{agent_role}' lacks 'manage_courses' permission."

    return True, f"Role '{agent_role}' — no boundary violations."


def gate_c6_anti_sycophancy(draft: str) -> tuple[bool, str]:
    """D6 — ASP-NEG pattern sweep."""
    violations = [
        f"[{lbl}] '{m.group()}'"
        for lbl, pat in ASP_NEG_PATTERNS
        if (m := pat.search(draft))
    ]
    if violations:
        return False, "ASP violations: " + "; ".join(violations)
    return True, "All ASP-NEG classifiers: NO_MATCH."


def gate_c7_blindspot(draft: str, is_empty: bool) -> tuple[bool, str]:
    """D7 — Ensure unanswerable queries carry a BlindSpot disclosure."""
    if is_empty and "bulamadım" not in draft.lower():
        return False, "Empty vector memory but BlindSpot phrase missing."
    return True, "BlindSpot completeness verified."


def gate_c8_redo_checksum() -> tuple[bool, str]:
    """D8 — No open REDO cycle (Zombi süreç kontrolü)."""
    # Eğer listede sadece 1 kayıt varsa ve o da şu anki aktif işlemse, buna izin ver.
    # Sadece geçmişten kalan, 'closed_at' damgası almamış zombi kayıtları engelle.
    open_cycles = [rec for rec in REDO_LOG if not rec.get("closed_at")]
    
    # Eğer birden fazla ucu açık işlem varsa veya garip bir durum varsa FAIL ver.
    if len(open_cycles) > 1:
        return False, f"Too many open REDO cycles: {len(open_cycles)}"
    
    return True, "Current REDO cycle in progress or clean state."


def evaluate_all_gates(
    draft: str,
    vector_context: str,
    is_empty: bool,
    query: str,
    agent_role: str = "StudentAgent",
) -> dict:
    """Execute C1∧C2∧…∧C8 and return a structured report."""
    c1p, c1e = gate_c1_pii_masking(draft)
    c2p, c2e = gate_c2_memory_grounding(draft, vector_context, is_empty)
    c3p, c3e = gate_c3_ontology_compliance(draft)
    c4p, c4e = gate_c4_hallucination(draft)
    c5p, c5e = gate_c5_role_permission(draft, agent_role)
    c6p, c6e = gate_c6_anti_sycophancy(draft)
    c7p, c7e = gate_c7_blindspot(draft, is_empty)
    c8p, c8e = gate_c8_redo_checksum()

    gates = {
        "C1": {"pass": c1p, "evidence": c1e},
        "C2": {"pass": c2p, "evidence": c2e},
        "C3": {"pass": c3p, "evidence": c3e},
        "C4": {"pass": c4p, "evidence": c4e},
        "C5": {"pass": c5p, "evidence": c5e},
        "C6": {"pass": c6p, "evidence": c6e},
        "C7": {"pass": c7p, "evidence": c7e},
        "C8": {"pass": c8p, "evidence": c8e},
    }
    return {"conjunction": all(g["pass"] for g in gates.values()), "gates": gates}


# ─────────────────────────────────────────────────────────────────────────────
#  REDO ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def open_redo(trigger_gate: str, failed_evidence: str) -> str:
    redo_id = str(uuid.uuid4())[:8]
    REDO_LOG.append({
        "redo_id":         redo_id,
        "trigger_gate":    trigger_gate,
        "failed_evidence": failed_evidence,
        "revision_action": None,
        "closure_gates":   {},
        "closed_at":       None,
    })
    return redo_id


def close_redo(redo_id: str, action: str, gate_results: dict) -> None:
    for rec in REDO_LOG:
        if rec["redo_id"] == redo_id:
            rec["revision_action"] = action
            rec["closure_gates"]   = {
                k: "PASS" if v["pass"] else "FAIL"
                for k, v in gate_results.items()
            }
            rec["closed_at"] = datetime.datetime.utcnow().isoformat()
            return


# ─────────────────────────────────────────────────────────────────────────────
#  BLINDSPOT DISCLOSURE BUILDER
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
#  4-STAGE VERIFICATION PIPELINE  (v2.2 — Vector Memory)
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    query: str,
    agent_role: str = "StudentAgent",
) -> str:
    """
    Execute the 4-stage verification pipeline.

    Stage 1 — RETRIEVAL & GROUNDING
        ChromaDB similarity search (k=15) + ontology context build.

    Stage 2 — DRAFT SYNTHESIS
        LLM drafts an answer grounded strictly in vector_context + ontology.

    Stage 3 — COMPLIANCE VERIFICATION  (Test Harness — C1–C8)
        Gate array evaluated; REDO loop if any gate fails (max 2 retries).

    Stage 4 — EMISSION
        BlindSpot prepended if needed; verified response returned.
    """

    # ── STAGE 1 — RETRIEVAL & GROUNDING ──────────────────────────────────────
    vector_context, is_empty = VECTOR_MEM.retrieve(query, k=VECTOR_TOP_K)
    ontology_context         = build_ontology_context()

    if is_empty:
        blindspot = build_blindspot_block(query, "VEKTÖr HAFIZA BOŞ")
        return blindspot + "Bunu hafızamda bulamadım."

    # ── STAGE 2 — DRAFT SYNTHESIS ─────────────────────────────────────────────
    user_message = (
        f"{vector_context}\n\n"
        f"{ontology_context}\n\n"
        f"ROLE: {agent_role}\n\n"
        f"SORU: {query}\n\n"
        "INSTRUCTION: Answer ONLY from the VECTOR MEMORY and ONTOLOGY CONTEXT "
        "above, using Turkish. "
        "Combine VECTOR MEMORY (dates, grades, specific records) with ONTOLOGY "
        "(course names, exam→course links, agent roles) for a complete answer. "
        "If the answer is NOT in either source, respond with exactly: "
        "\"Bunu hafızamda bulamadım.\" "
        "Do NOT hallucinate. Do NOT unmask or reproduce PII tokens as raw data. "
        "When using ontology inference, prefix with \"Akademik yapıya göre…\""
    )

    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    resp  = chat(model="llama3.2", messages=base_messages)
    draft = resp["message"]["content"].strip()

    # ── STAGE 3 — COMPLIANCE VERIFICATION (TEST HARNESS) ─────────────────────
    MAX_REDO       = 2
    active_redo_id: Optional[str] = None

    for attempt in range(MAX_REDO + 1):
        gate_report = evaluate_all_gates(
            draft, vector_context, is_empty, query, agent_role
        )

        if gate_report["conjunction"]:
            if active_redo_id:
                close_redo(
                    active_redo_id,
                    "Draft passed all gates after revision.",
                    gate_report["gates"],
                )
            break   # → Stage 4

        first_fail = next(
            (k for k, v in gate_report["gates"].items() if not v["pass"]),
            "UNKNOWN",
        )
        fail_ev = gate_report["gates"].get(first_fail, {}).get("evidence", "")

        if attempt == MAX_REDO:
            # REDO limit exhausted — emit safe refusal
            active_redo_id = open_redo(first_fail, fail_ev)
            blindspot = build_blindspot_block(query, f"REDO LIMIT ({first_fail} FAIL)")
            return (
                blindspot
                + f"⚠ Doğrulama başarısız (Gate {first_fail}). "
                  "Yanıt güvenli biçimde teslim edilemiyor.\n"
                  "Bunu hafızamda bulamadım."
            )

        # Open REDO and ask the LLM to self-correct
        active_redo_id = open_redo(first_fail, fail_ev)
        redo_instruction = (
            f"[REDO TRIGGERED — Gate {first_fail} FAILED]\n"
            f"Evidence: {fail_ev}\n"
            "Revise your previous draft to fix the failing dimension.\n"
            "Rules: Do NOT hallucinate. Do NOT unmask PII. "
            "Answer ONLY from VECTOR MEMORY and ONTOLOGY CONTEXT. "
            "If the answer is not there: \"Bunu hafızamda bulamadım.\""
        )
        redo_resp = chat(
            model="llama3.2",
            messages=base_messages + [
                {"role": "assistant", "content": draft},
                {"role": "user",      "content": redo_instruction},
            ],
        )
        draft = redo_resp["message"]["content"].strip()

    # ── STAGE 4 — EMISSION ────────────────────────────────────────────────────
    if BLINDSPOT_TRIGGERS.search(draft):
        return build_blindspot_block(query) + draft
    return draft


# ─────────────────────────────────────────────────────────────────────────────
#  DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_gate_report(gate_report: dict) -> None:
    conj = "✅ PASS" if gate_report["conjunction"] else "❌ FAIL"
    print(f"\n{'─' * 64}")
    print(f"  GATE ARRAY — C1∧C2∧…∧C8 = {conj}")
    print(f"{'─' * 64}")
    for gate, info in gate_report["gates"].items():
        status = "✅ PASS" if info["pass"] else "❌ FAIL"
        ev     = info["evidence"][:70]
        print(f"  {gate}: {status}  |  {ev}")
    print(f"{'─' * 64}\n")


def print_startup_banner() -> None:
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   COGNITWIN — Student Agent CLI  (v2.2)                      ║")
    print("║   Conjunctive Gate: C1∧C2∧C3∧C4∧C5∧C6∧C7∧C8               ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Memory backend status
    if CHROMA is not None:
        try:
            count = CHROMA.collection.count()
            print(f"  ✅ ChromaDB     : {count} records @ '{CHROMA_PATH}' (k={VECTOR_TOP_K})")
        except Exception:
            print(f"  ✅ ChromaDB     : connected @ '{CHROMA_PATH}' (k={VECTOR_TOP_K})")
    else:
        print("  ⚠  ChromaDB     : unavailable — falling back to footprints.txt")
        if not _CHROMA_AVAILABLE:
            print("     → Install: pip install chromadb")
            print("     → Ensure src/database/vector_store.py exists")

    # Ontology status
    if ONTOLOGY_GRAPH is not None:
        print(f"  ✅ Ontoloji     : {len(ONTOLOGY_GRAPH)} triple (cognitwin-upper + student)")
    else:
        print("  ⚠  Ontoloji     : unavailable")
        if not _RDFLIB_AVAILABLE:
            print("     → Install: pip install rdflib")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

VALID_ROLES = list(ONTOLOGY_AGENT_ROLES.keys())


def ask() -> None:
    print_startup_banner()
    print(f"\n  Geçerli roller : {', '.join(VALID_ROLES)}")
    print("  Komutlar       : /exit | /gates | /role <ROL> | /context\n")

    last_gate_report: dict = {}
    current_role = "StudentAgent"

    while True:
        try:
            q = input(f"[{current_role}] Soru > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nÇıkılıyor…")
            break

        if not q:
            continue

        # ── Built-in commands ─────────────────────────────────────────────────
        if q.lower() in ("/exit", "exit", "quit"):
            print("Oturum sonlandırıldı.")
            break

        if q.lower() == "/gates":
            print_gate_report(last_gate_report) if last_gate_report \
                else print("Henüz bir gate raporu yok.\n")
            continue

        if q.lower() == "/context":
            ctx, empty = VECTOR_MEM.retrieve(
                input("  Önizleme sorgusu > ").strip(), k=3
            )
            print(ctx if not empty else "  (boş sonuç)\n")
            continue

        if q.lower().startswith("/role "):
            requested = q.split(" ", 1)[1].strip()
            if requested in VALID_ROLES:
                current_role = requested
                print(f"  ✅ Rol: {current_role}\n")
            else:
                print(f"  ❌ Geçersiz rol. Seçenekler: {', '.join(VALID_ROLES)}\n")
            continue

        # ── Main pipeline ─────────────────────────────────────────────────────
        response = run_pipeline(q, agent_role=current_role)

        # Final gate evaluation for display (uses the last retrieved context)
        vec_ctx, vec_empty = VECTOR_MEM.retrieve(q, k=VECTOR_TOP_K)
        last_gate_report   = evaluate_all_gates(
            response, vec_ctx, vec_empty, q, current_role
        )

        print(f"\nYanıt:\n{response}\n")

        conj       = last_gate_report["conjunction"]
        symbol     = "✅" if conj else "❌"
        fail_gates = [k for k, v in last_gate_report["gates"].items() if not v["pass"]]

        if conj:
            print(f"  {symbol} Tüm doğrulama kapıları geçti (C1–C8).\n")
        else:
            print(f"  {symbol} Başarısız kapılar: {', '.join(fail_gates)} — /gates ile detay.\n")


if __name__ == "__main__":
    ask()