"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   student_agent.py — CogniTwin Hybrid Intelligence Agent  (v3.0)           ║
║                                                                              ║
║   Architecture  : Hybrid (Unstructured Memory + Structured Ontology)        ║
║   Memory Source : ChromaDB  (vector similarity, k=15)                       ║
║   Ontology      : cognitwin-upper.ttl  +  student_ontology.ttl  (rdflib)    ║
║   Gate Array    : C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5 ∧ C6 ∧ C7 ∧ C8                  ║
║   Response Lang : Turkish  (internal reasoning in English for LLM stability)║
╚══════════════════════════════════════════════════════════════════════════════╝

Design principles
─────────────────
• No internal jargon (PII, Gate, Redo, Checksum…) leaks into user-facing text.
• No English filler words in Turkish responses.
• Ontology-derived answers are prefixed with "Akademik yapıya göre…"
• Memory-derived answers cite the record directly.
• Both sources can be combined (C4 synthesis).
• Missing from both → "Bunu hafızamda bulamadım." + BlindSpot block.
"""

from __future__ import annotations

import re
import uuid
import datetime
from typing import Optional

from src.shared.permissions import ONTOLOGY_AGENT_ROLES
from src.shared.patterns import (
    PII_PATTERNS as _PII_PATTERNS,
    ASP_NEG_PATTERNS as _ASP_NEG_PATTERNS,
)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 0 ▸  INTERNAL PATTERN REGISTRY
#  PII / ASP-NEG patterns are now imported from src.shared.patterns
#  (single source of truth — previously duplicated with pipeline.py).
#  Only agent-local patterns remain below.
# ─────────────────────────────────────────────────────────────────────────────

# --- Internal-jargon leak patterns (C8 / stability) -------------------------
_JARGON_LEAK_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bPII\b"),
    re.compile(r"\bgate\b", re.I),
    re.compile(r"\bredo\b", re.I),
    re.compile(r"\bchecksum\b", re.I),
    re.compile(r"\bC[1-8]\b"),                   # "C1", "C2" … "C8"
    re.compile(r"\bASP-NEG\b"),
    re.compile(r"\bblindspot\b", re.I),
    re.compile(r"\bontology\b", re.I),           # structural term — keep internal
    re.compile(r"\bsparql\b", re.I),
    re.compile(r"\bchromadb\b", re.I),
    re.compile(r"\bobtained\b", re.I),
    re.compile(r"\brequired\b", re.I),
]

# --- Hallucination trigger (blindspot detection) ----------------------------
_BLINDSPOT_TRIGGER = re.compile(
    r"hafızamda\s+bulamadım|bilmiyorum|emin\s+değilim|kayıt\s+yok",
    re.I,
)

# --- Ontology-inference prefix that must appear when source is ontology ------
_ONTOLOGY_PREFIX = "Akademik yapıya göre"


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 ▸  STUDENT AGENT
# ─────────────────────────────────────────────────────────────────────────────

class StudentAgent:
    """
    CogniTwin Hybrid Intelligence — Student Agent.

    Bridges unstructured ChromaDB memory with structured ontology reasoning
    to answer academic queries with full gate-array compliance.

    Usage
    -----
    agent = StudentAgent()

    # Build the message list for the LLM
    messages = agent.format_message_with_memory(
        user_input       = "CS101 dersinin sınavı ne zaman?",
        memory           = vector_context_string,     # from VectorMemory.retrieve()
        ontology_context = ontology_context_string,   # from build_ontology_context()
    )

    # After getting the LLM draft, run gate validation
    report = agent.gate_report(draft, memory=vector_context_string)
    """

    # ---------------------------------------------------------------------- #
    #  Constructor                                                             #
    # ---------------------------------------------------------------------- #

    def __init__(self, agent_role: str = "StudentAgent") -> None:
        """
        Parameters
        ----------
        agent_role : str
            One of StudentAgent | InstructorAgent |
            HeadOfDepartmentAgent | ResearcherAgent.
            Used by C5 (role-permission gate).
        """
        self.agent_role    = agent_role
        self.system_prompt = self._build_system_prompt()

    # ---------------------------------------------------------------------- #
    #  Public: message builder                                                 #
    # ---------------------------------------------------------------------- #

    def format_message_with_memory(
        self,
        user_input: str,
        memory: str,
        ontology_context: str,
    ) -> list[dict]:
        """
        Construct the full message list for the LLM.

        This is the 'Academic Detective' prompt that instructs the model to:
          1. Treat MEMORY (ChromaDB snippets) as the primary factual source.
          2. Treat ONTOLOGY CONTEXT as the structural / relational source.
          3. Resolve name variations across both sources.
          4. Synthesize a single, coherent Turkish answer.
          5. Prefix ontology-inferred facts with 'Akademik yapıya göre…'
          6. Refuse gracefully when neither source contains the answer.

        Parameters
        ----------
        user_input       : Raw Turkish query from the student.
        memory           : Formatted vector-memory block (k=15 ChromaDB snippets).
        ontology_context : SPARQL-derived ontology summary string.

        Returns
        -------
        list[dict] — ready to pass directly to `ollama.chat(messages=…)`.
        """
        memory_block   = memory.strip()   if memory.strip()   else "(boş — ilgili kayıt bulunamadı)"
        ontology_block = ontology_context.strip() if ontology_context.strip() else "(boş)"

        user_content = (
            # ── Source A: unstructured memory ────────────────────────────────
            "### SOURCE A — ACADEMIC MEMORY RECORDS\n"
            f"{memory_block}\n\n"

            # ── Source B: structured ontology ────────────────────────────────
            "### SOURCE B — ACADEMIC STRUCTURE (Roles, Courses, Exams)\n"
            f"{ontology_block}\n\n"

            # ── Query ────────────────────────────────────────────────────────
            f"### STUDENT QUERY\n{user_input}\n\n"

            # ── Reasoning instructions ────────────────────────────────────────
            "### REASONING INSTRUCTIONS\n"

            # Step 1 — Detective scan
            "STEP 1 — ENTITY RESOLUTION\n"
            "  Before answering, resolve all naming variations between the two sources:\n"
            "  • Informal forms in memory (e.g. 'Yusuf Hoca', 'Yusuf bey') must be\n"
            "    matched to formal ontology entries (e.g. 'Yusuf Altunel', 'InstructorAgent').\n"
            "  • An e-mail address linked to a name in memory makes them the same entity.\n"
            "  • Course code abbreviations (e.g. 'CS101') link to their full ontology individual.\n\n"

            # Step 2 — Synthesis
            "STEP 2 — HYBRID SYNTHESIS\n"
            "  • Use SOURCE A for specific facts: dates, times, grades, deadlines, messages.\n"
            "  • Use SOURCE B for structural facts: who teaches what, exam→course links,\n"
            "    agent roles (Instructor, Student, Head of Department).\n"
            "  • When facts from both sources complement each other, combine them into\n"
            "    a single unified answer.\n\n"

            # Step 3 — Prefix rule
            "STEP 3 — RESPONSE PREFIX RULE\n"
            "  • If the answer comes ONLY from SOURCE B (ontology inference),\n"
            f"    start the response with: '{_ONTOLOGY_PREFIX}…'\n"
            "  • If the answer comes from SOURCE A, or from both A and B,\n"
            "    do NOT use that prefix — answer directly.\n\n"

            # Step 4 — Blindspot
            "STEP 4 — BLINDSPOT PROTOCOL\n"
            "  • If and ONLY IF neither source contains relevant information,\n"
            "    respond with EXACTLY this sentence and nothing more:\n"
            "    'Bunu hafızamda bulamadım.'\n\n"

            # Step 5 — Output constraints
            "STEP 5 — OUTPUT CONSTRAINTS\n"
            "  • Write the final answer in TURKISH.\n"
            "  • Be concise — 1 to 4 sentences is ideal.\n"
            "  • Do NOT mention technical systems, file names, database names,\n"
            "    or internal processing steps. The user must see only the answer.\n"
            "  • Do NOT use English filler words (obtained, required, provided, etc.).\n"
            "  • Do NOT reproduce any masked token (e.g. [EMAIL_MASKED]) as raw data.\n"
            "  • Do NOT speculate. Every factual claim must trace to SOURCE A or B."
        )

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_content},
        ]

    # ---------------------------------------------------------------------- #
    #  Public: gate validation                                                 #
    # ---------------------------------------------------------------------- #

    def gate_report(
        self,
        draft: str,
        memory: str = "",
        ontology_context: str = "",
    ) -> dict:
        """
        Run the full C1–C8 conjunctive gate array against a draft response.

        Returns
        -------
        {
          "conjunction": bool,       # True only if ALL gates pass
          "gates": {
              "C1": (bool, str),     # (pass, evidence)
              …
              "C8": (bool, str),
          }
        }
        """
        gates = {
            "C1": self._gate_c1_pii(draft),
            "C2": self._gate_c2_grounding(draft, memory, ontology_context),
            "C3": self._gate_c3_ontology_prefix(draft, ontology_context),
            "C4": self._gate_c4_synthesis(draft),
            "C5": self._gate_c5_role_permission(draft),
            "C6": self._gate_c6_anti_sycophancy(draft),
            "C7": self._gate_c7_blindspot(draft, memory, ontology_context),
            "C8": self._gate_c8_jargon_stability(draft),
        }
        return {
            "conjunction": all(v[0] for v in gates.values()),
            "gates": gates,
        }

    # ---------------------------------------------------------------------- #
    #  Public: blindspot disclosure block                                      #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def build_blindspot_block(query: str) -> str:
        """
        Produce a user-visible disclosure box when neither memory
        nor ontology can answer the query.

        The box deliberately avoids internal jargon — it uses
        plain academic language the student can understand.
        """
        q_short = query[:52]
        return (
            "\n┌──────────────────────────────────────────────────────────────────┐\n"
            "│  ⚠  Akademik Kayıt Bulunamadı                                    │\n"
            f"│  Sorgu   : {q_short:<54} │\n"
            "│  Durum   : Hafıza kayıtlarında eşleşen bilgi yok.               │\n"
            "│  Öneri   : Akademik danışmanınız veya kayıt birimiyle iletişime  │\n"
            "│            geçiniz.                                               │\n"
            "└──────────────────────────────────────────────────────────────────┘\n"
        )

    # ---------------------------------------------------------------------- #
    #  Public: helpers                                                         #
    # ---------------------------------------------------------------------- #

    def set_role(self, role: str) -> None:
        """Switch agent role at runtime (affects C5 permission gate)."""
        valid = {
            "StudentAgent", "InstructorAgent",
            "HeadOfDepartmentAgent", "ResearcherAgent",
        }
        if role not in valid:
            raise ValueError(f"Unknown role '{role}'. Valid roles: {valid}")
        self.agent_role = role

    # ---------------------------------------------------------------------- #
    #  Private: system prompt                                                  #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _build_system_prompt() -> str:
        """
        Core behavioral directive for the LLM.

        Written in English for maximum LLM instruction-following stability.
        The user-facing response is still Turkish (enforced in the user turn).
        """
        return (
            "You are the CogniTwin Academic Assistant, a high-integrity RAG-based agent "
            "embedded in a university information system.\n\n"

            "KNOWLEDGE SOURCES\n"
            "You have access to exactly two authoritative sources:\n"
            "  1. ACADEMIC MEMORY RECORDS — masked personal notes, emails, and logs "
            "retrieved from a vector database (ChromaDB) via semantic similarity search. "
            "These contain concrete facts: dates, grades, assignment deadlines, "
            "and conversational records.\n"
            "  2. ACADEMIC STRUCTURE — a formal knowledge graph (OWL/RDF ontology) "
            "that defines roles (Instructor, Student, Head of Department), "
            "course hierarchies, exam-to-course relationships, and agent classes. "
            "Source files: cognitwin-upper.ttl and student_ontology.ttl.\n\n"

            "ABSOLUTE RULES\n"
            "  • Answer ONLY from the two sources above. Never use outside knowledge.\n"
            "  • Never reveal raw identifiers, e-mail addresses, or phone numbers "
            "that appear in memory records — treat all PII tokens as confidential.\n"
            "  • When the ontology is the sole source of an answer, begin the "
            f"response with: '{_ONTOLOGY_PREFIX}…'\n"
            "  • When neither source answers the query, respond with exactly: "
            "'Bunu hafızamda bulamadım.' — nothing more.\n"
            "  • All user-facing responses must be in TURKISH.\n"
            "  • Do not mention databases, ontologies, file names, or system internals "
            "in the answer. The student must receive only academic information.\n"
            "  • Be concise. Do not hedge or speculate."
        )

    # ---------------------------------------------------------------------- #
    #  Private: individual gate implementations                                #
    # ---------------------------------------------------------------------- #

    # C1 — Privacy / PII masking
    def _gate_c1_pii(self, draft: str) -> tuple[bool, str]:
        """Scan draft for any unmasked PII token."""
        for pattern in _PII_PATTERNS:
            m = pattern.search(draft)
            if m:
                return False, f"Gizli veri sızıntısı tespit edildi: '{m.group()}'"
        return True, "Gizlilik taraması geçti."

    # C2 — Grounding (memory + ontology)
    def _gate_c2_grounding(
        self,
        draft: str,
        memory: str,
        ontology_context: str,
    ) -> tuple[bool, str]:
        """
        Verify the draft is grounded in at least one source.

        PASS if:
          • Draft contains ≥ 2 content words (6+ chars) shared with memory OR ontology.
          • Draft correctly issues the BlindSpot phrase when both sources are empty.
        """
        both_empty = (not memory.strip()) and (not ontology_context.strip())

        if both_empty:
            if "bulamadım" in draft.lower():
                return True, "Her iki kaynak boş; BlindSpot bildirimi mevcut."
            return False, "Her iki kaynak boş ancak BlindSpot bildirimi eksik."

        if "bulamadım" in draft.lower():
            # Acceptable: query genuinely not in either source
            return True, "Kaynaklarda bilgi yok; BlindSpot bildirimi verildi."

        # Lightweight lexical overlap across both sources
        combined_source = memory + " " + ontology_context
        source_words = {
            w.lower() for w in re.findall(r"\b\w{6,}\b", combined_source)
            if not re.match(r"\[.*_MASKED\]", w)
        }
        draft_words = {w.lower() for w in re.findall(r"\b\w{6,}\b", draft)}
        overlap = source_words & draft_words

        if len(overlap) >= 2:
            sample = sorted(overlap)[:4]
            return True, f"Kaynak bağlantısı doğrulandı (örtüşen terimler: {sample})."

        return (
            False,
            "Taslak yanıt, kaynakla yeterli sözcük örtüşmesi sağlamıyor "
            "(< 2 ortak terim). Olası üretim hatası.",
        )

    # C3 — Ontology reasoning prefix
    def _gate_c3_ontology_prefix(
        self,
        draft: str,
        ontology_context: str,
    ) -> tuple[bool, str]:
        """
        If the ontology context is non-empty AND the draft does not reference
        any memory content (likely ontology-only answer), it must carry the
        required prefix.

        Heuristic: if the draft contains any ontology individual name
        (extracted from the context block) but no memory date patterns,
        we expect the prefix.
        """
        if not ontology_context.strip():
            return True, "Ontoloji bağlamı boş — önek kontrolü atlandı."

        # Extract short labels from ontology context (e.g. 'midterm1', 'CS101', 'ayse')
        onto_labels = re.findall(r":\s+(\w+)", ontology_context)
        onto_labels += re.findall(r"\[(?:Exam|Role|Person)\]\s+(\w+)", ontology_context)

        draft_lower = draft.lower()
        memory_date_present = bool(re.search(r"\d{1,2}[./]\d{1,2}[./]\d{2,4}", draft))

        onto_match = any(lbl.lower() in draft_lower for lbl in onto_labels if len(lbl) > 2)

        if onto_match and not memory_date_present:
            if _ONTOLOGY_PREFIX.lower() not in draft_lower:
                return (
                    False,
                    f"Ontoloji çıkarımı tespit edildi ancak "
                    f"'{_ONTOLOGY_PREFIX}…' öneki eksik.",
                )

        return True, "Ontoloji önek kuralı karşılandı."

    # C4 — Synthesis (hallucination absence)
    def _gate_c4_synthesis(self, draft: str) -> tuple[bool, str]:
        """Detect hallucination / weight-only claim markers."""
        for label, pattern in _ASP_NEG_PATTERNS:
            if label in ("ASP-NEG-02_HALLUCINATION", "ASP-NEG-05_WEIGHT_ONLY"):
                m = pattern.search(draft)
                if m:
                    return False, f"Kanıtsız ifade tespit edildi: '{m.group()}'"
        return True, "Sentez doğrulaması geçti."

    # C5 — Role-permission boundary
    def _gate_c5_role_permission(self, draft: str) -> tuple[bool, str]:
        """
        Coarse permission check derived from ONTOLOGY_AGENT_ROLES.
        StudentAgent may not see bulk grade lists or course management actions.
        """
        permitted = ONTOLOGY_AGENT_ROLES.get(self.agent_role, set())

        if re.search(r"tüm öğrencilerin notları|bütün öğrenciler", draft, re.I):
            if "read_all_student_grades" not in permitted:
                return False, f"'{self.agent_role}' rolü toplu not erişimine yetkili değil."

        if re.search(r"dersi güncelle|ders planını değiştir", draft, re.I):
            if "manage_courses" not in permitted:
                return False, f"'{self.agent_role}' rolü ders yönetimine yetkili değil."

        return True, f"'{self.agent_role}' rol sınırı ihlali yok."

    # C6 — Anti-sycophancy
    def _gate_c6_anti_sycophancy(self, draft: str) -> tuple[bool, str]:
        """Sweep all ASP-NEG patterns."""
        violations = [
            f"'{m.group()}'"
            for label, pat in _ASP_NEG_PATTERNS
            if (m := pat.search(draft))
        ]
        if violations:
            return False, "Onay güdümlü ifade tespit edildi: " + ", ".join(violations)
        return True, "Onay denetimi geçti."

    # C7 — BlindSpot completeness
    def _gate_c7_blindspot(
        self,
        draft: str,
        memory: str,
        ontology_context: str,
    ) -> tuple[bool, str]:
        """
        When both sources are empty, the draft MUST contain the BlindSpot phrase.
        When sources are present but the draft claims it cannot answer,
        we still accept that as a valid outcome (query not in 129 records).
        """
        both_empty = (not memory.strip()) and (not ontology_context.strip())

        if both_empty and "bulamadım" not in draft.lower():
            return (
                False,
                "Her iki kaynak boş olmasına rağmen BlindSpot bildirimi verilmedi.",
            )
        return True, "BlindSpot bütünlüğü doğrulandı."

    # C8 — Stability / no jargon leak
    def _gate_c8_jargon_stability(self, draft: str) -> tuple[bool, str]:
        """
        Ensure no internal technical jargon bleeds into the final response.
        Also catches runaway / excessively long drafts that suggest a REDO loop.
        """
        # Jargon leak check
        for pattern in _JARGON_LEAK_PATTERNS:
            m = pattern.search(draft)
            if m:
                return (
                    False,
                    f"Teknik jargon sızıntısı tespit edildi: '{m.group()}' "
                    "— bu terim kullanıcıya gösterilmemeli.",
                )

        # Runaway length check (> 800 chars likely indicates uncontrolled generation)
        if len(draft) > 800:
            return (
                False,
                f"Yanıt çok uzun ({len(draft)} karakter). "
                "Kısa ve öz bir yanıt yeniden üretilmeli.",
            )

        return True, "Kararlılık ve jargon denetimi geçti."


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 ▸  CONVENIENCE: REDO LOG  (used by main_cli.py)
# ─────────────────────────────────────────────────────────────────────────────

class RedoLog:
    """
    Lightweight in-session audit trail for REDO cycles.
    Instantiate once per session and pass to the pipeline.
    """

    def __init__(self) -> None:
        self._log: list[dict] = []

    def open(self, trigger_gate: str, evidence: str) -> str:
        redo_id = str(uuid.uuid4())[:8]
        self._log.append({
            "redo_id":         redo_id,
            "trigger_gate":    trigger_gate,
            "failed_evidence": evidence,
            "revision_action": None,
            "closure_gates":   {},
            "closed_at":       None,
        })
        return redo_id

    def close(self, redo_id: str, action: str, gate_results: dict) -> None:
        for rec in self._log:
            if rec["redo_id"] == redo_id:
                rec["revision_action"] = action
                rec["closure_gates"]   = {
                    k: "PASS" if v[0] else "FAIL"
                    for k, v in gate_results.items()
                }
                rec["closed_at"] = datetime.datetime.utcnow().isoformat()
                return

    def has_open(self) -> bool:
        return any(r["closed_at"] is None for r in self._log)

    def open_ids(self) -> list[str]:
        return [r["redo_id"] for r in self._log if r["closed_at"] is None]

    def full_log(self) -> list[dict]:
        return list(self._log)