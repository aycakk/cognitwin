import re
import uuid
import datetime

# ─────────────────────────────────────────────────────────────────────────────
#  BÖLÜM 0 ▸ RUNTIME GATE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_PII_PATTERNS = [
    re.compile(r"\b\d{9,12}\b"),
    re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}"),
    re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
]

_ASP_NEG_PATTERNS = [
    ("ASP-NEG-01_PII_UNMASK",      re.compile(r"\b\d{8,11}\b")),
    ("ASP-NEG-02_HALLUCINATION",   re.compile(r"sanırım|galiba|muhtemelen|tahminim", re.I)),
    ("ASP-NEG-03_FALSE_PREMISE",   re.compile(r"haklısınız|evet,?\s+öyle\s+söylemiştim", re.I)),
    ("ASP-NEG-05_WEIGHT_ONLY",     re.compile(r"genel\s+bilgime\s+göre|eğitim\s+verilerime\s+göre", re.I)),
]

class StudentAgent:
    """
    CogniTwin Student Agent.
    All internal prompting is in English to ensure LLM stability and reasoning accuracy.
    """

    def __init__(self) -> None:
        self.role = "Universal Academic Student Persona"
        self.system_prompt = self._build_system_prompt()

    def format_message_with_memory(self, user_input: str, memory: str) -> list[dict]:
        """
        Dynamic RAG Prompt: Orchestrates the 'Academic Detective' reasoning.
        """
        user_content = (
            f"### AUTHORIZED ACADEMIC CONTEXT:\n{memory}\n\n"
            f"### STUDENT QUERY:\n{user_input}\n\n"
            "### EXECUTION INSTRUCTIONS:\n"
            "1. Act as an academic detective. Meticulously analyze the provided context.\n"
            "2. Resolve semantic links between naming variations (e.g., 'Yusuf Hoca' vs 'Yusuf Altunel').\n"
            "3. If an email address is linked to a name in the context, treat them as the same entity.\n"
            "4. Synthesize an answer if information is indirectly available through deductive reasoning.\n"
            "5. If and ONLY if the context is completely irrelevant, state: 'Bunu hafızamda bulamadım.'\n"
            "6. Stay strictly grounded within the context. Do not hallucinate external facts."
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_content},
        ]

    # ─────────────────────────────────────────────────────────────────────────────
    #  BÖLÜM 1 ▸ CORE SYSTEM PROMPT (ENGLISH)
    # ─────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_system_prompt() -> str:
        """
        The fundamental behavioral directive for the LLM.
        """
        return (
            "You are the CogniTwin Student Agent, an advanced RAG-based academic assistant. "
            "Your primary mission is to provide accurate information based EXCLUSIVELY on the "
            "provided MEMORY and ONTOLOGY records. You must maintain professional academic "
            "integrity, avoid unmasked PII, and ensure all claims are grounded in evidence. "
            "If the query cannot be answered using the provided records, trigger the BlindSpot protocol."
        )

    # ─────────────────────────────────────────────────────────────────────────────
    #  BÖLÜM 2 ▸ VERIFICATION GATES (C1-C8)
    # ─────────────────────────────────────────────────────────────────────────────

    def gate_report(self, draft: str, memory: str = "") -> dict:
        """Evaluates output safety and grounding across multiple checkpoints."""
        gates = {
            "C1": self._gate_c1(draft),     # PII Safety
            "C2": self._gate_c2(draft, memory), # Grounding
            "C4": self._gate_c4(draft),     # Hallucination
            "C7": self._gate_c7(draft, memory), # Memory Consistency
        }
        return {"conjunction": all(g[0] for g in gates.values()), "gates": gates}

    def _gate_c1(self, draft: str) -> tuple[bool, str]:
        for pattern in _PII_PATTERNS:
            if pattern.search(draft):
                return False, "PII Leakage Detected"
        return True, "PII Scan Passed"

    def _gate_c7(self, draft: str, memory: str) -> tuple[bool, str]:
        if not memory.strip() and "bulamadım" not in draft.lower():
            return False, "C7 Fail: Agent answered without memory evidence."
        return True, "C7 Pass: Proper handling of empty memory."

    @staticmethod
    def build_blindspot_block(query: str) -> str:
        """Visual disclosure for missing academic records."""
        return (
            "\n┌──────────────────────────────────────────────────────────────────┐\n"
            "│  ⚠  BLINDSPOT DISCLOSURE                                         │\n"
            f"│  Query      : {query[:50]:<48} │\n"
            "│  Resolution : No matching academic records found in memory.      │\n"
            "└──────────────────────────────────────────────────────────────────┘\n"
        )