"""
CogniTwin Query Router
======================
Gelen sorguyu üç adımda değerlendirip hangi yanıt stratejisinin
kullanılacağına karar verir.

Karar sırası
------------
1. Ontoloji  → SPARQL eşleşmesi var ve yeterince güvenliyse ONTOLOGY_DIRECT
2. Bellek    → ChromaDB'de kişisel kayıt varsa MEMORY_RAG
3. Fallback  → Her ikisi de boşsa MODEL_FALLBACK
+  Görev     → "yap / oluştur / gönder" gibi bir eylem komutu ise TASK
               (Aşama-2'de tool agent'a yönlendirilecek)

Diğer modüller sadece RoutingDecision.route'u okur;
strateji içeri gizlidir.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# Route enum
# ─────────────────────────────────────────────────────────────────────────────

class Route(str, Enum):
    ONTOLOGY_DIRECT = "ontology_direct"
    """Ontolojiden doğrudan yanıt — LLM çağrısı yapılmaz."""

    MEMORY_RAG      = "memory_rag"
    """Kişisel bellek bağlamı ile LLM yanıtı."""

    MODEL_FALLBACK  = "model_fallback"
    """Ne ontoloji ne bellek eşleşmedi; LLM kendi bilgisiyle yanıtlar."""

    TASK            = "task"
    """Eylem/görev komutu — Aşama-2'de tool agent'a yönlendirilecek."""


# ─────────────────────────────────────────────────────────────────────────────
# RoutingDecision — router'ın döndürdüğü sonuç paketi
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    route: Route

    # Ontoloji katmanından gelen ham triple metni (varsa)
    ontology_context: str = ""
    ontology_confidence: float = 0.0   # 0.0 – 1.0

    # ChromaDB katmanından gelen bellek parçaları (varsa)
    memory_lines: list[str] = field(default_factory=list)
    memory_confidence: float = 0.0    # 0.0 – 1.0

    # Tespit edilen görev türü (route==TASK ise dolu)
    task_type: str = ""               # "calendar_add", "file_read", vb.

    @property
    def has_ontology(self) -> bool:
        return bool(self.ontology_context.strip())

    @property
    def has_memory(self) -> bool:
        return bool(self.memory_lines)

    @property
    def memory_text(self) -> str:
        return "\n".join(self.memory_lines)


# ─────────────────────────────────────────────────────────────────────────────
# Eşik sabitleri — gerekirse .env üzerinden override edilebilir
# ─────────────────────────────────────────────────────────────────────────────

ONTOLOGY_DIRECT_THRESHOLD = 0.75   # Bu skorun üstünde LLM atlanır
MEMORY_MIN_LINES          = 1      # En az kaç bellek satırı gerekli


# ─────────────────────────────────────────────────────────────────────────────
# Görev (TASK) tetikleyicileri
# ─────────────────────────────────────────────────────────────────────────────

_TASK_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("calendar_add",  re.compile(
        r"\b(takvime\s+\w+(\s+\w+)?\s*ekle|takvime\s+ekle|etkinlik\s+oluştur|randevu\s+koy|remind\s+me)\b", re.I)),
    ("calendar_read", re.compile(
        r"\b(bugün\s+ne\s+var|yarın\s+ne\s+var|takvimim|program[ıi]m|schedule)\b", re.I)),
    ("file_read",     re.compile(
        r"\b(dosyayı\s+oku|dosya\s+aç|open\s+file|read\s+file)\b", re.I)),
    ("gmail_read",    re.compile(
        r"(mailleri\w*\s+göster|gelen\s+kutusu|inbox|son\s+mailler\w*|e.?postalar\w*\s+göster)", re.I)),
    ("gmail_send",    re.compile(
        r"(mail\w*\s+gönder|mail\w*\s+yaz|e.?posta\w*\s+gönder|e.?posta\w*\s+yaz|send\s+(an?\s+)?email)", re.I)),
    ("lms_check",     re.compile(
        r"\b(ödev\s+durumu|teslim\s+edildi\s+mi|lms|moodle)\b", re.I)),
]


# ─────────────────────────────────────────────────────────────────────────────
# QueryRouter
# ─────────────────────────────────────────────────────────────────────────────

class QueryRouter:
    """
    pipeline.py tarafından çağrılır; RoutingDecision döndürür.

    Kullanım::

        router = QueryRouter(ontology_fn=..., memory_fn=...)
        decision = router.route(query="COM8090 vize tarihi ne zaman?",
                                user_id="ilayda_karahan")

        if decision.route == Route.ONTOLOGY_DIRECT:
            return decision.ontology_context        # LLM yok
        elif decision.route == Route.MEMORY_RAG:
            return llm(context=decision.memory_text, query=query)
        elif decision.route == Route.MODEL_FALLBACK:
            return llm(query=query)
        elif decision.route == Route.TASK:
            return tool_agent.run(decision.task_type, query)
    """

    def __init__(
        self,
        ontology_lookup_fn,   # (query: str) -> tuple[str, float]
        memory_search_fn,     # (query: str, user_id: str) -> list[str]
    ) -> None:
        self._ont_fn = ontology_lookup_fn
        self._mem_fn = memory_search_fn

    # ── Ana giriş noktası ────────────────────────────────────────────────────

    def route(self, query: str, user_id: str) -> RoutingDecision:
        """
        Sorguyu değerlendirir ve RoutingDecision döndürür.

        Adımlar:
          1) Görev komutu mu?          → TASK
          2) Ontoloji yüksek güven?    → ONTOLOGY_DIRECT
          3) Bellek eşleşmesi var mı?  → MEMORY_RAG (ontoloji bağlam olarak eklenir)
          4) Her şey boş              → MODEL_FALLBACK
        """

        # Adım 1 — görev tespiti
        task_type = self._detect_task(query)
        if task_type:
            return RoutingDecision(route=Route.TASK, task_type=task_type)

        # Adım 2 — ontoloji kontrolü
        ont_context, ont_score = self._ont_fn(query)

        if ont_context and ont_score >= ONTOLOGY_DIRECT_THRESHOLD:
            return RoutingDecision(
                route=Route.ONTOLOGY_DIRECT,
                ontology_context=ont_context,
                ontology_confidence=ont_score,
            )

        # Adım 3 — bellek kontrolü
        mem_lines = self._mem_fn(query, user_id)

        if len(mem_lines) >= MEMORY_MIN_LINES:
            return RoutingDecision(
                route=Route.MEMORY_RAG,
                ontology_context=ont_context,        # düşük güvenli de olsa bağlam
                ontology_confidence=ont_score,
                memory_lines=mem_lines,
                memory_confidence=self._mem_score(mem_lines),
            )

        # Adım 4 — fallback
        return RoutingDecision(
            route=Route.MODEL_FALLBACK,
            ontology_context=ont_context,
            ontology_confidence=ont_score,
        )

    # ── Yardımcı metotlar ────────────────────────────────────────────────────

    @staticmethod
    def _detect_task(query: str) -> str:
        """Sorgu bir eylem komutu içeriyorsa görev türünü döndürür."""
        for task_name, pattern in _TASK_PATTERNS:
            if pattern.search(query):
                return task_name
        return ""

    @staticmethod
    def _mem_score(lines: list[str]) -> float:
        """Basit bellek güven skoru: satır sayısına göre 0.0–1.0."""
        return min(1.0, len(lines) / 5.0)
