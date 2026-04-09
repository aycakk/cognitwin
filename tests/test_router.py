"""
Router birim testleri
=====================
Çalıştırmak için:
    cd CogniTwin
    python -m pytest tests/test_router.py -v

Ollama veya ChromaDB bağlantısı GEREKMEZ — adaptörler mock'lanır.
"""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.api.router import QueryRouter, Route, ONTOLOGY_DIRECT_THRESHOLD

# pytest varsa kullan, yoksa unittest ile çalış
try:
    import pytest
    _parametrize = pytest.mark.parametrize
except ImportError:
    pytest = None  # type: ignore
    _parametrize = lambda *a, **k: (lambda f: f)  # no-op decorator


# ─────────────────────────────────────────────────────────────────────────────
# Mock adaptörler
# ─────────────────────────────────────────────────────────────────────────────

def _ont_high(query: str):
    """Ontoloji yüksek güven döndürür → ONTOLOGY_DIRECT beklenir."""
    return ("COM8090 vize tarihi: 2026-04-15", ONTOLOGY_DIRECT_THRESHOLD + 0.1)

def _ont_low(query: str):
    """Ontoloji düşük güven döndürür → geçmez."""
    return ("", 0.2)

def _ont_none(query: str):
    return ("", 0.0)

def _mem_has(query: str, user_id: str):
    """Bellekte kayıt var."""
    return ["Ben genellikle ödevleri Perşembe akşamı teslim ederim."]

def _mem_empty(query: str, user_id: str):
    """Bellek boş."""
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Test senaryoları
# ─────────────────────────────────────────────────────────────────────────────

class TestOntologyDirect(unittest.TestCase):
    def test_high_confidence_skips_llm(self):
        router = QueryRouter(_ont_high, _mem_empty)
        d = router.route("COM8090 vize ne zaman?", "ilayda")
        self.assertEqual(d.route, Route.ONTOLOGY_DIRECT)
        self.assertTrue(d.has_ontology)
        self.assertGreater(d.ontology_confidence, ONTOLOGY_DIRECT_THRESHOLD)

    def test_result_has_content(self):
        router = QueryRouter(_ont_high, _mem_empty)
        d = router.route("vize tarihi?", "ilayda")
        self.assertIn("COM8090", d.ontology_context)


class TestMemoryRAG(unittest.TestCase):
    def test_memory_hit_with_low_ontology(self):
        router = QueryRouter(_ont_low, _mem_has)
        d = router.route("ödevleri ne zaman teslim ederim?", "ilayda")
        self.assertEqual(d.route, Route.MEMORY_RAG)
        self.assertTrue(d.has_memory)
        self.assertGreaterEqual(len(d.memory_lines), 1)

    def test_memory_rag_includes_ontology_context(self):
        router = QueryRouter(_ont_low, _mem_has)
        d = router.route("bir şey sor", "ilayda")
        self.assertIsInstance(d.ontology_context, str)


class TestModelFallback(unittest.TestCase):
    def test_both_empty_gives_fallback(self):
        router = QueryRouter(_ont_none, _mem_empty)
        d = router.route("Türkiye'nin başkenti neresi?", "ilayda")
        self.assertEqual(d.route, Route.MODEL_FALLBACK)
        self.assertFalse(d.has_ontology)
        self.assertFalse(d.has_memory)


class TestTaskDetection(unittest.TestCase):
    def _check(self, query, expected_task):
        router = QueryRouter(_ont_none, _mem_empty)
        d = router.route(query, "ilayda")
        self.assertEqual(d.route, Route.TASK, f"Beklenen TASK, gelen: {d.route}")
        self.assertEqual(d.task_type, expected_task,
            f"Beklenen task_type={expected_task}, gelen={d.task_type}")

    def test_calendar_add(self):    self._check("Takvime bugün için randevu koy", "calendar_add")
    def test_calendar_read(self):   self._check("Bugün ne var takvimimde?",        "calendar_read")
    def test_gmail_send(self):      self._check("Hocama mail gönder",              "gmail_send")
    def test_gmail_read(self):      self._check("Son maillerimi göster",           "gmail_read")
    def test_lms_check(self):       self._check("Ödev durumunu moodle'dan kontrol et", "lms_check")

    def test_task_takes_priority_over_ontology(self):
        """Görev komutu varsa ontoloji güçlü de olsa TASK dönmeli."""
        router = QueryRouter(_ont_high, _mem_has)
        d = router.route("Takvime sınav ekle", "ilayda")
        self.assertEqual(d.route, Route.TASK)


class TestMemoryScore(unittest.TestCase):
    def test_score_grows_with_lines(self):
        """Daha fazla bellek satırı → daha yüksek güven skoru."""
        router = QueryRouter(_ont_none, _mem_empty)
        few  = router._mem_score(["a"])
        more = router._mem_score(["a", "b", "c", "d", "e"])
        self.assertGreater(more, few)
        self.assertLessEqual(more, 1.0)


if __name__ == "__main__":
    unittest.main()
