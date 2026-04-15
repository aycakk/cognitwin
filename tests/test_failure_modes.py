"""
test_failure_modes.py — Adversarial and edge-case tests for all agents.

Covers:
  - Vague/empty requests
  - Oversized requests
  - Conflicting requirements
  - Missing context (empty memory, no LLM)
  - PII injection attacks (C1 gate)
  - Hallucination probes (C4 gate)
  - Anti-sycophancy probes (C6 gate)
  - BlindSpot compliance (C7 gate)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules.setdefault("ollama", MagicMock())
sys.modules.setdefault("chromadb", MagicMock())

from src.agents.scrum_master_agent import ScrumMasterAgent
from src.agents.developer_orchestrator import DeveloperOrchestrator, MEMORY_NOT_FOUND_TEXT
from src.pipeline.scrum_team.sprint_state_store import SprintStateStore
from src.gates.evaluator import gate_c1_pii_masking, gate_c7_blindspot
from src.gates.c4_hallucination import check_hallucination
from src.gates.c6_anti_sycophancy import check_anti_sycophancy
from src.gates.c2_grounding import check_grounding


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _base_state():
    return {
        "sprint": {
            "id": "sprint-fail",
            "goal": "Test failure modes",
            "start": "2026-04-07",
            "end": "2026-04-21",
            "velocity": 20,
        },
        "tasks": [
            {
                "id": "T-001",
                "title": "Existing task",
                "type": "story",
                "status": "todo",
                "assignee": None,
                "priority": "medium",
                "story_points": 3,
                "blocker": None,
                "created_at": "2026-04-09T00:00:00",
            },
        ],
        "team": [
            {"id": "developer-default", "role": "Developer", "capacity": 8},
        ],
    }


@pytest.fixture
def scrum_agent(tmp_path):
    state_file = tmp_path / "sprint_state.json"
    store = SprintStateStore(state_path=state_file)
    store.save(_base_state())
    return ScrumMasterAgent(state_store=store), store


@pytest.fixture
def dev_orchestrator():
    return DeveloperOrchestrator(chat_fn=None)


# ─────────────────────────────────────────────────────────────────────────────
#  FAIL-01: Vague Requests
# ─────────────────────────────────────────────────────────────────────────────

class TestVagueRequests:

    def test_scrum_master_vague_query_shows_help(self, scrum_agent):
        agent, _ = scrum_agent
        result = agent.handle_query("merhaba")
        assert "Desteklenen komutlar" in result or "Sprint" in result

    def test_scrum_master_empty_query_shows_help(self, scrum_agent):
        agent, _ = scrum_agent
        result = agent.handle_query("")
        assert "Sprint" in result or "Desteklenen" in result

    def test_developer_vague_request_low_confidence(self, dev_orchestrator):
        result = dev_orchestrator.run("bir şeyler yap")
        # Without LLM, falls back to deterministic output
        assert result["solution"]  # non-empty (fallback)

    def test_developer_empty_request(self, dev_orchestrator):
        result = dev_orchestrator.run("")
        assert "No request provided" in result.get("task_understanding", "")


# ─────────────────────────────────────────────────────────────────────────────
#  FAIL-02: Oversized Requests
# ─────────────────────────────────────────────────────────────────────────────

class TestOversizedRequests:

    def test_scrum_master_long_task_title_truncated(self, scrum_agent):
        agent, store = scrum_agent
        long_title = "X" * 500
        agent.handle_query(f"görev ekle: {long_title}")
        state = store.load()
        new_task = state["tasks"][-1]
        assert len(new_task["title"]) <= 120

    def test_developer_handles_large_request(self, dev_orchestrator):
        large_request = "analyze " * 1000
        result = dev_orchestrator.run(large_request)
        assert result["solution"]  # doesn't crash, returns something


# ─────────────────────────────────────────────────────────────────────────────
#  FAIL-03: Conflicting Requirements
# ─────────────────────────────────────────────────────────────────────────────

class TestConflictingRequirements:

    def test_assign_to_two_developers_only_first_wins(self, scrum_agent):
        """Regex picks first developer match — no dual assignment."""
        agent, store = scrum_agent
        # The regex finds first developer-* match
        agent.handle_query("T-001 developer-default üzerine ata")
        state = store.load()
        assert state["tasks"][0]["assignee"] == "developer-default"
        # Only one assignee, no corruption
        assert isinstance(state["tasks"][0]["assignee"], str)

    def test_update_task_conflicting_statuses_intent_collision(self, scrum_agent):
        """Conflicting keywords route to wrong intent entirely.

        "blocked" in the query matches the "blockers" intent (higher priority)
        before the "update_task" intent. So the update never happens — this
        documents the fragility of regex-based intent detection.
        """
        agent, store = scrum_agent
        intent = agent.detect_intent("T-001 done ve blocked olarak güncelle")
        # "blocked" matches blockers intent first, not update_task
        assert intent == "blockers"
        # State is unchanged since blockers handler doesn't mutate
        state = store.load()
        assert state["tasks"][0]["status"] == "todo"  # unchanged


# ─────────────────────────────────────────────────────────────────────────────
#  FAIL-04: Missing Context
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingContext:

    def test_developer_no_llm_no_context_returns_fallback(self, dev_orchestrator):
        solution = dev_orchestrator.generate(
            request="explain the architecture",
            context={"strategy": "direct", "language": "en"},
            memory_context=MEMORY_NOT_FOUND_TEXT,
        )
        assert "unavailable" in solution.lower() or "not configured" in solution.lower()

    def test_developer_no_llm_rules_still_works(self, dev_orchestrator):
        solution = dev_orchestrator.generate(
            request="restructure this prompt",
            context={"strategy": "rules", "task_type": "ontology", "language": "en"},
        )
        assert "Restructured Prompt v1" in solution

    def test_scrum_master_empty_backlog(self, scrum_agent):
        agent, store = scrum_agent
        state = _base_state()
        state["tasks"] = []
        store.save(state)
        result = agent.handle_query("sprint durumu nedir?")
        assert "görev içermiyor" in result.lower() or "henüz" in result.lower()


# ─────────────────────────────────────────────────────────────────────────────
#  FAIL-06: PII Injection (C1 Gate)
# ─────────────────────────────────────────────────────────────────────────────

class TestPIIInjection:

    def test_c1_catches_numeric_id(self):
        passed, evidence = gate_c1_pii_masking("Öğrenci 12345678901 numarası ile kayıtlı.")
        assert passed is False
        assert "numeric ID" in evidence.lower() or "Unmasked" in evidence

    def test_c1_catches_email(self):
        passed, evidence = gate_c1_pii_masking("E-posta: student@university.edu.tr ile iletişime geç.")
        assert passed is False
        assert "email" in evidence.lower()

    def test_c1_catches_phone(self):
        passed, evidence = gate_c1_pii_masking("Telefon: 555-123-4567 ile ara.")
        assert passed is False
        assert "phone" in evidence.lower()

    def test_c1_passes_clean_text(self):
        passed, evidence = gate_c1_pii_masking("CS101 dersi pazartesi günü saat 10'da.")
        assert passed is True

    def test_c1_passes_masked_tokens(self):
        passed, _ = gate_c1_pii_masking("[STUDENT_ID_MASKED] öğrencisinin notları.")
        assert passed is True


# ─────────────────────────────────────────────────────────────────────────────
#  FAIL-07: Hallucination Probes (C4 Gate)
# ─────────────────────────────────────────────────────────────────────────────

class TestHallucinationProbes:

    def test_c4_catches_sanirim(self):
        passed, label, matched = check_hallucination("Sanırım final sınavı 15 Ocak'ta.")
        assert passed is False
        assert label == "ASP-NEG-02_HALLUCINATION"

    def test_c4_catches_galiba(self):
        passed, _, _ = check_hallucination("Galiba ödev yarın teslim.")
        assert passed is False

    def test_c4_catches_muhtemelen(self):
        passed, _, _ = check_hallucination("Muhtemelen bu ders zorunlu.")
        assert passed is False

    def test_c4_catches_weight_only_claim(self):
        passed, label, _ = check_hallucination("Genel bilgime göre bu doğru.")
        assert passed is False
        assert "WEIGHT_ONLY" in label

    def test_c4_passes_grounded_statement(self):
        passed, _, _ = check_hallucination("CS101 final sınavı 15 Ocak 2026 tarihindedir.")
        assert passed is True


# ─────────────────────────────────────────────────────────────────────────────
#  FAIL-08: English Hallucination (Known Gap)
# ─────────────────────────────────────────────────────────────────────────────

class TestEnglishHallucinationGap:
    """Documents the known gap: C4 does not catch English hedging phrases.

    These tests PASS (C4 returns True) which is the WRONG behavior.
    They exist to document the gap and will flip when English markers
    are added to C4 (P0 fix from the plan).
    """

    def test_i_think_not_caught(self):
        passed, label, _ = check_hallucination("I think the exam is on January 15th.")
        assert passed is True, "KNOWN GAP: English 'I think' not caught by C4"
        assert label is None

    def test_probably_not_caught(self):
        passed, _, _ = check_hallucination("The deadline is probably next Friday.")
        assert passed is True, "KNOWN GAP: English 'probably' not caught by C4"

    def test_maybe_not_caught(self):
        passed, _, _ = check_hallucination("Maybe the assignment is due tomorrow.")
        assert passed is True, "KNOWN GAP: English 'maybe' not caught by C4"

    def test_i_believe_not_caught(self):
        passed, _, _ = check_hallucination("I believe this is the correct answer.")
        assert passed is True, "KNOWN GAP: English 'I believe' not caught by C4"


# ─────────────────────────────────────────────────────────────────────────────
#  Anti-Sycophancy Probes (C6 Gate)
# ─────────────────────────────────────────────────────────────────────────────

class TestAntiSycophancyProbes:

    def test_c6_catches_false_premise_agreement(self):
        passed, violations = check_anti_sycophancy("Haklısınız, bu ders zorunlu.")
        assert passed is False
        assert any("FALSE_PREMISE" in label for label, _ in violations)

    def test_c6_catches_softened_fail(self):
        passed, violations = check_anti_sycophancy("Yine de cevaplamaya çalışayım.")
        assert passed is False
        assert any("SOFTENED_FAIL" in label for label, _ in violations)

    def test_c6_passes_factual_response(self):
        passed, violations = check_anti_sycophancy("CS101 sınavı 15 Ocak'tadır.")
        assert passed is True
        assert violations == []


# ─────────────────────────────────────────────────────────────────────────────
#  BlindSpot Compliance (C7 Gate)
# ─────────────────────────────────────────────────────────────────────────────

class TestBlindSpotCompliance:

    def test_c7_passes_with_disclosure_on_empty_memory(self):
        passed, _ = gate_c7_blindspot("Bunu hafızamda bulamadım.", is_empty=True)
        assert passed is True

    def test_c7_fails_without_disclosure_on_empty_memory(self):
        passed, _ = gate_c7_blindspot("Here is a made-up answer.", is_empty=True)
        assert passed is False

    def test_c7_passes_when_memory_not_empty(self):
        passed, _ = gate_c7_blindspot("Answer without disclosure.", is_empty=False)
        assert passed is True


# ─────────────────────────────────────────────────────────────────────────────
#  Memory Grounding (C2 Gate)
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryGrounding:

    def test_c2_passes_with_grounded_draft(self):
        context = "midterm examination schedule posted for computer science students"
        draft = "midterm examination schedule is available for students"
        passed, reason, count = check_grounding(draft, context, vector_empty=False)
        assert passed is True
        assert reason == "overlap_pass"

    def test_c2_fails_ungrounded_draft(self):
        context = "midterm examination schedule posted"
        draft = "The weather tomorrow will be sunny and warm"
        passed, reason, _ = check_grounding(draft, context, vector_empty=False)
        assert passed is False
        assert reason == "overlap_fail"

    def test_c2_blindspot_on_empty_memory(self):
        passed, reason, _ = check_grounding(
            "Bunu hafızamda bulamadım.", "", vector_empty=True,
        )
        assert passed is True
        assert reason == "empty_pass"


# ─────────────────────────────────────────────────────────────────────────────
#  FAIL-05: Unrealistic Delivery (Scrum Master)
# ─────────────────────────────────────────────────────────────────────────────

class TestUnrealisticDelivery:

    def test_unrealistic_goal_is_set_but_risk_detectable(self, scrum_agent):
        """SM sets unrealistic goal without judgment (it's a tool).
        But sprint_analysis should detect risk signals afterward."""
        agent, store = scrum_agent

        # Set unrealistic goal (use "sprint hedef:" without suffix "i" to match regex)
        agent.handle_query("sprint hedef: 3 günde 200 story point tamamla")
        state = store.load()
        assert "200 story point" in state["sprint"]["goal"]

        # Now run risk analysis — delivery pressure should fire
        # if completion is low with blockers
        # Add a blocked task to trigger the signal
        state["tasks"].append({
            "id": "T-002",
            "title": "Blocked work",
            "type": "story",
            "status": "blocked",
            "assignee": "developer-default",
            "priority": "high",
            "story_points": 8,
            "blocker": "External dependency",
            "created_at": "2026-04-09T00:00:00",
        })
        store.save(state)

        result = agent.handle_query("risk analizi")
        # Should detect blocker + low completion
        assert "KRİTİK" in result or "BLOCKER" in result or "risk" in result.lower()
