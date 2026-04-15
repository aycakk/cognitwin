"""Unit tests for src/gates/*.py — all six extracted gate modules.

Each gate section is fully self-contained. No mocking framework needed;
all functions are pure (no I/O, no shared state).
"""

import pytest

# ─────────────────────────────────────────────────────────────────────────────
#  C7 — check_blindspot
# ─────────────────────────────────────────────────────────────────────────────

from src.gates.c7_blindspot import check_blindspot


class TestC7Blindspot:
    def test_vector_not_empty_always_passes(self):
        passed, reason = check_blindspot("anything at all", vector_empty=False)
        assert passed is True
        assert reason is None

    def test_vector_not_empty_even_without_phrase(self):
        passed, _ = check_blindspot("no blindspot phrase here", vector_empty=False)
        assert passed is True

    def test_vector_empty_with_phrase_passes(self):
        passed, reason = check_blindspot("Bunu hafızamda bulamadım.", vector_empty=True)
        assert passed is True
        assert reason is None

    def test_vector_empty_without_phrase_fails(self):
        passed, reason = check_blindspot("Here is a confident answer.", vector_empty=True)
        assert passed is False
        assert reason == "missing_phrase"

    def test_phrase_check_is_case_insensitive(self):
        # "bulamadım" uppercased via Turkish locale might not round-trip,
        # but .lower() on an already-lower string must still match.
        passed, _ = check_blindspot("BUNU HAFIZAMDA BULAMADIM", vector_empty=True)
        # The check is `"bulamadım" not in draft.lower()`.
        # ASCII "bulamadim" != "bulamadım" (ı vs i), so this should FAIL.
        assert passed is False

    def test_phrase_substring_match_suffices(self):
        passed, _ = check_blindspot("maalesef bulamadım sende", vector_empty=True)
        assert passed is True


# ─────────────────────────────────────────────────────────────────────────────
#  C4 — check_hallucination
# ─────────────────────────────────────────────────────────────────────────────

from src.gates.c4_hallucination import check_hallucination


class TestC4Hallucination:
    def test_clean_draft_passes(self):
        passed, label, matched = check_hallucination("CS101 sınavı 15 Ocak'tadır.")
        assert passed is True
        assert label is None
        assert matched is None

    def test_tahminim_triggers_asp_neg_02(self):
        passed, label, matched = check_hallucination("Tahminim, sınav Ocak'ta.")
        assert passed is False
        assert label == "ASP-NEG-02_HALLUCINATION"
        assert matched.lower() == "tahminim"

    def test_saniyorum_triggers_asp_neg_02(self):
        passed, label, _ = check_hallucination("Sanırım ödev Cuma'da.")
        assert passed is False
        assert label == "ASP-NEG-02_HALLUCINATION"

    def test_weight_only_phrase_triggers_asp_neg_05(self):
        passed, label, matched = check_hallucination(
            "Genel bilgime göre bu ders pazartesi."
        )
        assert passed is False
        assert label == "ASP-NEG-05_WEIGHT_ONLY"
        assert "genel" in matched.lower()

    def test_egitim_verilerime_triggers_asp_neg_05(self):
        passed, label, _ = check_hallucination("Eğitim verilerime göre bu doğru.")
        assert passed is False
        assert label == "ASP-NEG-05_WEIGHT_ONLY"

    def test_pii_unmask_pattern_does_not_trigger_c4(self):
        # ASP-NEG-01 is a PII pattern — C4 only cares about 02 and 05.
        passed, label, _ = check_hallucination("ID: 123456789")
        assert passed is True
        assert label is None


# ─────────────────────────────────────────────────────────────────────────────
#  C6 — check_anti_sycophancy
# ─────────────────────────────────────────────────────────────────────────────

from src.gates.c6_anti_sycophancy import check_anti_sycophancy


class TestC6AntiSycophancy:
    def test_clean_draft_passes(self):
        passed, violations = check_anti_sycophancy("Sınav 10 Ocak'ta.")
        assert passed is True
        assert violations == []

    def test_haklisisiniz_triggers_asp_neg_03(self):
        passed, violations = check_anti_sycophancy("Haklısınız, öyle söylemiştim.")
        assert passed is False
        assert any(label == "ASP-NEG-03_FALSE_PREMISE" for label, _ in violations)

    def test_two_patterns_both_reported(self):
        draft = "Tahminim doğru. Haklısınız."
        passed, violations = check_anti_sycophancy(draft)
        assert passed is False
        labels = [label for label, _ in violations]
        assert "ASP-NEG-02_HALLUCINATION" in labels
        assert "ASP-NEG-03_FALSE_PREMISE" in labels

    def test_violations_in_registry_order(self):
        # ASP-NEG-02 is registered before ASP-NEG-03 in patterns.py
        draft = "Tahminim doğru. Haklısınız."
        _, violations = check_anti_sycophancy(draft)
        labels = [label for label, _ in violations]
        assert labels.index("ASP-NEG-02_HALLUCINATION") < labels.index("ASP-NEG-03_FALSE_PREMISE")

    def test_softened_fail_phrase_triggers(self):
        passed, violations = check_anti_sycophancy(
            "Yine de cevaplamaya çalışayım."
        )
        assert passed is False
        assert any(label == "ASP-NEG-04_SOFTENED_FAIL" for label, _ in violations)


# ─────────────────────────────────────────────────────────────────────────────
#  C5 — check_role_permission
# ─────────────────────────────────────────────────────────────────────────────

from src.gates.c5_role_permission import check_role_permission


class TestC5RolePermission:
    def test_student_clean_draft_passes(self):
        passed, kind = check_role_permission("Ödevini teslim et.", "StudentAgent")
        assert passed is True
        assert kind is None

    def test_student_bulk_grades_fails(self):
        passed, kind = check_role_permission(
            "tüm öğrencilerin notları burada.", "StudentAgent"
        )
        assert passed is False
        assert kind == "bulk_grades"

    def test_instructor_bulk_grades_passes(self):
        # InstructorAgent has read_all_student_grades
        passed, kind = check_role_permission(
            "tüm öğrencilerin notları burada.", "InstructorAgent"
        )
        assert passed is True
        assert kind is None

    def test_student_manage_courses_fails(self):
        passed, kind = check_role_permission(
            "dersi güncelle lütfen.", "StudentAgent"
        )
        assert passed is False
        assert kind == "manage_courses"

    def test_head_of_department_manage_courses_passes(self):
        passed, kind = check_role_permission(
            "dersi güncelle lütfen.", "HeadOfDepartmentAgent"
        )
        assert passed is True
        assert kind is None

    def test_unknown_role_bulk_grades_fails(self):
        # Unknown role → empty permitted set → violation
        passed, kind = check_role_permission(
            "tüm öğrencilerin notları.", "UnknownAgent"
        )
        assert passed is False
        assert kind == "bulk_grades"

    def test_both_violations_first_one_returned(self):
        # bulk_grades check comes before manage_courses in the function body
        draft = "tüm öğrencilerin notları ve dersi güncelle"
        passed, kind = check_role_permission(draft, "StudentAgent")
        assert passed is False
        assert kind == "bulk_grades"


# ─────────────────────────────────────────────────────────────────────────────
#  C2 — check_grounding
# ─────────────────────────────────────────────────────────────────────────────

from src.gates.c2_grounding import check_grounding


class TestC2Grounding:
    def test_developer_exempt_via_policy(self):
        """Exemption is now handled by GATE_POLICY — DeveloperAgent has no C2."""
        from src.governance.policy import GATE_POLICY
        assert "C2" not in GATE_POLICY["DeveloperAgent"]
        # C2_DEV is the developer-specific grounding gate
        assert "C2_DEV" in GATE_POLICY["DeveloperAgent"]

    def test_empty_vector_with_blindspot_phrase_passes(self):
        passed, reason, count = check_grounding(
            "Bunu hafızamda bulamadım.", "", vector_empty=True
        )
        assert passed is True
        assert reason == "empty_pass"
        assert count == 0

    def test_empty_vector_without_blindspot_phrase_fails(self):
        passed, reason, count = check_grounding(
            "CS101 sınavı Ocak'ta.", "", vector_empty=True        )
        assert passed is False
        assert reason == "empty_fail"
        assert count == 0

    def test_non_empty_vector_blindspot_phrase_passes(self):
        passed, reason, count = check_grounding(
            "Bunu hafızamda bulamadım.", "some context", vector_empty=False        )
        assert passed is True
        assert reason == "blindspot"
        assert count == 0

    def test_overlap_pass_with_two_shared_words(self):
        context = "student midterm examination schedule posted"
        draft   = "midterm examination is on Monday"
        passed, reason, count = check_grounding(
            draft, context, vector_empty=False        )
        assert passed is True
        assert reason == "overlap_pass"
        assert count >= 2

    def test_overlap_fail_with_one_shared_word(self):
        context = "midterm schedule posted here"
        draft   = "midterm is on Monday"
        # Only "midterm" (7 chars) is shared; "schedule"/"posted" not in draft
        passed, reason, count = check_grounding(
            draft, context, vector_empty=False        )
        assert passed is False
        assert reason == "overlap_fail"
        assert count == 0

    def test_short_words_excluded_from_overlap(self):
        # Words under 6 chars are ignored by the regex \b\w{6,}\b
        context = "cat dog run fly"
        draft   = "cat dog run fly"
        passed, reason, _ = check_grounding(
            draft, context, vector_empty=False        )
        assert passed is False
        assert reason == "overlap_fail"

    def test_masked_tokens_excluded_from_context_pool(self):
        # [STUDENT_ID_MASKED] should not count as a shared content word
        context = "[STUDENT_ID_MASKED] enrolled in midterm examination schedule"
        draft   = "STUDENT_ID_MASKED midterm examination schedule reviewed"
        passed, reason, count = check_grounding(
            draft, context, vector_empty=False        )
        # "midterm", "examination", "schedule" are shared (≥3) → overlap_pass
        assert passed is True
        assert reason == "overlap_pass"


# ─────────────────────────────────────────────────────────────────────────────
#  C2_DEV — check_dev_grounding (developer codebase context)
# ─────────────────────────────────────────────────────────────────────────────

from src.gates.c2_dev_grounding import check_dev_grounding


class TestC2DevGrounding:
    _CONTEXT = (
        "=== CODEBASE CONTEXT ===\n"
        "def run_pipeline(query, model, messages):\n"
        "    vector_context = VECTOR_MEM.retrieve(query)\n"
        "    draft = generate_response(query, vector_context)\n"
        "    return evaluate_gates(draft)\n"
        "=== END CODEBASE CONTEXT ==="
    )

    def test_no_context_with_blindspot_passes(self):
        passed, reason, count = check_dev_grounding(
            "Bunu hafızamda bulamadım.", "", context_empty=True
        )
        assert passed is True
        assert reason == "no_context_pass"
        assert count == 0

    def test_no_context_without_blindspot_fails(self):
        passed, reason, count = check_dev_grounding(
            "The pipeline uses a 3-stage architecture.", "", context_empty=True
        )
        assert passed is False
        assert reason == "no_context_fail"

    def test_blindspot_in_non_empty_context_passes(self):
        passed, reason, count = check_dev_grounding(
            "Bunu hafızamda bulamadım.", self._CONTEXT, context_empty=False
        )
        assert passed is True
        assert reason == "blindspot"

    def test_overlap_pass_with_grounded_content(self):
        draft = (
            "The run_pipeline function retrieves vector_context using VECTOR_MEM, "
            "then calls generate_response with the query and evaluates gates on the draft."
        )
        passed, reason, count = check_dev_grounding(
            draft, self._CONTEXT, context_empty=False
        )
        assert passed is True
        assert reason == "overlap_pass"
        assert count >= 3

    def test_overlap_fail_with_ungrounded_content(self):
        draft = "React components should use useState hooks for state management."
        passed, reason, count = check_dev_grounding(
            draft, self._CONTEXT, context_empty=False
        )
        assert passed is False
        assert reason == "overlap_fail"

    def test_policy_assigns_c2_dev_to_developer(self):
        from src.governance.policy import GATE_POLICY
        assert "C2_DEV" in GATE_POLICY["DeveloperAgent"]
        assert "C2" not in GATE_POLICY["DeveloperAgent"]

    def test_policy_does_not_assign_c2_dev_to_student(self):
        from src.governance.policy import GATE_POLICY
        assert "C2_DEV" not in GATE_POLICY["StudentAgent"]
        assert "C2" in GATE_POLICY["StudentAgent"]


# ─────────────────────────────────────────────────────────────────────────────
#  C3 — check_ontology_compliance
# ─────────────────────────────────────────────────────────────────────────────

from src.gates.c3_ontology_compliance import check_ontology_compliance


class TestC3OntologyCompliance:
    _PAIRS = [
        {"exam": "http://cognitwin.org/upper#midterm1",
         "course": "http://cognitwin.org/upper#CS101"},
    ]

    def test_empty_pairs_passes(self):
        passed, reason, detail = check_ontology_compliance("any draft", [])
        assert passed is True
        assert reason == "pass"
        assert detail is None

    def test_exam_not_mentioned_passes(self):
        passed, reason, detail = check_ontology_compliance(
            "Bu ders CS202 için geçerlidir.", self._PAIRS
        )
        assert passed is True
        assert reason == "pass"
        assert detail is None

    def test_exam_mentioned_correct_code_passes(self):
        passed, reason, detail = check_ontology_compliance(
            "midterm1 sınavı CS101 dersine aittir.", self._PAIRS
        )
        assert passed is True
        assert reason == "pass"
        assert detail is None

    def test_exam_mentioned_wrong_code_fails(self):
        passed, reason, detail = check_ontology_compliance(
            "midterm1 sınavı CS202 dersine aittir.", self._PAIRS
        )
        assert passed is False
        assert reason == "violation"
        assert detail is not None
        exam_lbl, bad_code, expected_lbl = detail
        assert exam_lbl == "midterm1"
        assert bad_code.upper() == "CS202"
        assert expected_lbl == "cs101"

    def test_violation_detail_is_tuple_of_three_strings(self):
        _, _, detail = check_ontology_compliance(
            "midterm1 exam CS999.", self._PAIRS
        )
        assert isinstance(detail, tuple)
        assert len(detail) == 3
        assert all(isinstance(s, str) for s in detail)

    def test_no_course_code_in_draft_passes(self):
        # exam label mentioned but no CSxxx pattern → no violation possible
        passed, reason, _ = check_ontology_compliance(
            "midterm1 sınavı yakında.", self._PAIRS
        )
        assert passed is True
        assert reason == "pass"


# ─────────────────────────────────────────────────────────────────────────────
#  C4 — English hallucination markers (KNOWN GAP)
# ─────────────────────────────────────────────────────────────────────────────


class TestC4EnglishHallucinationGap:
    """Documents the known gap: C4 only detects Turkish hedging phrases.

    All tests below assert that C4 PASSES on English hallucination markers.
    This is the INCORRECT behavior — these phrases SHOULD be caught.
    These tests exist as regression anchors: when the P0 fix adds English
    markers, these assertions should be flipped to ``passed is False``.
    """

    @pytest.mark.parametrize("phrase", [
        "I think the answer is 42.",
        "It probably happened last week.",
        "Maybe the deadline was extended.",
        "I believe this course is mandatory.",
        "Perhaps we should check again.",
        "It seems like the server is down.",
        "As far as I know, this is correct.",
    ])
    def test_english_hedging_not_caught(self, phrase):
        passed, label, matched = check_hallucination(phrase)
        assert passed is True, (
            f"KNOWN GAP: English phrase '{phrase}' is not caught by C4. "
            "Flip this assertion when English markers are added."
        )
        assert label is None
