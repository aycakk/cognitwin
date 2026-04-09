"""gates/c3_ontology_compliance.py — shared decision logic for C3.

Single source of truth for the ontology fact-consistency check.
Previously lived entirely inside pipeline.py::gate_c3_ontology_compliance.

Policy: the draft must not pair an exam with a course code that
contradicts the ontology's Exam→activityPartOf→Course triples.

Infrastructure split:
  The caller (pipeline.py wrapper) is responsible for:
    • checking whether the rdflib graph is available
    • running the SPARQL query and obtaining the raw result list
  This module receives the plain result list and applies the
  decision logic.  It has no rdflib import, no I/O, no side effects.

Input contract for exam_course_pairs:
  A list of dicts as returned by pipeline._sparql(), where each dict
  has at least "exam" and "course" keys whose values are URI strings,
  e.g. {"exam": "http://cognitwin.org/upper#midterm1",
         "course": "http://cognitwin.org/upper#CS101"}.
  An empty list means the ontology has no Exam triples to check;
  the function returns a clean PASS.

Return contract:
  (passed: bool, reason_code: str, violation_detail: tuple | None)

  reason_code values:
    "pass"      → no violations found; violation_detail is None
    "violation" → exam paired with wrong course code
                  violation_detail = (exam_lbl, bad_code, expected_lbl)
                  The wrapper formats these into the user message.
"""

from __future__ import annotations

import re
from typing import Optional


def check_ontology_compliance(
    draft: str,
    exam_course_pairs: list[dict[str, str]],
) -> tuple[bool, str, Optional[tuple[str, str, str]]]:
    """Decide whether `draft` contradicts any Exam→Course triple.

    Parameters
    ----------
    draft:              the LLM-produced response text to evaluate
    exam_course_pairs:  raw SPARQL result rows; each row must have
                        "exam" and "course" URI string values.
                        Pass an empty list when no triples exist.

    Returns (passed, reason_code, violation_detail).
    See module docstring for the full contract.
    """
    for row in exam_course_pairs:
        exam_lbl   = row["exam"].split("/")[-1].split("#")[-1].lower()
        course_lbl = row["course"].split("/")[-1].split("#")[-1].lower()

        if exam_lbl in draft.lower():
            for oc in re.findall(r"\bcs\d{3}\b", draft, re.I):
                if oc.lower() != course_lbl:
                    return False, "violation", (exam_lbl, oc, course_lbl)

    return True, "pass", None
