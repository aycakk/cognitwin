"""gates/c5_role_permission.py — shared decision logic for C5.

Single source of truth for role-permission boundary checks.
Previously duplicated as:
  - pipeline.py::gate_c5_role_permission
  - student_agent.py::StudentAgent._gate_c5_role_permission

Both call sites already used the shared ONTOLOGY_AGENT_ROLES dict
(unified in Step 1). The two copies differed only in:
  • Output language of the violation message (English vs Turkish)
  • Whether agent_role was a parameter or `self.agent_role`

This module exposes a pure decision function returning a machine-
readable reason code. Each caller maps the code to its own
localized message string, so existing return values are preserved
byte-for-byte.

Reason codes:
  None              → no violation (pass)
  "bulk_grades"     → draft references bulk-student-grades and role lacks
                      'read_all_student_grades'
  "manage_courses"  → draft references course management and role lacks
                      'manage_courses'
"""

from __future__ import annotations

import re
from typing import Optional

from src.shared.permissions import ONTOLOGY_AGENT_ROLES

# Regex patterns are intentionally identical to the originals in
# pipeline.py:660,664 and student_agent.py:435,439 — verbatim.
_BULK_GRADES_RE     = re.compile(r"tüm öğrencilerin notları|bütün öğrenciler", re.I)
_MANAGE_COURSES_RE  = re.compile(r"dersi güncelle|ders planını değiştir",      re.I)


def check_role_permission(
    draft: str,
    agent_role: str,
) -> tuple[bool, Optional[str]]:
    """Decide whether `draft` violates the permission set of `agent_role`.

    Returns (passed, violation_kind). The caller is responsible for
    formatting the user-visible message — this function never produces
    natural-language strings.
    """
    permitted = ONTOLOGY_AGENT_ROLES.get(agent_role, set())

    if _BULK_GRADES_RE.search(draft):
        if "read_all_student_grades" not in permitted:
            return False, "bulk_grades"

    if _MANAGE_COURSES_RE.search(draft):
        if "manage_courses" not in permitted:
            return False, "manage_courses"

    return True, None
