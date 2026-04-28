"""shared/permissions.py — role → permission set mapping.

Single source of truth. Previously duplicated as:
  - ONTOLOGY_AGENT_ROLES in src/services/api/pipeline.py
  - _permissions  (inline dict) in src/agents/student_agent.py::_gate_c5_role_permission

Both copies are now replaced by the single dict below.

Note: this is the *union* of the two prior copies. The pipeline copy
already contained a "DeveloperAgent" entry; the student copy did not.
StudentAgent.gate_c5 calls `.get(role, set())`, so adding the
DeveloperAgent key is a no-op for any agent_role currently used by
StudentAgent in production (it is only ever instantiated as
"StudentAgent"). Behavior is preserved for all existing call sites.
"""

from __future__ import annotations

# role identifier → set of permission tokens
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
    "DeveloperAgent":        {"read_own_courses", "read_exam_dates",
                              "read_assignment_deadlines", "read_all_student_grades",
                              "manage_courses", "manage_department"},
    "ScrumMasterAgent":      {"manage_sprint", "read_team_progress",
                              "assign_tasks", "resolve_blockers",
                              "facilitate_ceremonies"},
    "ProductOwnerAgent":     {"manage_backlog", "create_stories",
                              "define_acceptance_criteria",
                              "prioritize_backlog", "accept_reject_stories",
                              "read_sprint_status"},
    "HRAgent":               {"read_candidate_profiles", "write_candidate_profiles",
                              "read_job_requisitions", "write_shortlists",
                              "write_interview_plans", "write_outreach_drafts",
                              "manage_recruiter_profile", "read_audit_log",
                              "manage_token_budget"},
}
