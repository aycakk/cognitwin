from __future__ import annotations

import os
import re
from dataclasses import dataclass


@dataclass
class OntologyIssue:
    code: str
    message: str
    severity: str = "error"


@dataclass
class OntologyAudit:
    is_consistent: bool
    issues: list[OntologyIssue]
    upper_text: str
    student_text: str


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _check_top_object_property_domain_range(upper_text: str) -> list[OntologyIssue]:
    # Global domain/range on owl:topObjectProperty over-constrains all object properties.
    issues: list[OntologyIssue] = []
    if "owl:topObjectProperty rdfs:domain" in upper_text and "rdfs:range" in upper_text:
        issues.append(
            OntologyIssue(
                code="ONTO-001",
                severity="error",
                message=(
                    "owl:topObjectProperty has global domain/range constraints. "
                    "This can force unintended class inferences across the ontology."
                ),
            )
        )
    return issues


def _check_financial_mask_label(upper_text: str) -> list[OntologyIssue]:
    # FINANCIAL_MASKED should define both category and visible mask text.
    issues: list[OntologyIssue] = []
    block_re = re.compile(r":FINANCIAL_MASKED\b.*?\.(?:\s|$)", re.DOTALL)
    m = block_re.search(upper_text)
    if not m:
        return issues

    block = m.group(0)
    has_mask_text = ":maskText" in block
    bad_mask_category_value = bool(re.search(r':maskCategory\s+[^.;]*"\[FINANCIAL_MASKED\]"', block))
    if bad_mask_category_value or not has_mask_text:
        issues.append(
            OntologyIssue(
                code="ONTO-002",
                severity="warning",
                message=(
                    "FINANCIAL_MASKED appears to use :maskCategory for '[FINANCIAL_MASKED]' "
                    "and/or misses :maskText."
                ),
            )
        )
    return issues


def _check_import_alignment(student_text: str) -> list[OntologyIssue]:
    issues: list[OntologyIssue] = []
    if "owl:imports <http://cognitwin.org/upper>" not in student_text:
        issues.append(
            OntologyIssue(
                code="ONTO-003",
                severity="error",
                message="Student ontology does not import the expected upper ontology IRI.",
            )
        )
    return issues


def audit_ontologies(project_root: str) -> OntologyAudit:
    upper_path = os.path.join(project_root, "ontologies", "cognitwin-upper.ttl")
    student_path = os.path.join(project_root, "ontologies", "student_ontology.ttl")

    upper_text = _read_text(upper_path)
    student_text = _read_text(student_path)

    issues: list[OntologyIssue] = []

    if not upper_text:
        issues.append(OntologyIssue(code="ONTO-004", severity="error", message="Upper ontology file is missing or unreadable."))
    if not student_text:
        issues.append(OntologyIssue(code="ONTO-005", severity="error", message="Student ontology file is missing or unreadable."))

    if upper_text:
        issues.extend(_check_top_object_property_domain_range(upper_text))
        issues.extend(_check_financial_mask_label(upper_text))
    if student_text:
        issues.extend(_check_import_alignment(student_text))

    is_consistent = not any(i.severity == "error" for i in issues)
    return OntologyAudit(
        is_consistent=is_consistent,
        issues=issues,
        upper_text=upper_text,
        student_text=student_text,
    )


def check_prompt_ontology_alignment(system_prompt: str, audit: OntologyAudit) -> str:
    # Prompt says ontology-grounded; if ontology has hard errors we should surface mismatch.
    prompt_mentions_ontology = "ONTOLOGY" in (system_prompt or "").upper()
    if prompt_mentions_ontology and not audit.is_consistent:
        return "PROMPT_ONTOLOGY_MISMATCH"
    if prompt_mentions_ontology and audit.is_consistent:
        return "PROMPT_ONTOLOGY_ALIGNED"
    return "PROMPT_OMITS_ONTOLOGY"


def build_ontology_context(query: str, audit: OntologyAudit, max_lines: int = 8) -> str:
    # Lightweight lexical retrieval over TTL lines to ground responses with ontology hints.
    text = f"{audit.upper_text}\n{audit.student_text}"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]

    q = re.sub(r"[^a-zA-Z0-9çğıöşüÇĞİÖŞÜ_#:/.-]+", " ", query or "").lower()
    tokens = {t for t in q.split() if len(t) >= 3}

    if not lines:
        return ""
    if not tokens:
        return "\n".join(lines[:max_lines])

    scored: list[tuple[int, int]] = []
    for idx, ln in enumerate(lines):
        low = ln.lower()
        score = sum(1 for t in tokens if t in low)
        if score > 0:
            scored.append((score, idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_indices = [idx for _, idx in scored[:max_lines]]

    # Include adjacent lines so object-property facts keep their linked literals
    # (e.g., activity relation + exam scope text) in the same context block.
    expanded: list[int] = []
    seen: set[int] = set()
    for idx in top_indices:
        for j in [idx - 1, idx, idx + 1]:
            if 0 <= j < len(lines) and j not in seen:
                seen.add(j)
                expanded.append(j)

    selected = [lines[i] for i in expanded[: max_lines * 2]]
    return "\n".join(selected)


