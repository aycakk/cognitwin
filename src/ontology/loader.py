"""ontology/loader.py — lazy rdflib graph loader and SPARQL helper.

Single source of truth for ontology infrastructure.
Previously defined as module-level helpers in src/services/api/pipeline.py;
moved here so that src/gates/evaluator.py can import them without
creating a circular dependency back into pipeline.py.

The graph is loaded once on first call and cached in module-level state.

Failure policy
--------------
If COGNITWIN_ONTOLOGY_REQUIRED=1 is set in the environment, a missing or
unparseable ontology file raises OntologyLoadError immediately so the gate
fails loudly rather than issuing a provisional pass.

Default (COGNITWIN_ONTOLOGY_REQUIRED unset / "0"): the old silent-None
behaviour is preserved, but the failure is now logged at WARNING (not DEBUG)
so it is visible in production logs without breaking existing deployments.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from src.core.exceptions import OntologyLoadError  # noqa: F401 (re-exported)

logger = logging.getLogger(__name__)

# Module-level cache — loaded once per process, never shared across requests.
_ONTOLOGY_GRAPH: Optional[object] = None
_ONTOLOGY_TRIED: bool = False

# Set COGNITWIN_ONTOLOGY_REQUIRED=1 to make ontology absence a hard error.
_ONTOLOGY_REQUIRED: bool = os.environ.get("COGNITWIN_ONTOLOGY_REQUIRED", "0") == "1"


def _get_ontology_graph():
    """
    Load and cache the rdflib Graph.

    Returns the Graph if successfully loaded, None otherwise.
    Raises OntologyLoadError if COGNITWIN_ONTOLOGY_REQUIRED=1 and loading fails.
    """
    global _ONTOLOGY_GRAPH, _ONTOLOGY_TRIED
    if _ONTOLOGY_TRIED:
        return _ONTOLOGY_GRAPH
    _ONTOLOGY_TRIED = True

    onto_dir = Path(__file__).resolve().parents[2] / "ontologies"
    expected_files = ("cognitwin-upper.ttl", "student_ontology.ttl")
    missing: list[str] = []

    try:
        from rdflib import Graph
        g = Graph()
        loaded = False

        for fname in expected_files:
            p = onto_dir / fname
            if p.exists():
                g.parse(str(p), format="turtle")
                loaded = True
                logger.debug("[ONTOLOGY] Loaded: %s", p)
            else:
                missing.append(fname)
                # Changed from DEBUG to WARNING — missing ontology files must
                # be visible in production logs, not silently swallowed.
                logger.warning(
                    "[ONTOLOGY] File not found: %s — C3 gate will operate in "
                    "degraded mode. Set COGNITWIN_ONTOLOGY_REQUIRED=1 to treat "
                    "this as a hard failure.",
                    p,
                )

        if not loaded:
            msg = (
                f"[ONTOLOGY] No ontology files loaded from {onto_dir}. "
                f"Expected: {list(expected_files)}. Missing: {missing}."
            )
            logger.warning(msg)
            if _ONTOLOGY_REQUIRED:
                raise OntologyLoadError(msg)
            _ONTOLOGY_GRAPH = None
            return None

        if missing and _ONTOLOGY_REQUIRED:
            raise OntologyLoadError(
                f"Required ontology files missing: {missing}"
            )

        _ONTOLOGY_GRAPH = g

    except OntologyLoadError:
        raise  # re-raise; do not swallow hard failures
    except Exception as exc:
        msg = f"[ONTOLOGY] Load failed: {exc}"
        logger.warning(msg)
        if _ONTOLOGY_REQUIRED:
            raise OntologyLoadError(msg) from exc
        _ONTOLOGY_GRAPH = None

    return _ONTOLOGY_GRAPH


def _sparql(query: str) -> list[dict]:
    g = _get_ontology_graph()
    if g is None:
        return []
    qres = g.query(query)
    return [{str(v): str(row[v]) for v in qres.vars} for row in qres]


# =============================================================================
#  Scrum Master graph — separate singleton for agile.ttl + scrum_master.ttl
#
#  Kept isolated from the student-path graph so:
#    • loading one path never blocks or pollutes the other
#    • scrum_master_runner.py can import build_scrum_master_ontology_context()
#      from here without pulling in the ChromaDB singletons in shared.py
# =============================================================================

_SM_GRAPH: Optional[object] = None
_SM_GRAPH_TRIED: bool = False


def _get_scrum_master_graph():
    """
    Load and cache the Scrum Master rdflib Graph.

    Parses agile.ttl then scrum_master.ttl into a dedicated graph instance.
    Returns the Graph on success, None on failure.
    Same failure-policy contract as _get_ontology_graph().
    """
    global _SM_GRAPH, _SM_GRAPH_TRIED
    if _SM_GRAPH_TRIED:
        return _SM_GRAPH
    _SM_GRAPH_TRIED = True

    onto_dir = Path(__file__).resolve().parents[2] / "ontologies"
    sm_files = ("agile.ttl", "scrum_master.ttl")
    missing: list[str] = []

    try:
        from rdflib import Graph
        g = Graph()
        loaded = False

        for fname in sm_files:
            p = onto_dir / fname
            if p.exists():
                g.parse(str(p), format="turtle")
                loaded = True
                logger.debug("[SM-ONTOLOGY] Loaded: %s", p)
            else:
                missing.append(fname)
                logger.warning(
                    "[SM-ONTOLOGY] File not found: %s — Scrum Master ontology "
                    "context will be unavailable. Set COGNITWIN_ONTOLOGY_REQUIRED=1 "
                    "to treat this as a hard failure.",
                    p,
                )

        if not loaded:
            msg = (
                f"[SM-ONTOLOGY] No SM ontology files loaded from {onto_dir}. "
                f"Expected: {list(sm_files)}. Missing: {missing}."
            )
            logger.warning(msg)
            if _ONTOLOGY_REQUIRED:
                raise OntologyLoadError(msg)
            _SM_GRAPH = None
            return None

        if missing and _ONTOLOGY_REQUIRED:
            raise OntologyLoadError(
                f"Required SM ontology files missing: {missing}"
            )

        _SM_GRAPH = g

    except OntologyLoadError:
        raise
    except Exception as exc:
        msg = f"[SM-ONTOLOGY] Load failed: {exc}"
        logger.warning(msg)
        if _ONTOLOGY_REQUIRED:
            raise OntologyLoadError(msg) from exc
        _SM_GRAPH = None

    return _SM_GRAPH


def _sparql_sm(query: str) -> list[dict]:
    """Execute a SPARQL SELECT against the Scrum Master graph."""
    g = _get_scrum_master_graph()
    if g is None:
        return []
    qres = g.query(query)
    return [{str(v): str(row[v]) for v in qres.vars} for row in qres]


def build_scrum_master_ontology_context() -> str:
    """
    Build a formatted ontology context block for the Scrum Master LLM path.

    Runs three SPARQL queries against agile.ttl + scrum_master.ttl:
      1. SprintHealthStatus individuals — health labels, descriptions, thresholds
      2. ImpedimentCategory individuals — blocker classification vocabulary
      3. RiskSignal individuals     — severity, description, remediation hints

    The result is a plain-text block injected into the SM LLM prompt so the
    model reasons about risks using the ontology vocabulary rather than
    freeform guessing.

    Kept in loader.py (not shared.py) so scrum_master_runner.py can import it
    without triggering the ChromaDB singletons instantiated in shared.py.
    """
    if _get_scrum_master_graph() is None:
        return "[SM-ONTOLOGY: unavailable]"

    lines = ["=== SCRUM MASTER ONTOLOGY CONTEXT ==="]

    # ── 1. Sprint health levels ───────────────────────────────────────────────
    health_rows = _sparql_sm("""
        PREFIX sm: <http://cognitwin.org/scrum_master#>
        SELECT ?label ?desc ?threshold WHERE {
            ?status sm:healthLabel ?label .
            OPTIONAL { ?status sm:healthDescription ?desc . }
            OPTIONAL { ?status sm:healthThreshold    ?threshold . }
        }
        ORDER BY ?label
    """)

    if health_rows:
        lines.append("  [SPRINT HEALTH STATES]")
        for r in health_rows:
            label     = r.get("label", "?")
            desc      = r.get("desc", "")
            threshold = r.get("threshold", "")
            lines.append(f"  • {label.upper()}: {desc}")
            if threshold:
                lines.append(f"    Koşul: {threshold}")

    # ── 2. Impediment categories ──────────────────────────────────────────────
    imp_rows = _sparql_sm("""
        PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX sm:   <http://cognitwin.org/scrum_master#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?label ?comment WHERE {
            ?cat rdf:type sm:ImpedimentCategory .
            OPTIONAL { ?cat rdfs:label   ?label . }
            OPTIONAL { ?cat rdfs:comment ?comment . }
        }
    """)

    if imp_rows:
        lines.append("  [ENGEL KATEGORİLERİ]")
        for r in imp_rows:
            label   = r.get("label", "?")
            comment = r.get("comment", "")
            lines.append(f"  • {label}: {comment}")

    # ── 3. Risk signals (sorted by severity desc in Python — safer than SPARQL
    #       ORDER BY on xsd:integer with some rdflib versions) ─────────────────
    risk_rows = _sparql_sm("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX sm:  <http://cognitwin.org/scrum_master#>
        SELECT ?label ?severity ?desc ?hint WHERE {
            ?signal rdf:type sm:RiskSignal .
            ?signal sm:riskLabel ?label .
            OPTIONAL { ?signal sm:severityLevel   ?severity . }
            OPTIONAL { ?signal sm:riskDescription ?desc . }
            OPTIONAL { ?signal sm:remediationHint ?hint . }
        }
    """)

    if risk_rows:
        # Sort highest severity first; default to 0 if missing
        risk_rows.sort(
            key=lambda r: int(r.get("severity", "0")),
            reverse=True,
        )
        _SEV = {"3": "KRİTİK", "2": "ORTA", "1": "DÜŞÜK"}
        lines.append("  [RİSK SİNYALLERİ — severity: 3=Kritik, 2=Orta, 1=Düşük]")
        for r in risk_rows:
            label    = r.get("label", "?")
            sev_raw  = r.get("severity", "?")
            sev_text = _SEV.get(sev_raw, sev_raw)
            desc     = r.get("desc", "")
            hint     = r.get("hint", "")
            lines.append(f"  • [{sev_text}] {label}: {desc}")
            if hint:
                lines.append(f"    Aksiyon: {hint}")

    if len(lines) == 1:
        lines.append("  (SM ontoloji bireyleri bulunamadı)")

    lines.append("=== END SCRUM MASTER ONTOLOGY CONTEXT ===")
    return "\n".join(lines)
