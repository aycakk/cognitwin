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
