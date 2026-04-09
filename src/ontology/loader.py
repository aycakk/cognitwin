"""ontology/loader.py — lazy rdflib graph loader and SPARQL helper.

Single source of truth for ontology infrastructure.
Previously defined as module-level helpers in src/services/api/pipeline.py;
moved here so that src/gates/evaluator.py can import them without
creating a circular dependency back into pipeline.py.

The graph is loaded once on first call and cached in module-level state.
Returns None silently when rdflib is unavailable or the .ttl files are missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# Module-level cache — loaded once per process, never shared across requests.
_ONTOLOGY_GRAPH: Optional[object] = None
_ONTOLOGY_TRIED: bool = False


def _get_ontology_graph():
    """Load and cache the rdflib Graph. Returns None if unavailable."""
    global _ONTOLOGY_GRAPH, _ONTOLOGY_TRIED
    if _ONTOLOGY_TRIED:
        return _ONTOLOGY_GRAPH
    _ONTOLOGY_TRIED = True
    try:
        from rdflib import Graph
        g = Graph()
        # loader.py lives at src/ontology/ → parents[2] is project root
        onto_dir = Path(__file__).resolve().parents[2] / "ontologies"
        loaded = False
        for fname in ("cognitwin-upper.ttl", "student_ontology.ttl"):
            p = onto_dir / fname
            if p.exists():
                g.parse(str(p), format="turtle")
                print(f"[ONTOLOGY] Loaded: {fname}")
                loaded = True
            else:
                print(f"[ONTOLOGY] Not found: {p}")
        _ONTOLOGY_GRAPH = g if loaded else None
    except Exception as exc:
        print(f"[ONTOLOGY] Load failed: {exc}")
        _ONTOLOGY_GRAPH = None
    return _ONTOLOGY_GRAPH


def _sparql(query: str) -> list[dict]:
    g = _get_ontology_graph()
    if g is None:
        return []
    qres = g.query(query)
    return [{str(v): str(row[v]) for v in qres.vars} for row in qres]
