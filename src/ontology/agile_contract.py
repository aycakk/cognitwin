"""ontology/agile_contract.py — read-only helper for Ontology4Agile v1.3.0.

Phase 4 of the AGES-lite refactor: surface a small, deterministic API around
ontology4agile_v1_3_0.ttl so later phases can validate Scrum-shaped output
without performing any OWL reasoning.

Failure policy mirrors src/ontology/loader.py:
  - Default: missing TTL → return empty result, log a warning, no crash.
  - COGNITWIN_ONTOLOGY_REQUIRED=1 → raise OntologyLoadError on miss.

The graph is loaded lazily on first call and cached in module-level state.
The TTL file is treated as read-only.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from src.core.exceptions import OntologyLoadError

logger = logging.getLogger(__name__)

_AGILE_GRAPH: Optional[object] = None
_AGILE_TRIED: bool = False

_AGILE_TTL_FILENAME = "ontology4agile_v1_3_0.ttl"
_AGILE_NS = "http://example.org/agile#"
_AGILE_PREFIX = "PREFIX agile: <http://example.org/agile#>"
_RDFS_PREFIX = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>"
_RDF_PREFIX = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>"


def _ontology_required() -> bool:
    return os.environ.get("COGNITWIN_ONTOLOGY_REQUIRED", "0") == "1"


def load_agile_graph():
    """
    Load and cache the Ontology4Agile rdflib Graph.

    Returns the Graph if successfully loaded, None otherwise.
    Raises OntologyLoadError if COGNITWIN_ONTOLOGY_REQUIRED=1 and loading
    fails or the TTL is absent.
    """
    global _AGILE_GRAPH, _AGILE_TRIED
    if _AGILE_TRIED:
        return _AGILE_GRAPH
    _AGILE_TRIED = True

    onto_dir = Path(__file__).resolve().parents[2] / "ontologies"
    ttl_path = onto_dir / _AGILE_TTL_FILENAME

    try:
        from rdflib import Graph

        if not ttl_path.exists():
            msg = (
                f"[AGILE-ONTOLOGY] File not found: {ttl_path} — agile_contract "
                "helpers will return empty results. Set "
                "COGNITWIN_ONTOLOGY_REQUIRED=1 to treat this as a hard failure."
            )
            logger.warning(msg)
            if _ontology_required():
                raise OntologyLoadError(msg)
            _AGILE_GRAPH = None
            return None

        g = Graph()
        g.parse(str(ttl_path), format="turtle")
        logger.debug("[AGILE-ONTOLOGY] Loaded: %s", ttl_path)
        _AGILE_GRAPH = g

    except OntologyLoadError:
        raise
    except Exception as exc:
        msg = f"[AGILE-ONTOLOGY] Load failed: {exc}"
        logger.warning(msg)
        if _ontology_required():
            raise OntologyLoadError(msg) from exc
        _AGILE_GRAPH = None

    return _AGILE_GRAPH


def _local_name(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rsplit("/", 1)[-1]


def _sparql(query: str) -> list[dict]:
    g = load_agile_graph()
    if g is None:
        return []
    qres = g.query(query)
    return [{str(v): str(row[v]) for v in qres.vars} for row in qres]


# ---------------------------------------------------------------------------
# Public helpers — pure data, no global side effects beyond the graph cache.
# ---------------------------------------------------------------------------


def valid_scrum_events() -> set[str]:
    """Local names of all subclasses of agile:ScrumEvent."""
    rows = _sparql(
        f"""
        {_AGILE_PREFIX}
        {_RDFS_PREFIX}
        SELECT DISTINCT ?ev WHERE {{
            ?ev rdfs:subClassOf agile:ScrumEvent .
        }}
        """
    )
    return {_local_name(r["ev"]) for r in rows if r.get("ev")}


def valid_scrum_roles() -> set[str]:
    """Local names of all subclasses of agile:ScrumRole."""
    rows = _sparql(
        f"""
        {_AGILE_PREFIX}
        {_RDFS_PREFIX}
        SELECT DISTINCT ?r WHERE {{
            ?r rdfs:subClassOf agile:ScrumRole .
        }}
        """
    )
    return {_local_name(r["r"]) for r in rows if r.get("r")}


def valid_facilitators(event_name: str) -> set[str]:
    """
    Local names of roles that facilitate the given Scrum event.

    `event_name` is the local name (e.g. "SprintPlanning"). Returns an empty
    set if the event is unknown or the ontology is unavailable.
    """
    if not event_name or not event_name.replace("_", "").isalnum():
        return set()

    rows = _sparql(
        f"""
        {_AGILE_PREFIX}
        SELECT DISTINCT ?facilitator WHERE {{
            agile:{event_name} agile:facilitatedBy ?facilitator .
        }}
        """
    )
    return {_local_name(r["facilitator"]) for r in rows if r.get("facilitator")}


def valid_sprint_goal_states() -> set[str]:
    """
    Local names of all individuals typed as agile:SprintGoalState.
    Strips the canonical "SprintGoalState_" prefix when present.
    """
    rows = _sparql(
        f"""
        {_AGILE_PREFIX}
        {_RDF_PREFIX}
        SELECT DISTINCT ?state WHERE {{
            ?state rdf:type agile:SprintGoalState .
        }}
        """
    )
    states: set[str] = set()
    for r in rows:
        if not r.get("state"):
            continue
        name = _local_name(r["state"])
        if name.startswith("SprintGoalState_"):
            name = name[len("SprintGoalState_"):]
        states.add(name)
    return states


_ARTEFACT_RELATION_PROPERTIES: tuple[str, ...] = (
    "hasSprintBacklog",
    "hasSprintGoal",
    "producesIncrement",
    "validatesIncrement",
    "produces",
)


def valid_artefact_relations() -> set[tuple[str, str]]:
    """
    Canonical (domain, range) pairs for a curated set of artefact-shape
    ObjectProperties (hasSprintBacklog, hasSprintGoal, producesIncrement,
    validatesIncrement, produces). Returns an empty set if the ontology is
    unavailable.
    """
    pairs: set[tuple[str, str]] = set()
    for prop in _ARTEFACT_RELATION_PROPERTIES:
        rows = _sparql(
            f"""
            {_AGILE_PREFIX}
            {_RDFS_PREFIX}
            SELECT ?dom ?rng WHERE {{
                agile:{prop} rdfs:domain ?dom .
                agile:{prop} rdfs:range  ?rng .
            }}
            """
        )
        for r in rows:
            dom = r.get("dom")
            rng = r.get("rng")
            if dom and rng:
                pairs.add((_local_name(dom), _local_name(rng)))
    return pairs


def dod_conditions() -> tuple[str, ...]:
    """
    Local names of agile:DefinitionOfDoneCondition individuals.
    Empty when the ontology has only the class declaration without instances.
    """
    rows = _sparql(
        f"""
        {_AGILE_PREFIX}
        {_RDF_PREFIX}
        SELECT DISTINCT ?cond WHERE {{
            ?cond rdf:type agile:DefinitionOfDoneCondition .
        }}
        ORDER BY ?cond
        """
    )
    return tuple(_local_name(r["cond"]) for r in rows if r.get("cond"))


def _reset_cache_for_tests() -> None:
    """Test hook — clears the singleton cache. Not for production use."""
    global _AGILE_GRAPH, _AGILE_TRIED
    _AGILE_GRAPH = None
    _AGILE_TRIED = False
