"""
CapabilityManifest — declarative contract for CogniTwin agents.

Phase 1 of the Ontology4Agile / AGES-lite refactor: introduce a frozen,
immutable manifest type plus a small registry. No agent currently consumes
this; later phases will have BaseAgent and concrete agents return manifests
so governance, audit, and the UI can reason about agent capabilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CapabilityManifest:
    role: str
    intents: tuple[str, ...] = field(default_factory=tuple)
    inputs: tuple[str, ...] = field(default_factory=tuple)
    outputs: tuple[str, ...] = field(default_factory=tuple)
    gates_consumed: tuple[str, ...] = field(default_factory=tuple)
    ontology_classes_referenced: tuple[str, ...] = field(default_factory=tuple)


_MANIFESTS: dict[str, CapabilityManifest] = {}


def register_manifest(manifest: CapabilityManifest) -> None:
    if not isinstance(manifest, CapabilityManifest):
        raise TypeError(
            f"register_manifest expected CapabilityManifest, got {type(manifest).__name__}"
        )
    if not manifest.role:
        raise ValueError("CapabilityManifest.role must be a non-empty string")
    if manifest.role in _MANIFESTS:
        raise ValueError(
            f"CapabilityManifest already registered for role '{manifest.role}'"
        )
    _MANIFESTS[manifest.role] = manifest


def get_manifest(role: str) -> Optional[CapabilityManifest]:
    return _MANIFESTS.get(role)


def list_manifests() -> tuple[CapabilityManifest, ...]:
    return tuple(_MANIFESTS.values())


def _clear_registry_for_tests() -> None:
    _MANIFESTS.clear()
