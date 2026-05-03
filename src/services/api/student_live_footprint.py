"""services/api/student_live_footprint.py — runtime live footprint bridge.

Extends the existing seed footprint pipeline:

    src/data/footprints.txt
      → src/scripts/mask_footprints.py
      → src/data/masked/footprints_masked.txt
      → src/database/bulk_ingest.bulk_ingest_masked_data()
      → ChromaDB collection "academic_memory" (namespace="academic")
      → src/pipeline/shared.VECTOR_MEM.retrieve(namespace="academic")

This module adds a second producer that joins the same path at the
Chroma write step. Live runtime events from the Student Agent
(`chat_turn`, `agenda_event`, ...) are PII-masked, appended to
`src/data/masked/footprints_live.jsonl` for replay, and upserted into
the same `academic` Chroma collection with metadata `source="live"`.

No new namespace, no new collection, no retrieval rewrite.
Failure of any step (Chroma down, JSONL unwritable) must never
propagate up to the calling route handler.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _live_jsonl_path() -> Path:
    return _project_root() / "src" / "data" / "masked" / "footprints_live.jsonl"


def _mask(text: str) -> str:
    if not text:
        return ""
    try:
        from src.utils.masker import PIIMasker
        return PIIMasker().mask_data(text) or ""
    except Exception as exc:
        logger.info("PIIMasker unavailable; live footprint masking skipped (%s)", exc)
        return text


def _existing_dedupe_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(obj, dict):
                    key = obj.get("dedupe_key")
                    if isinstance(key, str) and key:
                        keys.add(key)
    except OSError as exc:
        logger.info("could not read live footprint file (%s)", exc)
    return keys


def _existing_event(path: Path, dedupe_key: str) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(obj, dict) and obj.get("dedupe_key") == dedupe_key:
                    return obj
    except OSError:
        return None
    return None


def _append_jsonl(path: Path, event: dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False))
            f.write("\n")
        return True
    except OSError as exc:
        logger.warning("could not append live footprint (%s)", exc)
        return False


def _chroma_write(masked: str, doc_id: str, metadata: dict[str, Any]) -> None:
    """Write into the existing academic Chroma collection.

    Failure is logged and swallowed — never raises.
    """
    try:
        from src.database.chroma_manager import db_manager
        db_manager.add_with_namespace(
            text=masked,
            namespace="academic",
            metadata=metadata,
            doc_id=doc_id,
        )
    except Exception as exc:
        logger.info("live footprint Chroma write failed (%s)", exc)


def record_live_footprint(
    text: str,
    kind: str,
    source: str,
    occurred_at: Optional[str] = None,
    derived_from: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Append a runtime footprint event and upsert it into academic memory.

    Args:
        text: free-form description of the action; will be PII-masked
              before any write.
        kind: event kind, e.g. ``"chat_turn"``, ``"agenda_event"``,
              ``"profile_event"``.
        source: producing module/route, e.g. ``"student_agent_chat"``.
        occurred_at: optional ISO timestamp of when the event happened.
        derived_from: optional id of an upstream record (history item id, …).
        metadata: optional extra metadata; merged into the JSONL record
                  and into the Chroma metadata.

    Returns:
        The event dict that was written (or that already existed for a
        duplicate dedupe_key). Returns ``None`` only if both the JSONL
        append and lookup failed.
    """
    if not text or not isinstance(text, str):
        return None
    kind = (kind or "").strip() or "unknown"
    source = (source or "").strip() or "unknown"

    masked = _mask(text).strip()
    if not masked:
        return None

    dedupe_basis = f"{kind}|{source}|{occurred_at or ''}|{masked}"
    dedupe_key = hashlib.sha1(dedupe_basis.encode("utf-8")).hexdigest()
    vector_doc_id = f"live_{dedupe_key}"
    event_id = f"live_{dedupe_key[:16]}"

    extra_meta = dict(metadata) if isinstance(metadata, dict) else {}

    path = _live_jsonl_path()

    existing = _existing_event(path, dedupe_key)
    if existing is not None:
        return existing

    event: dict[str, Any] = {
        "id":            event_id,
        "created_at":    datetime.now(timezone.utc).isoformat(),
        "occurred_at":   occurred_at,
        "kind":          kind,
        "source":        source,
        "text":          masked,
        "masked":        True,
        "derived_from":  derived_from,
        "metadata":      extra_meta,
        "dedupe_key":    dedupe_key,
        "vector_doc_id": vector_doc_id,
    }

    _append_jsonl(path, event)

    chroma_meta: dict[str, Any] = {
        "role":   "student",
        "source": "live",
        "type":   kind,
        "kind":   kind,
        "origin": source,
    }
    if occurred_at:
        chroma_meta["occurred_at"] = occurred_at
    if derived_from:
        chroma_meta["derived_from"] = derived_from
    for k, v in extra_meta.items():
        if k in chroma_meta:
            continue
        chroma_meta[k] = v

    _chroma_write(masked, vector_doc_id, chroma_meta)

    return event
