from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DeveloperProfileStore:
    """Lightweight JSON-backed profile memory for developer personalization."""

    def __init__(
        self,
        profile_dir: str | Path | None = None,
        max_decision_history: int = 30,
    ) -> None:
        if profile_dir is None:
            project_root = Path(__file__).resolve().parents[2]
            profile_dir = project_root / "data" / "developer_profiles"

        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.max_decision_history = max(1, int(max_decision_history))

    def load_profile(self, developer_id: str) -> dict[str, Any]:
        normalized_id = self._normalize_developer_id(developer_id)
        profile_path = self._profile_path(normalized_id)

        if not profile_path.exists():
            profile = self.create_default_profile(normalized_id)
            return self.save_profile(profile)

        try:
            data = json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}

        profile = self._ensure_profile_shape(data, normalized_id)
        if profile != data:
            profile = self.save_profile(profile)
        return profile

    def save_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        developer_id = self._normalize_developer_id(str((profile or {}).get("developer_id") or "developer-default"))
        shaped = self._ensure_profile_shape(profile or {}, developer_id)
        shaped["last_updated"] = self._now_iso()

        profile_path = self._profile_path(developer_id)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(
            json.dumps(shaped, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return shaped

    def update_profile(
        self,
        profile: dict[str, Any],
        interaction_data: dict[str, Any],
    ) -> dict[str, Any]:
        interaction_data = interaction_data or {}
        context = interaction_data.get("context", {}) or {}
        if not isinstance(context, dict):
            context = {}

        developer_id = self._normalize_developer_id(
            str(
                (profile or {}).get("developer_id")
                or interaction_data.get("developer_id")
                or "developer-default"
            )
        )
        base = self._ensure_profile_shape(profile or {}, developer_id)

        request_text = str(interaction_data.get("request") or context.get("source_prompt") or "")
        solution_text = str(interaction_data.get("solution") or "")
        combined_text = f"{request_text}\n{solution_text}".lower()

        provided_frameworks = interaction_data.get("preferred_frameworks", [])
        preferred_frameworks = self._merge_strings(
            base.get("preferred_frameworks", []),
            list(provided_frameworks) if isinstance(provided_frameworks, list) else [],
        )

        framework_keywords = {
            "fastapi": "fastapi",
            "django": "django",
            "flask": "flask",
            "react": "react",
            "next.js": "nextjs",
            "nextjs": "nextjs",
            "vue": "vue",
            "angular": "angular",
            "express": "express",
            "nestjs": "nestjs",
            "spring": "spring",
        }
        inferred_frameworks = [
            label for token, label in framework_keywords.items() if token in combined_text
        ]
        base["preferred_frameworks"] = self._merge_strings(preferred_frameworks, inferred_frameworks)

        architecture_preferences = list(base.get("architecture_preferences", []))
        architecture_keywords = {
            "microservice": "microservices",
            "microservices": "microservices",
            "monolith": "monolith",
            "event driven": "event-driven",
            "event-driven": "event-driven",
            "clean architecture": "clean-architecture",
            "hexagonal": "hexagonal",
            "layered": "layered",
        }
        inferred_architecture = [
            label for token, label in architecture_keywords.items() if token in combined_text
        ]
        base["architecture_preferences"] = self._merge_strings(
            architecture_preferences,
            inferred_architecture,
        )

        planning_style = "mvp-focused"
        if any(token in combined_text for token in ("tradeoff", "compare", "analyze", "analiz", "alternatif")):
            planning_style = "analysis-first"

        coding_style = dict(base.get("coding_style", {}))
        language = str(context.get("language") or coding_style.get("preferred_language") or "en")
        coding_style["preferred_language"] = language
        coding_style["task_type"] = str(context.get("task_type") or coding_style.get("task_type") or "other")
        coding_style["strategy"] = str(context.get("strategy") or coding_style.get("strategy") or "auto")
        coding_style["planning_style"] = planning_style
        base["coding_style"] = coding_style

        error_handling_style = dict(base.get("error_handling_style", {}))
        if "fail-fast" in combined_text or "fail fast" in combined_text:
            error_handling_style["failure_mode"] = "fail-fast"
        elif "graceful" in combined_text:
            error_handling_style["failure_mode"] = "graceful-degradation"
        if "retry" in combined_text:
            error_handling_style["retry_policy"] = "enabled"
        base["error_handling_style"] = error_handling_style

        documentation_style = dict(base.get("documentation_style", {}))
        if any(token in combined_text for token in ("concise", "short", "brief")):
            documentation_style["verbosity"] = "concise"
        elif any(token in combined_text for token in ("detailed", "comprehensive", "deep")):
            documentation_style["verbosity"] = "detailed"
        else:
            documentation_style.setdefault("verbosity", "balanced")
        if "markdown" in combined_text:
            documentation_style["format"] = "markdown"
        base["documentation_style"] = documentation_style

        confidence = 0.0
        validation_report = interaction_data.get("validation_report", {})
        if isinstance(validation_report, dict):
            try:
                confidence = float(validation_report.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        decision_history = list(base.get("decision_history", []))
        decision_history.append(
            {
                "timestamp": self._now_iso(),
                "request_summary": self._summarize(request_text),
                "task_type": coding_style.get("task_type", "other"),
                "strategy": coding_style.get("strategy", "auto"),
                "planning_style": planning_style,
                "confidence": round(confidence, 2),
            }
        )
        base["decision_history"] = decision_history[-self.max_decision_history :]
        base["style_vector"] = self._build_style_vector(base)

        return self.save_profile(base)

    def create_default_profile(self, developer_id: str) -> dict[str, Any]:
        normalized_id = self._normalize_developer_id(developer_id)
        return self._base_profile(
            developer_id=normalized_id,
            last_updated=self._now_iso(),
        )

    def _profile_path(self, developer_id: str) -> Path:
        return self.profile_dir / f"{developer_id}.json"

    def _base_profile(self, developer_id: str, last_updated: str) -> dict[str, Any]:
        return {
            "developer_id": developer_id,
            "preferred_frameworks": [],
            "coding_style": {},
            "architecture_preferences": [],
            "error_handling_style": {},
            "documentation_style": {},
            "decision_history": [],
            "style_vector": [],
            "last_updated": last_updated,
        }

    def _ensure_profile_shape(self, profile: dict[str, Any], developer_id: str) -> dict[str, Any]:
        if not isinstance(profile, dict):
            profile = {}

        last_updated = profile.get("last_updated")
        if not isinstance(last_updated, str) or not last_updated.strip():
            last_updated = self._now_iso()

        normalized = self._base_profile(developer_id=developer_id, last_updated=last_updated)

        normalized["preferred_frameworks"] = self._merge_strings(
            [],
            profile.get("preferred_frameworks", []),
        )
        normalized["architecture_preferences"] = self._merge_strings(
            [],
            profile.get("architecture_preferences", []),
        )

        for object_key in ("coding_style", "error_handling_style", "documentation_style"):
            value = profile.get(object_key, {})
            normalized[object_key] = dict(value) if isinstance(value, dict) else {}

        decision_history = profile.get("decision_history", [])
        if isinstance(decision_history, list):
            normalized["decision_history"] = [entry for entry in decision_history if isinstance(entry, dict)]

        style_vector = profile.get("style_vector", [])
        normalized["style_vector"] = self._coerce_numeric_list(style_vector) if isinstance(style_vector, list) else []

        return normalized

    def _build_style_vector(self, profile: dict[str, Any]) -> list[float]:
        history = profile.get("decision_history", [])
        confidences = []
        for item in history:
            if not isinstance(item, dict):
                continue
            value = item.get("confidence")
            try:
                confidences.append(float(value))
            except Exception:
                continue

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        verbosity = str((profile.get("documentation_style") or {}).get("verbosity") or "balanced")
        verbosity_score_map = {
            "concise": 0.3,
            "balanced": 0.6,
            "detailed": 0.9,
        }
        verbosity_score = verbosity_score_map.get(verbosity, 0.6)

        return [
            float(len(profile.get("preferred_frameworks", []))),
            float(len(profile.get("architecture_preferences", []))),
            float(len(history)),
            round(float(avg_confidence), 2),
            float(verbosity_score),
        ]

    def _merge_strings(self, current_values: list[Any], new_values: list[Any]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()

        for value in list(current_values) + list(new_values):
            if not isinstance(value, str):
                continue
            cleaned = value.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            merged.append(cleaned)
        return merged

    def _coerce_numeric_list(self, values: list[Any]) -> list[float]:
        coerced: list[float] = []
        for value in values:
            try:
                coerced.append(float(value))
            except Exception:
                continue
        return coerced

    def _normalize_developer_id(self, developer_id: str) -> str:
        candidate = (developer_id or "").strip()
        if not candidate:
            return "developer-default"
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate)
        return cleaned.strip("._-") or "developer-default"

    def _summarize(self, text: str, max_len: int = 180) -> str:
        compact = " ".join((text or "").split())
        if not compact:
            return ""
        if len(compact) <= max_len:
            return compact
        return compact[: max_len - 3] + "..."

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
