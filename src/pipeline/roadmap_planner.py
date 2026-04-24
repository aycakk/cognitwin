"""pipeline/roadmap_planner.py — Product roadmap generation and persistence.

RoadmapPlanner converts backlog items and sprint goals into a structured
roadmap with release packages, target sprints, target dates, scope,
dependencies, and success criteria.

State is persisted under sprint_state.json["roadmap"] via SprintStateStore.
No LLM calls — roadmap entries are derived deterministically from backlog data.

Roadmap entry schema
--------------------
{
  "package_id":       "PKG-001",
  "release_package":  "MVP Authentication",
  "target_sprint":    "sprint-2",
  "target_date":      "2026-05-07",
  "scope":            ["S-001", "S-002"],
  "dependencies":     [],
  "success_criteria": ["User can register and log in", "Auth gates pass"],
  "status":           "planned",   # planned | in_progress | released
  "created_at":       "2026-04-24T12:00:00",
}
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Any

from src.pipeline.scrum_team.sprint_state_store import SprintStateStore

logger = logging.getLogger(__name__)

_SPRINT_DURATION_DAYS = 14


class RoadmapPlanner:
    """
    Derives a product roadmap from the current backlog and sprint state.

    Usage
    -----
    planner = RoadmapPlanner()
    entries = planner.build_from_backlog()
    print(planner.get_roadmap_text())
    """

    def __init__(self, state_store: SprintStateStore | None = None) -> None:
        self._store = state_store or SprintStateStore()

    # ─────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────

    def build_from_backlog(
        self,
        sprint_start_date: date | None = None,
        sprint_number_offset: int = 1,
    ) -> list[dict[str, Any]]:
        """Group backlog stories by deployment_package/epic and create roadmap entries.

        One entry is created per unique group. Entries are persisted via
        SprintStateStore.add_roadmap_entry().

        Parameters
        ----------
        sprint_start_date:
            Base date for calculating target sprint end dates. Defaults to today.
        sprint_number_offset:
            Sprint number to start from (default 1 → sprint-1).

        Returns
        -------
        List of roadmap entry dicts that were persisted.
        """
        backlog = self._store.get_backlog()
        if not backlog:
            logger.info("roadmap: backlog is empty — nothing to plan")
            return []

        base_date = sprint_start_date or date.today()
        groups    = self._group_by_package(backlog)
        entries: list[dict[str, Any]] = []

        for i, (package_name, stories) in enumerate(groups.items()):
            sprint_num  = sprint_number_offset + i
            sprint_id   = f"sprint-{sprint_num}"
            target_date = base_date + timedelta(days=_SPRINT_DURATION_DAYS * (i + 1))
            entry = self._build_entry(
                package_name=package_name,
                stories=stories,
                sprint_id=sprint_id,
                target_date=target_date,
                package_index=i + 1,
            )
            self._store.add_roadmap_entry(entry)
            entries.append(entry)

        logger.info("roadmap: created %d entries", len(entries))
        return entries

    def get_roadmap_text(self) -> str:
        """Return a human-readable roadmap summary."""
        roadmap = self._store.get_roadmap()
        if not roadmap:
            return "Roadmap is empty. Call build_from_backlog() to generate it."

        lines = ["=== PRODUCT ROADMAP ==="]
        for entry in roadmap:
            lines.append(
                f"\n[{entry.get('package_id', '?')}] {entry.get('release_package', '-')}"
            )
            lines.append(f"  Target Sprint : {entry.get('target_sprint', 'TBD')}")
            lines.append(f"  Target Date   : {entry.get('target_date', 'TBD')}")
            lines.append(f"  Status        : {entry.get('status', 'planned')}")

            scope = entry.get("scope", [])
            if scope:
                lines.append(f"  Scope         : {', '.join(scope)}")

            deps = entry.get("dependencies", [])
            if deps:
                lines.append(f"  Dependencies  : {', '.join(deps)}")

            sc = entry.get("success_criteria", [])
            if sc:
                lines.append("  Success Criteria:")
                for criterion in sc:
                    lines.append(f"    • {criterion}")

        lines.append("\n=== END ROADMAP ===")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────
    #  Internals
    # ─────────────────────────────────────────────────────────────────

    def _group_by_package(
        self, backlog: list[dict]
    ) -> dict[str, list[dict]]:
        """Group stories by deployment_package → epic → 'Core' fallback."""
        groups: dict[str, list[dict]] = {}
        for story in backlog:
            if story.get("status") in ("accepted", "rejected"):
                continue
            key = (
                story.get("deployment_package")
                or story.get("epic")
                or "Core"
            )
            groups.setdefault(key, []).append(story)
        return groups

    def _build_entry(
        self,
        package_name: str,
        stories: list[dict],
        sprint_id: str,
        target_date: date,
        package_index: int,
    ) -> dict[str, Any]:
        story_ids = [s.get("story_id", "") for s in stories if s.get("story_id")]
        # Collect up to 4 acceptance criteria across stories as success criteria
        ac_pool: list[str] = []
        for s in stories:
            ac_pool.extend(s.get("acceptance_criteria", [])[:2])

        return {
            "package_id":       f"PKG-{package_index:03d}",
            "release_package":  package_name[:80],
            "target_sprint":    sprint_id,
            "target_date":      str(target_date),
            "scope":            story_ids,
            "dependencies":     [],
            "success_criteria": ac_pool[:4],
            "status":           "planned",
            "created_at":       datetime.now().isoformat(timespec="seconds"),
        }
