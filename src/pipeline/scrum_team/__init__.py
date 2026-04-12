"""pipeline/scrum_team — shared Scrum team infrastructure.

Both Scrum Master and Developer runners live in this package and
share a single SprintStateStore instance for sprint/task/backlog state.

Architectural rules enforced by this package:
  - NO personal footprint data
  - NO personal profile/identity memory
  - NO developer_id as a person key
  - Sprint state has a single owner (SprintStateStore)
  - Scrum Master writes; Developer reads only
"""
