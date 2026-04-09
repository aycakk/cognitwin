"""agents/scrum_master_agent.py — Rule-based Scrum Master coordination agent.

This agent does NOT use ChromaDB / vector memory.  It operates on local
sprint state persisted at data/sprint_state.json using deterministic rule
logic.  Coordination with the Developer pipeline happens through shared
sprint state: developer_runner (or any caller) can call
ScrumMasterAgent.get_current_assignments() to read the active task list.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
#  Sprint state location
# ─────────────────────────────────────────────────────────────────────────────

# scrum_master_agent.py lives at src/agents/ ->parents[2] is the project root
_PROJECT_ROOT     = Path(__file__).resolve().parents[2]
_SPRINT_STATE_PATH = _PROJECT_ROOT / "data" / "sprint_state.json"

_DEFAULT_SPRINT_STATE: dict[str, Any] = {
    "sprint": {
        "id":       "sprint-1",
        "goal":     "Sprint hedefi henüz tanımlanmamış.",
        "start":    str(date.today()),
        "end":      "",
        "velocity": 0,
    },
    "tasks": [],
    "team": [
        {"id": "developer-default", "role": "Developer", "capacity": 8},
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
#  Intent detection — Turkish + English keywords
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("assign",        re.compile(
        r"\bata\b|assign|ver\s+görevi|devret|görevi\s+ver|üzerine\s+al", re.I)),
    ("blockers",      re.compile(
        r"engell?e|blocker|impedi|tıkan|durdu|askıda|blocked", re.I)),
    ("sprint_status", re.compile(
        r"sprint\s+durum|ilerleme|progress|ne\s+kadar|tamamland|status\b", re.I)),
    ("standup",       re.compile(
        r"standup|daily\b|günlük|dün\s+ne|bugün\s+ne|gündem", re.I)),
    ("retrospective", re.compile(
        r"retro|geribild|iyileştir|öğren|aksayan|retrospektif", re.I)),
    ("review",        re.compile(
        r"\breview\b|sprint\s+inceleme|demo\b|göster|teslim\s+et", re.I)),
    ("delegate",      re.compile(
        r"delegasyon|deleg[ae]|kim\s+yap|kime\s+ver|kim\s+üstlen|distribute|dağıt", re.I)),
    ("add_task",      re.compile(
        r"görev\s+ekle|yeni\s+görev|backlog.?a\s+ekle|sprint.?e\s+ekle|add\s+task", re.I)),
    ("update_task",   re.compile(
        r"güncelle|update\s+task|durum\s+değiştir|status\s+set|tamamlandı\s+olarak", re.I)),
    ("set_goal",      re.compile(
        r"sprint\s+hedef|hedef\s+belirle|goal\s+set|set\s+goal", re.I)),
]


class ScrumMasterAgent:
    """
    Rule-based Scrum Master coordination agent.

    Reads and writes local sprint state (data/sprint_state.json).
    Applies deterministic rules for task assignment, blocker detection,
    sprint health aggregation, ceremony facilitation, and developer delegation.

    Developer pipeline integration
    ────────────────────────────────
    Call get_current_assignments() to inject active sprint tasks into
    any other pipeline context (e.g. developer_runner's codebase context).
    Call get_sprint_goal() to surface the current sprint goal string.
    """

    def __init__(self) -> None:
        _SPRINT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    #  Public API (called by scrum_master_runner and external callers)
    # ─────────────────────────────────────────────────────────────────────────

    def handle_query(self, query: str) -> str:
        """
        Route the query to the appropriate rule handler and return a
        structured text response.  No LLM is called here.
        """
        intent = self._detect_intent(query)
        state  = self._load_state()

        handlers = {
            "assign":        lambda: self._handle_assign(query, state),
            "blockers":      lambda: self._handle_blockers(state),
            "sprint_status": lambda: self._handle_sprint_status(state),
            "standup":       lambda: self._handle_standup(state),
            "retrospective": lambda: self._handle_retrospective(state),
            "review":        lambda: self._handle_review(state),
            "delegate":      lambda: self._handle_delegate(query, state),
            "add_task":      lambda: self._handle_add_task(query, state),
            "update_task":   lambda: self._handle_update_task(query, state),
            "set_goal":      lambda: self._handle_set_goal(query, state),
        }
        return handlers.get(intent, lambda: self._handle_general(state))()

    def get_current_assignments(self) -> list[dict]:
        """
        Return non-done assigned tasks for consumption by other pipeline
        components (e.g. developer_runner sprint context injection).
        """
        state = self._load_state()
        return [
            t for t in state.get("tasks", [])
            if t.get("assignee") and t.get("status") != "done"
        ]

    def get_sprint_goal(self) -> str:
        """Return the current sprint goal string."""
        return self._load_state().get("sprint", {}).get(
            "goal", "Sprint hedefi tanımlanmamış."
        )

    def get_blocked_tasks(self) -> list[dict]:
        """Return all blocked tasks (for external monitoring)."""
        return [
            t for t in self._load_state().get("tasks", [])
            if t.get("status") == "blocked"
        ]

    # ─────────────────────────────────────────────────────────────────────────
    #  State I/O
    # ─────────────────────────────────────────────────────────────────────────

    def _load_state(self) -> dict[str, Any]:
        if not _SPRINT_STATE_PATH.exists():
            self._save_state(_DEFAULT_SPRINT_STATE)
            return dict(_DEFAULT_SPRINT_STATE)
        try:
            return json.loads(_SPRINT_STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return dict(_DEFAULT_SPRINT_STATE)

    def _save_state(self, state: dict[str, Any]) -> None:
        _SPRINT_STATE_PATH.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Intent detection
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_intent(self, query: str) -> str:
        for intent, pattern in _INTENT_PATTERNS:
            if pattern.search(query):
                return intent
        return "general"

    # ─────────────────────────────────────────────────────────────────────────
    #  Rule Handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_assign(self, query: str, state: dict) -> str:
        """
        Parse task ID and developer ID from query, record the assignment.

        Rule: task must exist in the sprint backlog; developer must be in the
        team roster.  Falls back to 'developer-default' if no match.
        """
        tasks    = state.get("tasks", [])
        team     = state.get("team", [])
        team_ids = {m["id"] for m in team}

        task_match = re.search(r"\b(T-\d+)\b", query, re.I)
        dev_match  = re.search(r"(developer[-\w]*)", query, re.I)

        if not task_match:
            return (
                "Görev atama için bir görev ID'si gerekli (örn. T-001).\n"
                f"Sprint backlog'undaki görevler: {self._task_list_short(tasks)}"
            )

        task_id = task_match.group(1).upper()
        dev_id  = dev_match.group(1).lower() if dev_match else "developer-default"
        if dev_id not in team_ids:
            dev_id = "developer-default"

        task = next((t for t in tasks if t["id"] == task_id), None)
        if task is None:
            return (
                f"Görev {task_id} sprint backlog'unda bulunamadı.\n"
                f"Mevcut görevler: {self._task_list_short(tasks)}"
            )

        task["assignee"]    = dev_id
        task["assigned_at"] = datetime.now().isoformat(timespec="seconds")
        if task.get("status") == "todo":
            task["status"] = "in_progress"
        self._save_state(state)

        return (
            f"Görev Atandı\n"
            f"  ID       : {task_id}\n"
            f"  Başlık   : {task.get('title', '-')}\n"
            f"  Atanan   : {dev_id}\n"
            f"  Durum    : {task['status']}\n"
            f"  Öncelik  : {task.get('priority', 'medium')}\n"
            f"  Sprint   : {state['sprint']['id']}"
        )

    def _handle_blockers(self, state: dict) -> str:
        """List blocked tasks with owner and blocker description."""
        tasks   = state.get("tasks", [])
        blocked = [t for t in tasks if t.get("status") == "blocked"]

        if not blocked:
            return (
                f"Sprint '{state['sprint']['id']}' içinde şu an "
                "engellenmiş görev bulunmuyor.\n"
                "Sprint akışı sağlıklı görünüyor."
            )

        lines = [
            f"Engellenmiş Görevler — Sprint: {state['sprint']['id']}",
            f"Hedef: {state['sprint'].get('goal', '-')}",
            "",
        ]
        for t in blocked:
            lines += [
                f"  • [{t['id']}] {t.get('title', '-')}",
                f"    Atanan  : {t.get('assignee', 'atanmamış')}",
                f"    Engel   : {t.get('blocker', 'açıklama girilmemiş')}",
                f"    Öncelik : {t.get('priority', 'medium')}",
                "",
            ]
        lines.append(
            f"Toplam {len(blocked)} engellenmiş görev. "
            "Bu engelleri sprint günü içinde çözün."
        )
        return "\n".join(lines)

    def _handle_sprint_status(self, state: dict) -> str:
        """Aggregate task counts by status and compute story-point completion."""
        tasks  = state.get("tasks", [])
        sprint = state.get("sprint", {})

        if not tasks:
            return (
                f"Sprint '{sprint.get('id', '?')}' henüz görev içermiyor.\n"
                "Sprint backlog'unu oluşturmak için 'görev ekle: <başlık>' "
                "komutunu kullanın."
            )

        counts: dict[str, int] = {"todo": 0, "in_progress": 0, "blocked": 0, "done": 0}
        total_pts = done_pts = 0

        for t in tasks:
            s = t.get("status", "todo")
            counts[s] = counts.get(s, 0) + 1
            pts = int(t.get("story_points", 0) or 0)
            total_pts += pts
            if s == "done":
                done_pts += pts

        completion = (done_pts / total_pts * 100) if total_pts else 0.0
        health = (
            "Kritik" if counts["blocked"] > 1
            else "Risk Var" if counts["blocked"] == 1
            else "İyi"
        )

        return "\n".join([
            f"Sprint Durum Raporu — {sprint.get('id', '?')}",
            f"  Hedef       : {sprint.get('goal', '-')}",
            f"  Tarih       : {sprint.get('start', '-')} ->{sprint.get('end', '-')}",
            f"  Toplam      : {len(tasks)} görev",
            f"  Yapılacak   : {counts['todo']}",
            f"  Devam Eden  : {counts['in_progress']}",
            f"  Engellenmiş : {counts['blocked']}",
            f"  Tamamlandı  : {counts['done']}",
            f"  Tamamlanma  : %{completion:.0f}  (story point bazlı)",
            f"  Sprint Sağlığı: {health}",
        ])

    def _handle_standup(self, state: dict) -> str:
        """Generate a structured standup prompt for the team."""
        tasks   = state.get("tasks", [])
        sprint  = state.get("sprint", {})
        in_prog = [t for t in tasks if t.get("status") == "in_progress"]
        blocked = [t for t in tasks if t.get("status") == "blocked"]

        lines = [
            f"Günlük Standup — {date.today()} | Sprint: {sprint.get('id', '?')}",
            f"Hedef: {sprint.get('goal', '-')}",
            "",
            "Her takım üyesi için:",
            "  1. Dün ne tamamladın?",
            "  2. Bugün ne yapacaksın?",
            "  3. Önünde engel var mı?",
        ]

        if in_prog:
            lines += ["", "Devam Eden Görevler:"]
            for t in in_prog:
                lines.append(
                    f"  • [{t['id']}] {t.get('title', '-')}"
                    f" ->{t.get('assignee', 'atanmamış')}"
                )

        if blocked:
            lines += ["", "⚠ Engellenmiş (Acil İnceleme):"]
            for t in blocked:
                lines.append(
                    f"  • [{t['id']}] {t.get('title', '-')}"
                    f" | Engel: {t.get('blocker', '?')}"
                )

        lines += ["", "Scrum Master Notu: Engelleri sprint günü içinde çözün."]
        return "\n".join(lines)

    def _handle_retrospective(self, state: dict) -> str:
        """Generate retrospective prompts based on sprint outcome."""
        sprint  = state.get("sprint", {})
        tasks   = state.get("tasks", [])
        done    = [t for t in tasks if t.get("status") == "done"]
        blocked = [t for t in tasks if t.get("status") == "blocked"]
        todo    = [t for t in tasks if t.get("status") == "todo"]

        lines = [
            f"Sprint Retrospektif — {sprint.get('id', '?')}",
            f"Hedef: {sprint.get('goal', '-')}",
            "",
            "İyi Gidenler (Keep):",
            "  • Neler beklendiği gibi ilerledi?",
            "  • Hangi pratikler bir sonraki sprintte devam etmeli?",
            "",
            "İyileştirilebilecekler (Improve):",
            "  • Neler beklenenden yavaş veya zorlu ilerledi?",
        ]
        if blocked:
            lines.append(f"  • {len(blocked)} görev engellendi — nedenleri nelerdi?")
        if todo:
            lines.append(f"  • {len(todo)} görev tamamlanamadı — neden?")

        lines += [
            "",
            "Aksiyon Maddeleri (Actions):",
            "  • Bu sprintten öğrenilen en önemli ders nedir?",
            "  • Bir sonraki sprintte değiştirmek istediğiniz bir şey nedir?",
            "",
            f"Özet: {len(done)} tamamlandı | "
            f"{len(todo)} kaldı | "
            f"{len(blocked)} engellenmiş",
        ]
        return "\n".join(lines)

    def _handle_review(self, state: dict) -> str:
        """Generate sprint review / demo summary."""
        sprint = state.get("sprint", {})
        tasks  = state.get("tasks", [])
        done   = [t for t in tasks if t.get("status") == "done"]
        remain = [t for t in tasks if t.get("status") != "done"]

        total_pts = sum(int(t.get("story_points", 0) or 0) for t in tasks)
        done_pts  = sum(int(t.get("story_points", 0) or 0) for t in done)

        lines = [
            f"Sprint Review — {sprint.get('id', '?')}",
            f"  Hedef  : {sprint.get('goal', '-')}",
            f"  Tarih  : {sprint.get('start', '-')} ->{sprint.get('end', '-')}",
            "",
            f"Tamamlanan ({len(done)}/{len(tasks)}):",
        ]
        for t in done:
            lines.append(f"  [OK][{t['id']}] {t.get('title', '-')}")

        if remain:
            lines += ["", f"Tamamlanmayan ({len(remain)}):"]
            for t in remain:
                lines.append(
                    f"  [--][{t['id']}] {t.get('title', '-')} — {t.get('status', '-')}"
                )

        lines.append(
            f"\nVelocity: {done_pts} / {total_pts} story point tamamlandı."
        )
        return "\n".join(lines)

    def _handle_delegate(self, query: str, state: dict) -> str:
        """
        Suggest which developer should handle each unassigned task.

        Rule:
          - Count active (non-done) tasks per developer.
          - Assign the next unassigned task to the developer with the lowest load.
          - Ties broken by team roster order.
        """
        tasks  = state.get("tasks", [])
        team   = state.get("team", [])

        if not team:
            return "Takımda kayıtlı geliştirici bulunamadı."

        load: dict[str, int] = {m["id"]: 0 for m in team}
        for t in tasks:
            assignee = t.get("assignee")
            if assignee and t.get("status") not in ("done",) and assignee in load:
                load[assignee] += 1

        sorted_devs = sorted(team, key=lambda m: load.get(m["id"], 0))
        unassigned  = [
            t for t in tasks
            if not t.get("assignee") and t.get("status") == "todo"
        ]

        lines = ["Görev Dağılımı Önerisi (Scrum Master Delegasyon Kuralı):", ""]

        if not unassigned:
            lines.append("Atanmayı bekleyen görev yok — tüm görevler atanmış durumda.")
        else:
            for i, task in enumerate(unassigned):
                dev    = sorted_devs[i % len(sorted_devs)]
                reason = (
                    "en düşük yük"
                    if load.get(dev["id"], 0) == min(load.values())
                    else "kapasite dengesi"
                )
                lines.append(
                    f"  • [{task['id']}] {task.get('title', '-')}"
                    f" ->öneri: {dev['id']}"
                    f"  ({reason}, aktif görev: {load.get(dev['id'], 0)})"
                )

        lines += ["", "Mevcut Geliştirici Yükleri:"]
        for m in team:
            lines.append(f"  {m['id']}: {load.get(m['id'], 0)} aktif görev")

        return "\n".join(lines)

    def _handle_add_task(self, query: str, state: dict) -> str:
        """
        Add a new task to the sprint backlog.

        Parses the title from text after 'görev ekle:' / 'add task:' or
        uses the entire query as the title.  Generates the next T-NNN ID.
        """
        tasks = state.get("tasks", [])

        title_match = re.search(
            r"(?:görev\s+ekle|add\s+task|yeni\s+görev|backlog.?a\s+ekle"
            r"|sprint.?e\s+ekle)[:\s]+(.+)",
            query, re.I,
        )
        title = title_match.group(1).strip() if title_match else query.strip()

        nums   = [
            int(t["id"].split("-")[1])
            for t in tasks
            if re.match(r"T-\d+", t.get("id", ""))
        ] or [0]
        new_id = f"T-{(max(nums) + 1):03d}"

        new_task: dict[str, Any] = {
            "id":           new_id,
            "title":        title[:120],
            "type":         "story",
            "status":       "todo",
            "assignee":     None,
            "priority":     "medium",
            "story_points": 0,
            "blocker":      None,
            "created_at":   datetime.now().isoformat(timespec="seconds"),
        }
        tasks.append(new_task)
        state["tasks"] = tasks
        self._save_state(state)

        return (
            f"Yeni görev eklendi\n"
            f"  ID      : {new_id}\n"
            f"  Başlık  : {title[:80]}\n"
            f"  Durum   : todo\n"
            f"  Sprint  : {state['sprint']['id']}\n"
            "  Not: Atamak için ->'{id} developer-default üzerine ata'"
        )

    def _handle_update_task(self, query: str, state: dict) -> str:
        """
        Update task status.

        Looks for a T-NNN ID and a status keyword in the query.
        Status keywords: todo, in_progress/devam, blocked/engel, done/tamamlandı.
        """
        tasks = state.get("tasks", [])

        task_match   = re.search(r"\b(T-\d+)\b", query, re.I)
        status_match = re.search(
            r"\b(todo|in[_\s]progress|devam|blocked|engel|done|tamamlandı)\b",
            query, re.I,
        )

        if not task_match:
            return (
                "Görev güncelleme için görev ID'si gerekli (örn. T-001).\n"
                f"Mevcut görevler: {self._task_list_short(tasks)}"
            )

        task_id = task_match.group(1).upper()
        task    = next((t for t in tasks if t["id"] == task_id), None)
        if task is None:
            return f"Görev {task_id} bulunamadı."

        if status_match:
            raw = status_match.group(1).lower()
            status_map = {
                "todo": "todo",
                "in_progress": "in_progress",
                "in progress": "in_progress",
                "devam": "in_progress",
                "blocked": "blocked",
                "engel": "blocked",
                "done": "done",
                "tamamlandı": "done",
            }
            new_status = status_map.get(raw, raw)
            task["status"]     = new_status
            task["updated_at"] = datetime.now().isoformat(timespec="seconds")
            self._save_state(state)
            return (
                f"Görev {task_id} güncellendi.\n"
                f"  Başlık : {task.get('title', '-')}\n"
                f"  Durum  : {new_status}"
            )

        return (
            f"Görev {task_id} bulundu ama yeni durum belirtilmedi.\n"
            "Geçerli durumlar: todo | in_progress | blocked | done"
        )

    def _handle_set_goal(self, query: str, state: dict) -> str:
        """Update the sprint goal text."""
        goal_match = re.search(
            r"(?:sprint\s+hedef|hedef\s+belirle|goal\s+set|set\s+goal)[:\s]+(.+)",
            query, re.I,
        )
        if not goal_match:
            return (
                "Sprint hedefi belirtilmedi.\n"
                "Örnek: 'sprint hedefi: Kimlik doğrulama modülünü tamamla'"
            )
        goal = goal_match.group(1).strip()
        state["sprint"]["goal"] = goal
        self._save_state(state)
        return (
            f"Sprint hedefi güncellendi.\n"
            f"  Sprint : {state['sprint']['id']}\n"
            f"  Hedef  : {goal}"
        )

    def _handle_general(self, state: dict) -> str:
        """Fallback: return sprint summary and usage hints."""
        sprint  = state.get("sprint", {})
        tasks   = state.get("tasks", [])
        total   = len(tasks)
        done    = sum(1 for t in tasks if t.get("status") == "done")
        blocked = sum(1 for t in tasks if t.get("status") == "blocked")

        return "\n".join([
            f"Scrum Master Agent — Sprint: {sprint.get('id', '?')}",
            f"Hedef: {sprint.get('goal', '-')}",
            f"Toplam: {total} görev | Tamamlanan: {done} | Engellenmiş: {blocked}",
            "",
            "Desteklenen komutlar:",
            "  • Sprint durumu nedir?",
            "  • Engellenmiş görevler neler?",
            "  • T-001 developer-default üzerine ata",
            "  • Günlük standup",
            "  • Retrospektif",
            "  • Sprint review",
            "  • Görev ekle: <başlık>",
            "  • T-001 durumunu done olarak güncelle",
            "  • Sprint hedefi: <hedef metni>",
            "  • Görevi kime delegasyonu yapmalıyım?",
        ])

    # ─────────────────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _task_list_short(self, tasks: list[dict]) -> str:
        if not tasks:
            return "(boş backlog)"
        return ", ".join(t["id"] for t in tasks[:12])
