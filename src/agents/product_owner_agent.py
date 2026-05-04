"""agents/product_owner_agent.py — Rule-based Product Owner agent.

Backlog owner for the CogniTwin Scrum team. Converts external requests
into user stories with acceptance criteria and priority, and performs
acceptance review on completed work.

Architectural boundaries:
  - Writes ONLY to the ``backlog`` array in sprint_state.json.
  - Never touches the ``tasks`` array (ScrumMaster is the sprint write owner).
  - Does NOT call any LLM — fully deterministic rule-based output.

Design follows the same pattern as ScrumMasterAgent:
  - Intent detection via Turkish + English regex patterns
  - Handler dispatch through a dict of lambdas
  - All state mutations run under SprintStateStore.state_lock()
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.agents.capability_manifest import CapabilityManifest
from src.pipeline.scrum_team.sprint_state_store import SprintStateStore

logger = logging.getLogger(__name__)


_PO_MANIFEST = CapabilityManifest(
    role="ProductOwnerAgent",
    intents=(
        "create_story",
        "list_backlog",
        "prioritize",
        "define_criteria",
        "accept_story",
        "reject_story",
        "backlog_status",
        "review_completed",
    ),
    inputs=(
        "product_goal",
        "stakeholder_feedback",
        "completed_story",
        "acceptance_criteria",
    ),
    outputs=(
        "product_backlog_item",
        "ordered_backlog",
        "po_review_decision",
    ),
    gates_consumed=("C1", "C4", "C5", "C7", "A1"),
    ontology_classes_referenced=(
        "ProductOwner",
        "ProductBacklog",
        "BacklogItem",
        "SprintReview",
        "ProductGoal",
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
#  Intent detection — Turkish + English keywords
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("create_story", re.compile(
        r"hikaye\s+oluştur|story\s+ekle|create\s+story|yeni\s+hikaye"
        r"|hikaye\s+yarat|add\s+story",
        re.I)),
    ("list_backlog", re.compile(
        r"backlog\s+listele|backlog\s+göster|show\s+backlog|list\s+backlog"
        r"|hikayeleri?\s+göster|hikayeleri?\s+listele",
        re.I)),
    ("prioritize", re.compile(
        r"öncelik|priority|prioritize|önem\s+sırala|öncelik\s+belirle",
        re.I)),
    ("define_criteria", re.compile(
        r"kabul\s+kriter|acceptance\s+criter|kriter\s+belirle|define\s+criteria"
        r"|kabul\s+koşul",
        re.I)),
    ("accept_story", re.compile(
        r"kabul\s+et|accept\b|onayla|approve",
        re.I)),
    ("reject_story", re.compile(
        r"reddet|reject\b|geri\s+çevir",
        re.I)),
    # review_completed: PO wants to see tasks the Developer finished
    ("review_completed", re.compile(
        r"tamamlanan.*incele|incele.*tamamlanan|inceleme\s+bekleyen"
        r"|ready.{0,10}review|review.{0,10}complet"
        r"|geliştirici.*bitirdi|developer.*(?:tamamladı|bitti|done)"
        r"|kabul\s+bekleyen\s+görev|hangi\s+görev.*tamamlandı",
        re.I)),
    # backlog_status stays last — open-ended reasoning fallthrough
    ("backlog_status", re.compile(
        r"backlog\s+durum|backlog\s+status|hikaye\s+durum|backlog\s+özet"
        r"|backlog\s+summary",
        re.I)),
]


class ProductOwnerAgent:
    """
    Rule-based Product Owner agent.

    Owns the backlog — creates user stories with acceptance criteria,
    assigns priority, and performs acceptance/rejection review.
    State I/O is delegated to SprintStateStore.

    Sprint flow integration
    ───────────────────────
    PO creates stories in backlog → SM promotes stories into sprint tasks →
    Dev executes → SM tracks → PO accepts/rejects completed work.
    """

    def __init__(self, state_store: SprintStateStore | None = None) -> None:
        self._store = state_store or SprintStateStore()

    @classmethod
    def capability_manifest(cls) -> CapabilityManifest:
        return _PO_MANIFEST

    # ─────────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────────

    def handle_query(self, query: str) -> str:
        """
        Route the query to the appropriate rule handler and return a
        structured text response.  No LLM is called here.
        """
        intent = self._detect_intent(query)

        with self._store.state_lock():
            state = self._store.load()
            handlers: dict[str, Any] = {
                "create_story":    lambda: self._handle_create_story(query, state),
                "list_backlog":    lambda: self._handle_list_backlog(state),
                "prioritize":      lambda: self._handle_prioritize(query, state),
                "define_criteria": lambda: self._handle_define_criteria(query, state),
                "accept_story":    lambda: self._handle_accept(query, state),
                "reject_story":    lambda: self._handle_reject(query, state),
                "backlog_status":  lambda: self._handle_backlog_status(state),
                "review_completed": lambda: self._handle_review_completed(state),
            }
            result = handlers.get(intent, lambda: self._handle_unknown(state))()

        logger.debug("po: intent=%r query=%r", intent, query[:80])
        return result

    def detect_intent(self, query: str) -> str:
        """Public entry point for intent detection."""
        return self._detect_intent(query)

    # ─────────────────────────────────────────────────────────────────────────
    #  Intent detection
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_intent(self, query: str) -> str:
        for intent_name, pattern in _INTENT_PATTERNS:
            if pattern.search(query):
                return intent_name
        return "unknown"

    # ─────────────────────────────────────────────────────────────────────────
    #  Handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_create_story(self, query: str, state: dict) -> str:
        """Create a new user story in the backlog.

        Parses title from text after 'hikaye oluştur:' / 'create story:' /
        'story ekle:' or uses the entire query as title.
        """
        title_match = re.search(
            r"(?:hikaye\s+oluştur|create\s+story|story\s+ekle"
            r"|yeni\s+hikaye|hikaye\s+yarat|add\s+story)[:\s]+(.+)",
            query, re.I,
        )
        title = title_match.group(1).strip() if title_match else query.strip()

        story_id = self._store.add_story(
            title=title,
            source_request=query,
        )

        return (
            f"Yeni hikaye oluşturuldu\n"
            f"  ID       : {story_id}\n"
            f"  Başlık   : {title[:80]}\n"
            f"  Durum    : draft\n"
            f"  Öncelik  : medium\n"
            f"  Not: Kabul kriterleri eklemek için -> "
            f"'{story_id} kabul kriterleri: ...'"
        )

    def _handle_list_backlog(self, state: dict) -> str:
        """List all active backlog stories."""
        backlog = state.get("backlog", [])
        active = [s for s in backlog if s.get("status") not in ("accepted", "rejected")]

        if not active:
            return "Backlog boş — aktif hikaye yok."

        lines = [f"Backlog ({len(active)} aktif hikaye):"]
        for s in active:
            criteria_count = len(s.get("acceptance_criteria", []))
            lines.append(
                f"  [{s['story_id']}] {s.get('title', '-')}"
                f"  | öncelik: {s.get('priority', '?')}"
                f"  | durum: {s.get('status', '?')}"
                f"  | kriter: {criteria_count}"
            )
        return "\n".join(lines)

    def _handle_prioritize(self, query: str, state: dict) -> str:
        """Set priority on a backlog story (S-NNN öncelik high/medium/low)."""
        story_match = re.search(r"\b(S-\d+)\b", query, re.I)
        if not story_match:
            # Check whether this looks like a new-project planning request that
            # slipped into command mode — give a helpful redirect instead of an error.
            if re.search(r"yeni\s*proje|proje\s*başlat|hikaye\w*\s+oluştur|backlog\s+oluştur", query, re.I):
                return (
                    "Yeni proje için hikayeler otomatik oluşturulabilir.\n"
                    "Örnek: 'Kullanıcı kaydı ve görev yönetimi için epic ve hikayeler oluştur'\n"
                    "Veya Ürün Sahibi modeline geçerek agile workflow'u başlatın."
                )
            return (
                "Önceliklendirme için hikaye ID'si gerekli (örn. S-001).\n"
                "Kullanım: S-001 öncelik high"
            )
        story_id = story_match.group(1).upper()

        priority_match = re.search(r"\b(high|medium|low|yüksek|orta|düşük)\b", query, re.I)
        if not priority_match:
            return f"Öncelik seviyesi belirtilmedi. Kullanım: {story_id} öncelik high/medium/low"

        priority_raw = priority_match.group(1).lower()
        priority_map = {"yüksek": "high", "orta": "medium", "düşük": "low"}
        priority = priority_map.get(priority_raw, priority_raw)

        if self._store.update_story(story_id, priority=priority):
            return f"{story_id} önceliği '{priority}' olarak güncellendi."
        return f"Hikaye bulunamadı: {story_id}"

    def _handle_define_criteria(self, query: str, state: dict) -> str:
        """Define acceptance criteria for a backlog story.

        Parses S-NNN and criteria text from query.
        Criteria can be comma-separated or newline-separated.
        """
        story_match = re.search(r"\b(S-\d+)\b", query, re.I)
        if not story_match:
            # Redirect new-project requests to the planning path instead of showing an error.
            if re.search(r"yeni\s*proje|proje\s*başlat|hikaye\w*\s+(?:çıkar|oluştur|üret|belirle)|backlog\s+oluştur", query, re.I):
                return (
                    "Yeni proje için kabul kriterleri otomatik oluşturulabilir.\n"
                    "Örnek: 'Kullanıcı kaydı ve görev yönetimi için epic ve hikayeler oluştur'\n"
                    "Veya Ürün Sahibi modeline geçerek agile workflow'u başlatın."
                )
            return (
                "Kabul kriterleri için hikaye ID'si gerekli (örn. S-001).\n"
                "Kullanım: S-001 kabul kriterleri: kriter1, kriter2"
            )
        story_id = story_match.group(1).upper()

        # Extract criteria text after the story ID + keyword
        criteria_match = re.search(
            r"(?:kabul\s+kriter(?:ler)?i?|acceptance\s+criteria?|kriter\s+belirle"
            r"|define\s+criteria|kabul\s+koşul(?:lar)?ı?)[:\s]+(.+)",
            query, re.I | re.DOTALL,
        )
        if not criteria_match:
            return f"Kriter metni bulunamadı. Kullanım: {story_id} kabul kriterleri: kriter1, kriter2"

        raw = criteria_match.group(1).strip()
        criteria = [c.strip() for c in re.split(r"[,\n;]", raw) if c.strip()]

        if not criteria:
            return "Boş kriter listesi. En az bir kriter belirtin."

        if self._store.update_story(story_id, acceptance_criteria=criteria):
            lines = [f"{story_id} kabul kriterleri güncellendi ({len(criteria)} kriter):"]
            for i, c in enumerate(criteria, 1):
                lines.append(f"  {i}. {c[:100]}")
            return "\n".join(lines)
        return f"Hikaye bulunamadı: {story_id}"

    def _handle_accept(self, query: str, state: dict) -> str:
        """Accept a completed story (PO acceptance review)."""
        story_match = re.search(r"\b(S-\d+)\b", query, re.I)
        if not story_match:
            return (
                "Kabul için hikaye ID'si gerekli (örn. S-001).\n"
                "Kullanım: S-001 kabul et"
            )
        story_id = story_match.group(1).upper()

        if self._store.accept_story(story_id):
            return f"{story_id} kabul edildi (accepted)."
        return f"Hikaye bulunamadı: {story_id}"

    def _handle_reject(self, query: str, state: dict) -> str:
        """Reject a story with an optional reason."""
        story_match = re.search(r"\b(S-\d+)\b", query, re.I)
        if not story_match:
            return (
                "Reddetme için hikaye ID'si gerekli (örn. S-001).\n"
                "Kullanım: S-001 reddet [sebep]"
            )
        story_id = story_match.group(1).upper()

        # Extract optional reason
        reason_match = re.search(
            r"(?:reddet|reject|geri\s+çevir)[:\s]+(.+)",
            query, re.I,
        )
        reason = reason_match.group(1).strip() if reason_match else ""

        if self._store.reject_story(story_id, reason):
            msg = f"{story_id} reddedildi (rejected)."
            if reason:
                msg += f"\n  Sebep: {reason[:100]}"
            return msg
        return f"Hikaye bulunamadı: {story_id}"

    def _handle_backlog_status(self, state: dict) -> str:
        """Show backlog summary with counts by status and priority."""
        backlog = state.get("backlog", [])

        if not backlog:
            return "Backlog boş — henüz hikaye oluşturulmamış."

        status_counts: dict[str, int] = {}
        priority_counts: dict[str, int] = {}
        for s in backlog:
            st = s.get("status", "unknown")
            pr = s.get("priority", "unknown")
            status_counts[st] = status_counts.get(st, 0) + 1
            priority_counts[pr] = priority_counts.get(pr, 0) + 1

        lines = [f"Backlog Durumu (toplam: {len(backlog)} hikaye)"]
        lines.append("  Duruma göre:")
        for st, cnt in sorted(status_counts.items()):
            lines.append(f"    {st}: {cnt}")
        lines.append("  Önceliğe göre:")
        for pr, cnt in sorted(priority_counts.items()):
            lines.append(f"    {pr}: {cnt}")

        # Show stories with criteria defined
        with_criteria = sum(
            1 for s in backlog if s.get("acceptance_criteria")
        )
        lines.append(f"  Kabul kriterleri tanımlı: {with_criteria}/{len(backlog)}")

        return "\n".join(lines)

    def _handle_review_completed(self, state: dict) -> str:
        """Show sprint tasks that the Developer completed and are awaiting PO review.

        Filters tasks[] for po_status == 'ready_for_review'.  These are tasks
        that originated from a backlog story (have source_story_id) and were
        marked done by the Developer via complete_task().
        """
        tasks = [
            t for t in state.get("tasks", [])
            if t.get("po_status") == "ready_for_review"
        ]
        if not tasks:
            return (
                "İnceleme bekleyen tamamlanmış görev yok.\n"
                "Geliştirici henüz hikayeye bağlı bir görevi tamamlamadı."
            )
        lines = [f"İnceleme Bekleyen Görevler ({len(tasks)}):"]
        for t in tasks:
            lines.append(
                f"\n  [{t['id']}] {t.get('title', '-')}"
                f"  | hikaye: {t.get('source_story_id', '-')}"
                f"  | öncelik: {t.get('priority', '?')}"
            )
            if t.get("result_summary"):
                lines.append(f"    Geliştirici özeti : {t['result_summary'][:120]}")
            if t.get("acceptance_criteria"):
                lines.append(
                    f"    Kabul kriterleri  : "
                    + " | ".join(t["acceptance_criteria"][:3])
                )
            story_id = t.get("source_story_id", "")
            if story_id:
                lines.append(
                    f"    Kabul için        : '{story_id} kabul et'\n"
                    f"    Red için          : '{story_id} reddet [sebep]'"
                )
        return "\n".join(lines)

    def _handle_unknown(self, state: dict) -> str:
        """Fallback for unrecognized queries — show usage help."""
        return (
            "Product Owner komutları:\n"
            "  • hikaye oluştur: <başlık>         — Yeni hikaye oluştur\n"
            "  • backlog listele                   — Aktif hikayeleri göster\n"
            "  • S-001 öncelik high                — Öncelik belirle (high/medium/low)\n"
            "  • S-001 kabul kriterleri: k1, k2    — Kabul kriterleri tanımla\n"
            "  • S-001 kabul et                    — Hikayeyi kabul et\n"
            "  • S-001 reddet [sebep]              — Hikayeyi reddet\n"
            "  • backlog durumu                    — Backlog özeti\n"
            "  • tamamlanan görevleri incele       — Geliştirici bitirdi, kabul bekliyor"
        )
