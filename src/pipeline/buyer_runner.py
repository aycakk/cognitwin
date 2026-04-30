"""pipeline/buyer_runner.py — deterministic Fashion Buyer Agent MVP skeleton.

This runner is intentionally template-based:
  - No LLM calls
  - No external API calls
  - Buyer-only token economy (fixed-cost policy)
"""

from __future__ import annotations

import re

from src.pipeline.buyer.token_economy import BuyerTokenEconomy


_TOKEN_ECONOMY = BuyerTokenEconomy()


def _detect_season(text: str) -> str:
    lowered = text.lower()
    season_map = {
        "ilkbahar": "İlkbahar",
        "yaz": "Yaz",
        "sonbahar": "Sonbahar",
        "kış": "Kış",
        "kis": "Kış",
        "spring": "Spring",
        "summer": "Summer",
        "autumn": "Autumn",
        "fall": "Fall",
        "winter": "Winter",
    }
    for token, label in season_map.items():
        if token in lowered:
            return label
    return "Belirtilmedi"


def _detect_focus_category(text: str) -> str:
    lowered = text.lower()
    category_map = [
        ("outerwear", "Dış Giyim"),
        ("mont", "Dış Giyim"),
        ("ceket", "Dış Giyim"),
        ("dress", "Elbise"),
        ("elbise", "Elbise"),
        ("denim", "Denim"),
        ("jean", "Denim"),
        ("ayakkabı", "Ayakkabı"),
        ("ayakkabi", "Ayakkabı"),
        ("aksesuar", "Aksesuar"),
        ("triko", "Triko"),
    ]
    for token, label in category_map:
        if token in lowered:
            return label
    return "Genel Koleksiyon"


def _detect_budget_signal(text: str) -> str:
    has_currency = bool(re.search(r"\b(\d[\d.,]*)\s*(tl|try|usd|eur|\$|€)\b", text, re.I))
    if has_currency:
        return "Bütçe sinyali alındı (detaylı limit analizi sonraki sürümde)."
    return "Net bütçe bilgisi verilmedi."


def run_buyer_pipeline(user_message: str, session_id: str | None = None) -> str:
    """Return deterministic Fashion Buyer Agent MVP response."""
    token_decision = _TOKEN_ECONOMY.evaluate_request(session_id=session_id)
    if not token_decision["allowed"]:
        remaining = token_decision["remaining"]
        required = token_decision["used_this_request"]
        return (
            "Buyer token budget yetersiz. "
            f"Bu analiz için {required} token gerekli, kalan token: {remaining}."
        )

    source_text = user_message or ""
    season = _detect_season(source_text)
    category = _detect_focus_category(source_text)
    budget_signal = _detect_budget_signal(source_text)
    sid = session_id.strip() if session_id else "n/a"
    budget_total = token_decision["budget_total"]
    used_this_request = token_decision["used_this_request"]
    total_used = token_decision["total_used"]
    remaining = token_decision["remaining"]

    return (
        "Fashion Buyer Agent Analysis\n"
        f"- Session: {sid}\n"
        f"- Input Focus: {category}\n"
        f"- Season: {season}\n\n"
        "Trend / Demand Signal\n"
        "- Mevcut girişe göre temel talep sinyali pozitif-ılımlı görünüyor.\n"
        "- Hızlı yenilenen ürün gruplarında kısa çevrimli replenishment önerilir.\n\n"
        "Purchase Recommendation\n"
        "- Çekirdek ürünlerde dengeli alım, trend parçalarda kontrollü test alımı yapın.\n"
        f"- {budget_signal}\n\n"
        "Suggested Quantity Range\n"
        "- Başlangıç siparişi: 120 - 260 adet (kategori ve mağaza kırılımına göre ayarlanır).\n\n"
        "Assortment Mix\n"
        "- Renk dağılımı: %50 nötr / %30 sezon rengi / %20 vurgu rengi\n"
        "- Beden dağılımı: XS %10 / S %25 / M %30 / L %22 / XL %13\n"
        "- Kategori dağılımı: Core %60 / Trend %25 / Tamamlayıcı %15\n\n"
        "Vendor RFQ Questions\n"
        "1. MOQ, termin süresi ve kapasite teyidi nedir?\n"
        "2. Kumaş/aksesuar alternatifleri ve maliyet etkileri nelerdir?\n"
        "3. Kalite kontrol, iade ve gecikme durumunda SLA şartları nelerdir?\n"
        "4. Numune ve toplu üretim arasında tolerans farkı nasıl yönetilecek?\n\n"
        "Risk Level\n"
        "- Seviye: Medium\n"
        "- Gerekçe: Talep dalgalanması ve tedarik termin belirsizliği.\n\n"
        "Token Usage\n"
        f"- Budget: {budget_total}\n"
        f"- Used this request: {used_this_request}\n"
        f"- Total used: {total_used}\n"
        f"- Remaining: {remaining}\n"
    )

