from __future__ import annotations

import json
import os
import re
from typing import Any


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROFILE_DIR = os.path.join(BASE_DIR, "data", "user_profiles")


def normalize_user_id(user_id: str | None) -> str:
    raw = (user_id or "").strip().lower()
    if not raw:
        return "anonymous"
    safe = re.sub(r"[^a-z0-9_.-]+", "_", raw)
    return safe[:80] or "anonymous"


def _profile_path(user_id: str) -> str:
    os.makedirs(PROFILE_DIR, exist_ok=True)
    return os.path.join(PROFILE_DIR, f"{normalize_user_id(user_id)}.json")


def load_profile(user_id: str) -> dict[str, Any]:
    path = _profile_path(user_id)
    if not os.path.exists(path):
        return {
            "user_id": normalize_user_id(user_id),
            "display_name": "",
            "message_count": 0,
            "total_chars": 0,
            "formal_signals": 0,
            "casual_signals": 0,
            "language": "tr",
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "user_id": normalize_user_id(user_id),
            "display_name": "",
            "message_count": 0,
            "total_chars": 0,
            "formal_signals": 0,
            "casual_signals": 0,
            "language": "tr",
        }


def save_profile(user_id: str, profile: dict[str, Any]) -> None:
    path = _profile_path(user_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


def update_user_profile(user_id: str, text: str, display_name: str | None = None) -> dict[str, Any]:
    profile = load_profile(user_id)
    t = (text or "").strip()
    if display_name and display_name.strip():
        profile["display_name"] = display_name.strip()

    profile["message_count"] = int(profile.get("message_count", 0)) + 1
    profile["total_chars"] = int(profile.get("total_chars", 0)) + len(t)

    low = t.lower()
    formal_markers = ["merhaba", "rica", "teşekkür", "tesekkur", "sayın", "sn."]
    casual_markers = ["kanka", "abi", "ya", "nolur", "napcam", "acil"]

    if any(m in low for m in formal_markers):
        profile["formal_signals"] = int(profile.get("formal_signals", 0)) + 1
    if any(m in low for m in casual_markers):
        profile["casual_signals"] = int(profile.get("casual_signals", 0)) + 1

    if re.search(r"[çğıöşüÇĞİÖŞÜ]", t):
        profile["language"] = "tr"

    save_profile(user_id, profile)
    return profile


def get_user_style_hint(user_id: str) -> str:
    p = load_profile(user_id)
    count = max(int(p.get("message_count", 0)), 1)
    avg_len = int(p.get("total_chars", 0)) // count
    formal = int(p.get("formal_signals", 0))
    casual = int(p.get("casual_signals", 0))

    tone = "formal" if formal >= casual else "casual"
    length_pref = "short" if avg_len < 120 else "detailed"

    if tone == "formal" and length_pref == "short":
        return "Use polite, concise Turkish."
    if tone == "formal" and length_pref == "detailed":
        return "Use polite, structured Turkish with brief explanations."
    if tone == "casual" and length_pref == "short":
        return "Use clear, direct Turkish with short sentences."
    return "Use natural Turkish and include compact explanatory details."


def get_user_display_name(user_id: str) -> str:
    p = load_profile(user_id)
    display_name = (p.get("display_name") or "").strip()
    if display_name:
        return display_name
    scoped = normalize_user_id(user_id)
    if re.fullmatch(r"[a-f0-9]{16,64}", scoped):
        return "Ogrenci"
    parts = [x for x in re.split(r"[_\-.]+", scoped) if x]
    if not parts:
        return "Ogrenci"
    return " ".join(x.capitalize() for x in parts)
