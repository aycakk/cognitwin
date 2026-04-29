from __future__ import annotations

import src.intent_classifier as ic


def test_predict_intent_uses_rule_based_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(ic, "model", None)
    monkeypatch.setattr(ic, "vectorizer", None)

    assert ic.predict_intent("Matematik vize sınavı ne zaman?") == "exam_info"
    assert ic.predict_intent("Bu dönem hangi derslerim var?") == "course_info"
    assert ic.predict_intent("COM8090 dersi hocası kim?") == "instructor_info"
    assert ic.predict_intent("Yazılım test prensiplerini açıkla") == "general"


def test_get_model_paths_exposes_load_status():
    info = ic.get_model_paths()
    assert "model_loaded" in info
    assert "load_error" in info

