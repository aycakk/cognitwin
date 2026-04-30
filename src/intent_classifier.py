"""Intent classifier with safe ML loading and rule-based fallback.

This module must never crash API import-time.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

try:
    import joblib  # type: ignore
except Exception as exc:  # pragma: no cover - depends on runtime env
    joblib = None  # type: ignore[assignment]
    logger.warning("joblib import failed; intent classifier will use fallback. error=%s", exc)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "student_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

model: Any | None = None
vectorizer: Any | None = None
model_load_error: str | None = None

if joblib is not None:
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    except Exception as exc:  # pragma: no cover - runtime/data dependent
        model = None
        vectorizer = None
        model_load_error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "Intent model load failed, switching to rule-based fallback. "
            "model_path=%s vectorizer_path=%s error=%s",
            model_path,
            vectorizer_path,
            model_load_error,
        )
else:
    model_load_error = "joblib import unavailable"


def _predict_intent_rule_based(text: str) -> str:
    normalized = (text or "").lower()

    exam_patterns = (
        r"\bsınav\b",
        r"\bvize\b",
        r"\bfinal\b",
        r"\bexam\b",
    )
    if any(re.search(pat, normalized, re.I) for pat in exam_patterns):
        return "exam_info"

    course_patterns = (
        r"\bders\b",
        r"\bcourse\b",
    )
    if any(re.search(pat, normalized, re.I) for pat in course_patterns):
        return "course_info"

    instructor_patterns = (
        r"\bhoca\b",
        r"\binstructor\b",
        r"\bkim veriyor\b",
    )
    if any(re.search(pat, normalized, re.I) for pat in instructor_patterns):
        return "instructor_info"

    return "general"


def predict_intent(text: str) -> str:
    """Return intent label without raising runtime exceptions."""
    if model is None or vectorizer is None:
        return _predict_intent_rule_based(text)
    try:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        return str(pred)
    except Exception as exc:
        logger.warning("Intent model inference failed; fallback used. error=%s", exc)
        return _predict_intent_rule_based(text)


def get_model_paths() -> dict[str, str | bool | None]:
    return {
        "model_path": model_path,
        "vectorizer_path": vectorizer_path,
        "model_exists": os.path.exists(model_path),
        "vectorizer_exists": os.path.exists(vectorizer_path),
        "model_loaded": model is not None and vectorizer is not None,
        "load_error": model_load_error,
    }

