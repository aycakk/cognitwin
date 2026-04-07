"""shared/patterns.py — regex registry shared by gate evaluators.

Single source of truth. Previously duplicated as:
  - ASP_NEG_PATTERNS in src/services/api/pipeline.py
  - _ASP_NEG_PATTERNS in src/agents/student_agent.py
  - _PII_PATTERNS    in src/agents/student_agent.py
    (pipeline.py used inline regex literals inside gate_c1_pii_masking;
     those remain inline and are NOT migrated in Step 1 to avoid any
     change in C1 behavior on the developer/student paths.)

Behavior preserved exactly: identical regexes, identical flags.
The two prior ASP_NEG copies differed only in the alternation order
inside ASP-NEG-02 ("tahminim|sanırım|..." vs "sanırım|...|tahminim").
Both matches are equivalent because the alternatives are disjoint;
the canonical version below uses the pipeline.py ordering.
"""

from __future__ import annotations

import re

# ─────────────────────────────────────────────────────────────────────
#  PII leak patterns  (used by StudentAgent C1)
# ─────────────────────────────────────────────────────────────────────
PII_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b\d{9,12}\b"),                                  # student / TC ID
    re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}"),  # e-mail
    re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),             # phone
]

# ─────────────────────────────────────────────────────────────────────
#  Anti-sycophancy patterns  (used by C4 hallucination + C6 sweep)
# ─────────────────────────────────────────────────────────────────────
ASP_NEG_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ASP-NEG-01_PII_UNMASK",    re.compile(r"\b\d{8,11}\b")),
    ("ASP-NEG-02_HALLUCINATION", re.compile(r"tahminim|sanırım|galiba|muhtemelen", re.I)),
    ("ASP-NEG-03_FALSE_PREMISE", re.compile(r"haklısınız|evet,?\s+öyle\s+söylemiştim", re.I)),
    ("ASP-NEG-04_SOFTENED_FAIL", re.compile(r"yine de cevaplamaya çalışayım|bence şöyle olabilir", re.I)),
    ("ASP-NEG-05_WEIGHT_ONLY",   re.compile(r"genel\s+bilgime\s+göre|eğitim\s+verilerime\s+göre", re.I)),
]
