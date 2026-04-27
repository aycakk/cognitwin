"""core/llm_config.py — lightweight LLM configuration constants.

Kept dependency-free (stdlib only) so any module — including test fixtures
and lazy-import code paths that must not pull in the ollama client — can
import these constants without dragging the full pipeline graph along.

Override at deploy time via the OLLAMA_MODEL env var, e.g.:
    OLLAMA_MODEL=llama3.2:1b   # smaller, faster
    OLLAMA_MODEL=llama3.2      # default
"""

from __future__ import annotations

import os

DEFAULT_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama3.2")
