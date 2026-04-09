"""Unit tests for src/pipeline/router.py — resolve_mode."""

import pytest
from src.pipeline.router import resolve_mode


@pytest.mark.parametrize("model, expected", [
    # developer branch — various casing and forms
    ("cognitwin-developer",  ("developer", "auto")),
    ("COGNITWIN-DEVELOPER",  ("developer", "auto")),
    ("developer",            ("developer", "auto")),
    # student branch — default model and edge inputs
    ("llama3.2",             ("student", "llm")),
    ("",                     ("student", "llm")),
    (None,                   ("student", "llm")),
])
def test_resolve_mode(model, expected):
    assert resolve_mode(model) == expected
