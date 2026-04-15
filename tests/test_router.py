"""Unit tests for src/pipeline/router.py — resolve_mode."""

import pytest
from src.pipeline.router import resolve_mode


@pytest.mark.parametrize("model, expected", [
    # product_owner branch
    ("cognitwin-product-owner",  ("product_owner", "rule")),
    ("COGNITWIN-PRODUCT_OWNER",  ("product_owner", "rule")),
    ("product_owner",            ("product_owner", "rule")),
    # developer branch — various casing and forms
    ("cognitwin-developer",  ("developer", "auto")),
    ("COGNITWIN-DEVELOPER",  ("developer", "auto")),
    ("developer",            ("developer", "auto")),
    # scrum branch
    ("cognitwin-scrum",      ("scrum_master", "rule")),
    # student branch — default model and edge inputs
    ("llama3.2",             ("student", "llm")),
    ("",                     ("student", "llm")),
    (None,                   ("student", "llm")),
])
def test_resolve_mode(model, expected):
    assert resolve_mode(model) == expected
