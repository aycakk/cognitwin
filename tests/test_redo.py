"""Unit tests for src/pipeline/redo.py.

Stubs replace all injected callables — no mocking framework needed.
"""

import datetime
import pytest

from src.pipeline.redo import _open_redo, _close_redo, run_redo_loop


# ─────────────────────────────────────────────────────────────────────────────
#  _open_redo
# ─────────────────────────────────────────────────────────────────────────────

def test_open_redo_appends_record():
    log = []
    redo_id = _open_redo(log, "C4", "tahminim")
    assert len(log) == 1
    rec = log[0]
    assert rec["redo_id"] == redo_id
    assert rec["trigger_gate"] == "C4"
    assert rec["failed_evidence"] == "tahminim"
    assert rec["revision_action"] is None
    assert rec["closure_gates"] == {}
    assert rec["closed_at"] is None


def test_open_redo_returns_8char_id():
    log = []
    redo_id = _open_redo(log, "C1", "")
    assert isinstance(redo_id, str)
    assert len(redo_id) == 8


def test_open_redo_two_calls_produce_separate_records():
    log = []
    id1 = _open_redo(log, "C4", "ev1")
    id2 = _open_redo(log, "C6", "ev2")
    assert len(log) == 2
    assert id1 != id2
    assert log[0]["trigger_gate"] == "C4"
    assert log[1]["trigger_gate"] == "C6"


# ─────────────────────────────────────────────────────────────────────────────
#  _close_redo
# ─────────────────────────────────────────────────────────────────────────────

def _make_gate_results(passed: bool) -> dict:
    return {"C4": {"pass": passed, "evidence": "x"}}


def test_close_redo_fills_record():
    log = []
    redo_id = _open_redo(log, "C4", "evidence")
    gate_results = {"C4": {"pass": True, "evidence": ""}}
    _close_redo(log, redo_id, "Draft passed all gates after revision.", gate_results)

    rec = log[0]
    assert rec["revision_action"] == "Draft passed all gates after revision."
    assert rec["closure_gates"] == {"C4": "PASS"}
    assert rec["closed_at"] is not None
    # closed_at should be a parseable ISO datetime
    datetime.datetime.fromisoformat(rec["closed_at"])


def test_close_redo_maps_pass_and_fail():
    log = []
    redo_id = _open_redo(log, "C4", "")
    gate_results = {
        "C4": {"pass": True,  "evidence": ""},
        "C6": {"pass": False, "evidence": "bad"},
    }
    _close_redo(log, redo_id, "action", gate_results)
    assert log[0]["closure_gates"] == {"C4": "PASS", "C6": "FAIL"}


def test_close_redo_leaves_other_records_untouched():
    log = []
    id1 = _open_redo(log, "C4", "ev1")
    id2 = _open_redo(log, "C6", "ev2")
    _close_redo(log, id2, "action", {"C6": {"pass": True, "evidence": ""}})

    assert log[0]["closed_at"] is None      # id1 untouched
    assert log[1]["closed_at"] is not None  # id2 closed


def test_close_redo_unknown_id_is_noop():
    log = []
    _open_redo(log, "C4", "ev")
    _close_redo(log, "deadbeef", "action", {})   # should not raise
    assert log[0]["closed_at"] is None


# ─────────────────────────────────────────────────────────────────────────────
#  run_redo_loop — stubs
# ─────────────────────────────────────────────────────────────────────────────

class _FakeMessage:
    def __init__(self, content): self.content = content

class _FakeResp:
    def __init__(self, content): self.message = _FakeMessage(content)


def _gate_pass(draft, vector_context, is_empty, agent_role, redo_log):
    return {"conjunction": True, "gates": {"C4": {"pass": True, "evidence": ""}}}

def _gate_fail(draft, vector_context, is_empty, agent_role, redo_log):
    return {"conjunction": False, "gates": {"C4": {"pass": False, "evidence": "bad evidence"}}}

def _gate_fail_then_pass():
    calls = []
    def _fn(draft, vector_context, is_empty, agent_role, redo_log):
        calls.append(draft)
        if len(calls) == 1:
            return {"conjunction": False, "gates": {"C4": {"pass": False, "evidence": "bad"}}}
        return {"conjunction": True,  "gates": {"C4": {"pass": True,  "evidence": ""}}}
    return _fn

def _chat_fn(model, messages):
    return _FakeResp("revised draft")

def _blindspot_fn(query, reason=""):
    return f"[BLINDSPOT:{reason}]"

_BASE_KWARGS = dict(
    base_messages=[],
    vector_context="ctx",
    is_empty=False,
    redo_log=None,          # replaced per test
    agent_role="StudentAgent",
    query="test query",
    redo_rules="stay grounded.",
    limit_message_template="LIMIT HIT Gate {gate}.",
    post_process=lambda s: s,
    gate_fn=_gate_pass,
    chat_fn=_chat_fn,
    blindspot_fn=_blindspot_fn,
)


def _call(draft="initial draft", **overrides):
    kwargs = {**_BASE_KWARGS, **overrides}
    if kwargs["redo_log"] is None:
        kwargs["redo_log"] = []
    return run_redo_loop(draft, **kwargs)


# ── passes first attempt ──────────────────────────────────────────────────────

def test_passes_first_attempt_returns_draft():
    result, limit_hit = _call(gate_fn=_gate_pass)
    assert limit_hit is False
    assert result == "initial draft"


def test_passes_first_attempt_opens_no_redo_record():
    log = []
    _call(gate_fn=_gate_pass, redo_log=log)
    assert log == []


# ── fails once then passes ────────────────────────────────────────────────────

def test_fail_once_then_pass_returns_revised_draft():
    result, limit_hit = _call(gate_fn=_gate_fail_then_pass())
    assert limit_hit is False
    assert result == "revised draft"


def test_fail_once_then_pass_closes_redo_record():
    log = []
    _call(gate_fn=_gate_fail_then_pass(), redo_log=log)
    assert len(log) == 1
    assert log[0]["closed_at"] is not None
    assert log[0]["revision_action"] == "Draft passed all gates after revision."


def test_fail_once_redo_instruction_contains_gate_and_evidence():
    """Verify chat_fn receives the REDO instruction mentioning gate and evidence."""
    captured = []
    def chat_fn(model, messages):
        captured.append(messages[-1]["content"])
        return _FakeResp("revised draft")

    _call(gate_fn=_gate_fail_then_pass(), chat_fn=chat_fn)
    instruction = captured[0]
    assert "C4" in instruction
    assert "bad" in instruction


def test_fail_once_post_process_applied():
    result, _ = _call(
        gate_fn=_gate_fail_then_pass(),
        post_process=lambda s: s.upper(),
    )
    assert result == "REVISED DRAFT"


# ── REDO limit reached ────────────────────────────────────────────────────────

def test_redo_limit_returns_limit_hit_true():
    _, limit_hit = _call(gate_fn=_gate_fail)
    assert limit_hit is True


def test_redo_limit_result_contains_blindspot_and_limit_message():
    result, _ = _call(gate_fn=_gate_fail)
    assert "[BLINDSPOT:" in result
    assert "LIMIT HIT Gate C4." in result


def test_redo_limit_blindspot_called_with_query_and_reason():
    captured = {}
    def blindspot_fn(query, reason=""):
        captured["query"] = query
        captured["reason"] = reason
        return "[BS]"

    _call(gate_fn=_gate_fail, blindspot_fn=blindspot_fn, query="my query")
    assert captured["query"] == "my query"
    assert "C4" in captured["reason"]
    assert "REDO LIMIT" in captured["reason"]
