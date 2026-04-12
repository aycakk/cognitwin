"""
test_agent_contracts.py — Typed contract tests for pipeline runners.

Verifies that run_pipeline (student path) returns AgentResponse with the
correct type, non-empty draft, and accurate TaskStatus on both the success
and REDO-limit-hit paths.

External I/O (Ollama, ChromaDB, ontology) is stubbed so these tests run
without any running infrastructure.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Stub heavy external dependencies before any project imports resolve them.
sys.modules.setdefault("ollama",    MagicMock())
sys.modules.setdefault("chromadb",  MagicMock())

import pytest  # noqa: E402

from src.core.schemas import AgentTask, AgentResponse, AgentRole, TaskStatus  # noqa: E402
from src.pipeline.student_runner import run_pipeline                           # noqa: E402

# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------

# Ollama's chat() returns an object with attribute access: resp.message.content
# Build a lightweight stand-in using MagicMock so attribute traversal works.
_FAKE_CHAT_RETURN = MagicMock()
_FAKE_CHAT_RETURN.message.content = "Bu bir test yanıtıdır."

# (context_str, is_empty=False) — non-empty memory so the pipeline reaches
# draft synthesis instead of the early-exit blindspot path.
_FAKE_RETRIEVE = ("Bağlam bilgisi mevcut.", False)

_REDO_SUCCESS_DRAFT = "Bu bir test yanıtıdır."
_REDO_LIMIT_DRAFT   = "Doğrulama başarısız — son taslak."


def _make_task(text: str = "Test sorgusu.") -> AgentTask:
    return AgentTask(role=AgentRole.STUDENT, masked_input=text)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------

@patch("src.pipeline.student_runner.run_redo_loop", return_value=(_REDO_SUCCESS_DRAFT, False))
@patch("src.pipeline.student_runner.build_ontology_context", return_value="")
@patch("src.pipeline.student_runner.VECTOR_MEM")
@patch("src.pipeline.student_runner.chat", return_value=_FAKE_CHAT_RETURN)
def test_run_pipeline_returns_agent_response(mock_chat, mock_mem, mock_onto, mock_redo):
    mock_mem.retrieve.return_value = _FAKE_RETRIEVE
    result = run_pipeline(_make_task())
    assert isinstance(result, AgentResponse), (
        f"run_pipeline must return AgentResponse, got {type(result)}"
    )


# ---------------------------------------------------------------------------
# 2. draft is non-empty on success
# ---------------------------------------------------------------------------

@patch("src.pipeline.student_runner.run_redo_loop", return_value=(_REDO_SUCCESS_DRAFT, False))
@patch("src.pipeline.student_runner.build_ontology_context", return_value="")
@patch("src.pipeline.student_runner.VECTOR_MEM")
@patch("src.pipeline.student_runner.chat", return_value=_FAKE_CHAT_RETURN)
def test_run_pipeline_draft_not_empty_on_success(mock_chat, mock_mem, mock_onto, mock_redo):
    mock_mem.retrieve.return_value = _FAKE_RETRIEVE
    result = run_pipeline(_make_task("Ontoloji nedir?"))
    assert result.draft, "draft must not be empty on a successful run"


# ---------------------------------------------------------------------------
# 3. status == COMPLETED on success
# ---------------------------------------------------------------------------

@patch("src.pipeline.student_runner.run_redo_loop", return_value=(_REDO_SUCCESS_DRAFT, False))
@patch("src.pipeline.student_runner.build_ontology_context", return_value="")
@patch("src.pipeline.student_runner.VECTOR_MEM")
@patch("src.pipeline.student_runner.chat", return_value=_FAKE_CHAT_RETURN)
def test_run_pipeline_status_completed_on_success(mock_chat, mock_mem, mock_onto, mock_redo):
    mock_mem.retrieve.return_value = _FAKE_RETRIEVE
    result = run_pipeline(_make_task("Sprint nedir?"))
    assert result.status == TaskStatus.COMPLETED, (
        f"Expected COMPLETED, got {result.status}"
    )


# ---------------------------------------------------------------------------
# 4. status == FAILED when REDO loop exhausts its limit
# ---------------------------------------------------------------------------

@patch("src.pipeline.student_runner.run_redo_loop",
       return_value=(_REDO_LIMIT_DRAFT, True))   # limit_hit=True
@patch("src.pipeline.student_runner.build_ontology_context", return_value="")
@patch("src.pipeline.student_runner.VECTOR_MEM")
@patch("src.pipeline.student_runner.chat", return_value=_FAKE_CHAT_RETURN)
def test_run_pipeline_status_failed_on_redo_limit(mock_chat, mock_mem, mock_onto, mock_redo):
    mock_mem.retrieve.return_value = _FAKE_RETRIEVE
    result = run_pipeline(_make_task("Başarısız senaryo."))
    assert result.status == TaskStatus.FAILED, (
        f"Expected FAILED when REDO limit hit, got {result.status}"
    )
    assert result.draft == _REDO_LIMIT_DRAFT, (
        "draft must carry the last REDO output even on FAILED status"
    )


# ---------------------------------------------------------------------------
# 5. task_id is echoed in response
# ---------------------------------------------------------------------------

@patch("src.pipeline.student_runner.run_redo_loop", return_value=(_REDO_SUCCESS_DRAFT, False))
@patch("src.pipeline.student_runner.build_ontology_context", return_value="")
@patch("src.pipeline.student_runner.VECTOR_MEM")
@patch("src.pipeline.student_runner.chat", return_value=_FAKE_CHAT_RETURN)
def test_run_pipeline_echoes_task_id(mock_chat, mock_mem, mock_onto, mock_redo):
    mock_mem.retrieve.return_value = _FAKE_RETRIEVE
    task = _make_task()
    result = run_pipeline(task)
    assert result.task_id == task.task_id, (
        "AgentResponse.task_id must echo the originating AgentTask.task_id"
    )
