import sys
from pathlib import Path
from unittest.mock import MagicMock

# root path fix
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# mock external deps
sys.modules["ollama"] = MagicMock()
sys.modules["chromadb"] = MagicMock()

from unittest.mock import patch
import pytest

from src.services.api.pipeline import process_user_message

@pytest.fixture
def fake_chat_response():
    return {
        "message": {
            "content": "TEST_RESPONSE"
        }
    }


@pytest.fixture
def fake_db_manager():
    manager = MagicMock()
    manager.search.return_value = ["memory item 1", "memory item 2"]
    return manager


@patch("src.services.api.pipeline.chat")
@patch("src.pipeline.shared.db_manager")
def test_student_query_basic_behavior(mock_db_manager, mock_chat, fake_chat_response, fake_db_manager):
    mock_chat.return_value = fake_chat_response
    mock_db_manager.search = fake_db_manager.search

    result = process_user_message(
        user_text="What is artificial intelligence?",
        agent_role="StudentAgent",
        model="cognitwin-student",
        messages=[]
    )

    assert result is not None
    assert isinstance(result, dict)


@patch("src.services.api.pipeline.chat")
@patch("src.pipeline.shared.db_manager")
def test_developer_query_basic_behavior(mock_db_manager, mock_chat, fake_chat_response, fake_db_manager):
    mock_chat.return_value = fake_chat_response
    mock_db_manager.search = fake_db_manager.search

    result = process_user_message(
        user_text="Analyze pipeline architecture",
        agent_role="DeveloperAgent",
        model="cognitwin-developer",
        messages=[]
    )

    assert result is not None
    assert isinstance(result, dict)


@patch("src.services.api.pipeline.chat")
@patch("src.pipeline.shared.db_manager")
def test_empty_vector_case_does_not_crash(mock_db_manager, mock_chat, fake_chat_response):
    mock_chat.return_value = fake_chat_response
    mock_db_manager.search.return_value = []

    result = process_user_message(
        user_text="Explain ontology",
        agent_role="StudentAgent",
        model="cognitwin-student",
        messages=[]
    )

    assert result is not None
    assert isinstance(result, dict)


@patch("src.services.api.pipeline.chat")
@patch("src.pipeline.shared.db_manager")
def test_pii_query_does_not_crash(mock_db_manager, mock_chat, fake_chat_response, fake_db_manager):
    mock_chat.return_value = fake_chat_response
    mock_db_manager.search = fake_db_manager.search

    result = process_user_message(
        user_text="My email is test@example.com and my phone is 5551234567",
        agent_role="StudentAgent",
        model="cognitwin-student",
        messages=[]
    )

    assert result is not None
    assert isinstance(result, dict)


@patch("src.services.api.pipeline.chat")
@patch("src.pipeline.shared.db_manager")
def test_routing_edge_case_with_developer_model(mock_db_manager, mock_chat, fake_chat_response, fake_db_manager):
    mock_chat.return_value = fake_chat_response
    mock_db_manager.search = fake_db_manager.search

    result = process_user_message(
        user_text="Help me debug this repository",
        agent_role="StudentAgent",
        model="cognitwin-developer",
        messages=[]
    )

    assert result is not None
    assert isinstance(result, dict)


@patch("src.services.api.pipeline.chat")
@patch("src.pipeline.shared.db_manager")
def test_blank_like_query_does_not_crash(mock_db_manager, mock_chat, fake_chat_response):
    mock_chat.return_value = fake_chat_response
    mock_db_manager.search.return_value = []

    result = process_user_message(
        user_text="",
        agent_role="StudentAgent",
        model="cognitwin-student",
        messages=[]
    )

    assert result is not None
    assert isinstance(result, dict)