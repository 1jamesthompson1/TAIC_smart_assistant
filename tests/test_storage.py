import datetime
import os
import uuid

import pandas as pd
import plotly.express as px
import pytest

from app import conversation_store, knowledge_search_store
from backend.Assistant import CompleteHistory
from backend.Searching import SearchParams


@pytest.mark.skipif(
    not os.getenv("TEST_USE_REAL_SERVICES"),
    reason="Requires real services",
)
def test_conversation_save_and_load(mock_assistant):
    """Test saving and loading conversations."""

    # Create a conversation
    conversation_id = str(uuid.uuid4())
    history = CompleteHistory([])
    history.add_message("user", "Test question")
    history.add_message("assistant", "Test answer")

    username = "testuser"

    # Save conversation
    success = conversation_store.create_or_update_conversation(
        username=username,
        conversation_id=conversation_id,
        history=history,
        conversation_title="Test Conversation",
        db_version=mock_assistant.searcher.db_version,
    )

    assert success

    # Load conversation
    loaded = conversation_store.load_single_conversation(username, conversation_id)
    assert loaded is not None
    assert loaded["conversation_title"] == "Test Conversation"
    assert len(loaded["messages"]) == len(history)

    # Delete conversation
    deleted = conversation_store.delete_conversation(username, conversation_id)
    assert deleted


@pytest.mark.skipif(
    not os.getenv("TEST_USE_REAL_SERVICES"),
    reason="Requires real services",
)
def test_knowledge_search_logging():
    """Test that knowledge searches are logged."""

    searches = knowledge_search_store.get_user_search_history("testuser")
    pre_number = len(searches)

    fake_plotly_figure = px.bar(x=[1, 2, 3], y=[4, 5, 6])
    figures = {
        "document_type": fake_plotly_figure,
        "mode": fake_plotly_figure,
        "agency": fake_plotly_figure,
        "year": fake_plotly_figure,
        "event_type": fake_plotly_figure,
    }
    search_settings = SearchParams(
        query="test query",
        search_type="fts",
        year_range=(2010, 2024),
        document_type=["safety_issue"],
        modes=["0", "1"],
        agencies=["TAIC"],
    )

    fake_results = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "title": ["Doc 1", "Doc 2", "Doc 3"],
            "content": [
                "Content of document 1",
                "Content of document 2",
                "Content of document 3",
            ],
        },
    )

    fake_download_dict = {
        "settings": search_settings,
        "search_start_time": datetime.datetime.now(),  # noqa: DTZ005
        "results": fake_results,
        "username": "testuser",
    }

    search_id = str(uuid.uuid4())
    expected_results = 3
    knowledge_search_store.store_search_log(
        username="testuser",
        search_id=search_id,
        search_settings=search_settings,
        relevance=0.5,
        results_info={"total_results": expected_results},
        results=fake_results,
        message="Test search message",
        download_dict=fake_download_dict,
        plots=figures,
        error_info=None,
    )

    # Check that search was logged
    searches = knowledge_search_store.get_user_search_history("testuser")
    assert len(searches) == pre_number + 1

    # Find our search
    our_search = knowledge_search_store.load_detailed_search("testuser", search_id)
    assert our_search is not None
    assert our_search["total_results"] == expected_results
    assert our_search["plots"].keys() == figures.keys()

    # Clean up by deleting the search log
    deleted = knowledge_search_store.delete_search_log("testuser", search_id)
    assert deleted
