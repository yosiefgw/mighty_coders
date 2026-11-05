
"""Unit tests for the Solar System Explorer Streamlit UI."""

import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest


UI_PATH = (
    Path(__file__).resolve().parents[1]
    / "python"
    / "query_ui.py"
)


@pytest.fixture
def ui_mocks(monkeypatch):
    """Mock embedding, database connection and MariaDB search."""

    import db
    import embed
    import search

    # Mock database connection
    conn = Mock()

    cursor = Mock()
    cursor.fetchone.return_value = (
        "mighty_coders",
        "localhost",
        3308,
        "11.8.9-MariaDB",
    )

    conn.cursor.return_value = cursor

    get_connection_mock = Mock(return_value=conn)

    # Mock embedding generation
    embed_mock = Mock(return_value=[0.1] * 384)

    # Mock MariaDB search
    search_mock = Mock(return_value=[])

    monkeypatch.setattr(db, "get_connection", get_connection_mock)
    monkeypatch.setattr(embed, "embed", embed_mock)
    monkeypatch.setattr(search, "search", search_mock)

    return {
        "conn": conn,
        "get_connection": get_connection_mock,
        "embed": embed_mock,
        "search": search_mock,
    }


def create_app():
    """Create a fresh Streamlit application test instance."""
    return AppTest.from_file(str(UI_PATH)).run(timeout=20)


def submit_question(app, question):
    """Enter a question and click the search button."""
    app.text_input[0].set_value(question)
    app.button[0].click()
    return app.run()


def sample_result(
    title="Jupiter",
    chunk_index=0,
    distance=0.15,
    chunk_text="Jupiter is a gas giant.",
):
    """Return a sample MariaDB search result."""
    return {
        "chunk_id": 1,
        "article_id": 1,
        "title": title,
        "chunk_index": chunk_index,
        "chunk_text": chunk_text,
        "word_count": 500,
        "last_edited": datetime.date(2025, 1, 1),
        "distance": distance,
        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
    }


def test_page_loads():
    """The application should load without errors."""
    app = create_app()

    assert len(app.exception) == 0
    assert any(
        "Solar System Explorer" in element.value
        for element in app.title
    )


def test_empty_question_displays_warning(ui_mocks):
    """An empty question should display a warning."""
    app = create_app()

    app = submit_question(app, "")

    assert any(
        "Please enter a question or select a category." in warning.value
        for warning in app.warning
    )

    ui_mocks["embed"].assert_not_called()
    ui_mocks["get_connection"].assert_not_called()
    ui_mocks["search"].assert_not_called()


def test_valid_question_generates_embedding(ui_mocks):
    """A valid question should generate a 384-dimensional embedding."""
    ui_mocks["search"].return_value = [
        sample_result()
    ]

    app = create_app()
    app = submit_question(app, "What is Jupiter made of?")

    ui_mocks["embed"].assert_called_once_with(
        "What is Jupiter made of?"
    )

    assert len(ui_mocks["embed"].return_value) == 384
    assert len(app.exception) == 0


def test_valid_question_connects_to_database(ui_mocks):
    """A valid question should connect to MariaDB."""
    ui_mocks["search"].return_value = [
        sample_result()
    ]

    app = create_app()
    app = submit_question(app, "What is Jupiter made of?")

    ui_mocks["get_connection"].assert_called_once()
    ui_mocks["conn"].close.assert_called_once()

    assert len(app.exception) == 0


def test_valid_question_calls_vector_search(ui_mocks):
    """The UI should pass the generated vector to search()."""
    query_vector = [0.1] * 384
    ui_mocks["embed"].return_value = query_vector
    ui_mocks["search"].return_value = [
        sample_result()
    ]

    app = create_app()
    app = submit_question(app, "What is Jupiter made of?")

    ui_mocks["search"].assert_called_once()

    args, kwargs = ui_mocks["search"].call_args

    assert args[0] is ui_mocks["conn"]
    assert args[1] == query_vector
    assert kwargs["k"] == 25


def test_search_applies_word_count_filter(ui_mocks):
    """The minimum word-count filter should reach MariaDB search."""
    ui_mocks["search"].return_value = [
        sample_result()
    ]

    app = create_app()

    app.number_input[0].set_value(500)

    app = submit_question(app, "Jupiter")

    _, kwargs = ui_mocks["search"].call_args

    assert kwargs["min_words"] == 500


def test_search_applies_edited_since_filter(ui_mocks):
    """The edited-since filter should reach MariaDB search."""
    ui_mocks["search"].return_value = [
        sample_result()
    ]

    app = create_app()

    selected_date = datetime.date(2025, 1, 1)
    app.date_input[0].set_value(selected_date)

    app = submit_question(app, "Jupiter")

    _, kwargs = ui_mocks["search"].call_args

    assert kwargs["edited_since"] == "2025-01-01"


def test_duplicate_articles_are_removed(ui_mocks):
    """Keep one result per article and show alternatives in sidebar."""
    ui_mocks["search"].return_value = [
        sample_result(
            title="Jupiter",
            chunk_index=0,
            distance=0.10,
        ),
        sample_result(
            title="Jupiter",
            chunk_index=1,
            distance=0.15,
        ),
        sample_result(
            title="Saturn",
            chunk_index=0,
            distance=0.20,
        ),
    ]

    app = create_app()
    app = submit_question(app, "Planets")

    # The best matching article appears in the main page.
    displayed_text = "\n".join(
        element.value for element in app.markdown
    )
    assert "Jupiter" in displayed_text

    # Both unique articles should be selectable in the sidebar.
    assert len(app.sidebar.radio) == 1

    suggestion_radio = app.sidebar.radio[0]

    assert len(suggestion_radio.options) == 2
    assert "Jupiter" in suggestion_radio.format_func(0)
    assert "Saturn" in suggestion_radio.format_func(1)

def test_no_results_displays_information(ui_mocks):
    """An empty database search result should display an information message."""
    ui_mocks["search"].return_value = []

    app = create_app()
    app = submit_question(app, "Unknown object")

    assert any(
        "No matching results found" in item.value
        for item in app.info
    )


def test_search_failure_displays_error(ui_mocks):
    """A search exception should be handled by the UI."""
    ui_mocks["search"].side_effect = RuntimeError(
        "Database search failed"
    )

    app = create_app()
    app = submit_question(app, "Jupiter")

    assert any(
        "Search failed" in item.value
        for item in app.error
    )

    ui_mocks["conn"].close.assert_called_once()


def test_database_connection_closes_after_success(ui_mocks):
    """The database connection should close after successful retrieval."""
    ui_mocks["search"].return_value = [
        sample_result()
    ]

    app = create_app()
    app = submit_question(app, "Jupiter")

    ui_mocks["conn"].close.assert_called_once()
    assert len(app.exception) == 0
