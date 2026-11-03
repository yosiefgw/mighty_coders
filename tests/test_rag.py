"""Tests for python/embed.py and python/rag.py.

embed tests need no database.
retrieve tests insert one article + one chunk inside the test transaction
(rolled back automatically by the conn fixture in conftest.py).
"""
import json
import math
import sys
import os

import pytest

from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from embed import EMBEDDING_DIM, embed, embed_batch  # noqa: E402
from rag import _build_prompt, _retrieve, answer     # noqa: E402

# Fake Wikipedia page id far from real ones so it never collides.
TEST_ARTICLE_ID = 4_294_960_001


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------

def test_embed_returns_correct_dimension():
    vec = embed("The Solar System contains eight planets.")
    assert len(vec) == EMBEDDING_DIM


def test_embed_returns_floats():
    vec = embed("Jupiter is a gas giant.")
    assert all(isinstance(x, float) for x in vec)


def test_embed_is_unit_normalized():
    vec = embed("Mars is the fourth planet from the Sun.")
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-5


def test_embed_different_texts_differ():
    a = embed("Jupiter")
    b = embed("Neptune")
    assert a != b


def test_embed_batch_matches_single():
    texts = ["Mercury", "Venus", "Earth"]
    batch = embed_batch(texts)
    singles = [embed(t) for t in texts]
    assert len(batch) == len(texts)
    for b_vec, s_vec in zip(batch, singles):
        assert len(b_vec) == EMBEDDING_DIM
        # Allow tiny floating-point differences between batch and single encoding.
        diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(b_vec, s_vec)))
        assert diff < 1e-4


# ---------------------------------------------------------------------------
# retrieve  (requires a running MariaDB with the schema loaded)
# ---------------------------------------------------------------------------

def _insert_test_data(conn, vec):
    """Insert one article and one chunk into the test transaction."""
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO articles (id, revision_id, title, url, word_count, last_edited) "
        "VALUES (%s, 1, %s, %s, 500, '2025-01-01 00:00:00')",
        (TEST_ARTICLE_ID, "zz_rag_test_article", "https://example.org/rag_test"),
    )
    cur.execute(
        "INSERT INTO chunks (article_id, chunk_index, word_count, chunk_text, embedding) "
        "VALUES (%s, 0, 180, %s, VEC_FromText(%s))",
        (TEST_ARTICLE_ID, "Jupiter is a gas giant composed mostly of hydrogen and helium.",
         json.dumps(vec)),
    )


def test_retrieve_returns_list_of_dicts(conn):
    query_vec = embed("What is Jupiter made of?")
    _insert_test_data(conn, query_vec)  # insert a chunk whose embedding matches the query

    results = _retrieve(query_vec, top_k=5, conn=conn)

    assert isinstance(results, list)
    assert len(results) >= 1


def test_retrieve_result_has_expected_keys(conn):
    query_vec = embed("What is Jupiter made of?")
    _insert_test_data(conn, query_vec)

    results = _retrieve(query_vec, top_k=5, conn=conn)
    row = results[0]

    assert set(row.keys()) == {"article_title", "chunk_index", "chunk_text", "distance"}


def test_retrieve_nearest_chunk_is_correct(conn):
    """The chunk we inserted with the query vector itself should be the closest hit."""
    query_vec = embed("What is Jupiter made of?")
    _insert_test_data(conn, query_vec)

    results = _retrieve(query_vec, top_k=5, conn=conn)
    best = results[0]

    assert best["article_title"] == "zz_rag_test_article"
    assert best["distance"] < 1e-5  # identical vectors → cosine distance ≈ 0


def test_retrieve_respects_top_k(conn):
    query_vec = embed("What is Jupiter made of?")
    _insert_test_data(conn, query_vec)

    results = _retrieve(query_vec, top_k=1, conn=conn)

    assert len(results) <= 1


def test_retrieve_distance_is_float(conn):
    query_vec = embed("Solar System")
    _insert_test_data(conn, query_vec)

    results = _retrieve(query_vec, top_k=5, conn=conn)

    assert all(isinstance(r["distance"], float) for r in results)


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

def test_build_prompt_contains_question():
    chunks = [{"article_title": "Jupiter", "chunk_text": "Jupiter is a gas giant."}]
    prompt = _build_prompt("What is Jupiter?", chunks)
    assert "What is Jupiter?" in prompt


def test_build_prompt_contains_chunk_text():
    chunks = [{"article_title": "Jupiter", "chunk_text": "Jupiter is a gas giant."}]
    prompt = _build_prompt("What is Jupiter?", chunks)
    assert "Jupiter is a gas giant." in prompt


def test_build_prompt_numbers_sources():
    chunks = [
        {"article_title": "Jupiter", "chunk_text": "Gas giant."},
        {"article_title": "Saturn", "chunk_text": "Ringed planet."},
    ]
    prompt = _build_prompt("Planets?", chunks)
    assert "[1]" in prompt
    assert "[2]" in prompt


# ---------------------------------------------------------------------------
# answer  (mocked Anthropic client — no real API call)
# ---------------------------------------------------------------------------

def _make_fake_chunks(conn):
    """Insert test data and return matching chunks for the mock."""
    query_vec = embed("What is Jupiter made of?")
    _insert_test_data(conn, query_vec)
    return _retrieve(query_vec, top_k=1, conn=conn)


def _mock_anthropic_client(reply: str):
    """Return a mock anthropic.Anthropic() that returns *reply*."""
    fake_content = MagicMock()
    fake_content.text = reply
    fake_message = MagicMock()
    fake_message.content = [fake_content]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_message
    return fake_client


def test_answer_returns_expected_keys(conn):
    _insert_test_data(conn, embed("What is Jupiter made of?"))
    fake_client = _mock_anthropic_client("Jupiter is mostly hydrogen.")

    with patch("rag.anthropic.Anthropic", return_value=fake_client), \
         patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        result = answer("What is Jupiter made of?", top_k=1, conn=conn)

    assert set(result.keys()) == {"answer", "sources", "chunks"}


def test_answer_string_and_list_types(conn):
    _insert_test_data(conn, embed("What is Jupiter made of?"))
    fake_client = _mock_anthropic_client("Jupiter is mostly hydrogen.")

    with patch("rag.anthropic.Anthropic", return_value=fake_client), \
         patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        result = answer("What is Jupiter made of?", top_k=1, conn=conn)

    assert isinstance(result["answer"], str)
    assert isinstance(result["sources"], list)
    assert isinstance(result["chunks"], list)


def test_answer_sources_deduplicated(conn):
    """Two chunks from the same article should produce only one source entry."""
    vec = embed("What is Jupiter made of?")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO articles (id, revision_id, title, url, word_count, last_edited) "
        "VALUES (%s, 1, %s, %s, 500, '2025-01-01 00:00:00')",
        (TEST_ARTICLE_ID, "zz_rag_test_article", "https://example.org/rag_test"),
    )
    for idx in range(2):
        cur.execute(
            "INSERT INTO chunks (article_id, chunk_index, word_count, chunk_text, embedding) "
            "VALUES (%s, %s, 180, %s, VEC_FromText(%s))",
            (TEST_ARTICLE_ID, idx, "Some chunk text.", json.dumps(vec)),
        )
    fake_client = _mock_anthropic_client("Answer.")

    with patch("rag.anthropic.Anthropic", return_value=fake_client), \
         patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        result = answer("What is Jupiter made of?", top_k=5, conn=conn)

    assert result["sources"].count("zz_rag_test_article") == 1


def test_answer_uses_llm_reply(conn):
    _insert_test_data(conn, embed("What is Jupiter made of?"))
    expected_reply = "Jupiter is composed of hydrogen and helium."
    fake_client = _mock_anthropic_client(expected_reply)

    with patch("rag.anthropic.Anthropic", return_value=fake_client), \
         patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        result = answer("What is Jupiter made of?", top_k=1, conn=conn)

    assert result["answer"] == expected_reply
