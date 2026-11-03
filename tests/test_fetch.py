"""Tests for python/fetch.py.

All tests run without network access: the MediaWiki API is mocked.
Heavy ML dependencies (torch / sentence_transformers) are stubbed out so the
tests run without a GPU or a full torch installation.
"""
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

# Stub out ML libs before any project import triggers them.
for _mod in ("torch", "sentence_transformers", "sentence_transformers.SentenceTransformer"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from fetch import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_TARGET,
    _fetch_batch,
    build_chunks,
    chunk_text,
    collect_titles,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api_response(**extra):
    """Minimal MediaWiki API response skeleton."""
    return {"query": {}, **extra}


def _make_page(pageid, title, extract="Some text " * 50, revid=100, timestamp="2025-01-01T00:00:00Z",
               categories=None, links=None):
    return {
        "pageid": pageid,
        "title": title,
        "extract": extract,
        "revisions": [{"revid": revid, "timestamp": timestamp}],
        "categories": [{"title": f"Category:{c}"} for c in (categories or [])],
        "links": [{"title": t} for t in (links or [])],
    }


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

def test_chunk_text_empty_string():
    assert chunk_text("") == []


def test_chunk_text_short_text_returns_one_chunk():
    text = "word " * 10
    result = chunk_text(text.strip())
    assert len(result) == 1


def test_chunk_text_long_text_produces_multiple_chunks():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text)
    assert len(chunks) > 1


def test_chunk_text_respects_target_size():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text)
    for c in chunks[:-1]:  # last chunk may be shorter
        assert len(c.split()) <= CHUNK_TARGET


def test_chunk_text_overlap_exists():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text)
    if len(chunks) >= 2:
        last_of_first = set(chunks[0].split()[-CHUNK_OVERLAP:])
        first_of_second = set(chunks[1].split()[:CHUNK_OVERLAP])
        assert last_of_first & first_of_second  # some overlap


def test_chunk_text_covers_all_words():
    words = [f"w{i}" for i in range(300)]
    text = " ".join(words)
    chunks = chunk_text(text)
    # Every word must appear in at least one chunk.
    all_in_chunks = set(" ".join(chunks).split())
    assert set(words) == all_in_chunks


def test_chunk_text_very_short_below_overlap():
    text = "just three words"
    result = chunk_text(text)
    assert result == [text]


# ---------------------------------------------------------------------------
# collect_titles (mocked API)
# ---------------------------------------------------------------------------

def _make_catmembers_response(members, cm_type):
    return {
        "query": {
            "categorymembers": [{"title": m} for m in members]
        }
    }


def test_collect_titles_basic():
    pages = ["Mars", "Jupiter", "Saturn"]

    def fake_get(params):
        if params.get("cmtype") == "page":
            return _make_catmembers_response(pages, "page")
        return _make_catmembers_response([], "subcat")

    with patch("fetch._get", side_effect=fake_get):
        titles = collect_titles("Solar System", depth=0)

    assert set(titles) == set(pages)


def test_collect_titles_depth_recurses_into_subcats():
    def fake_get(params):
        title = params.get("cmtitle", "")
        if params.get("cmtype") == "subcat":
            if "Solar System" in title:
                return _make_catmembers_response(["Category:Planets"], "subcat")
            return _make_catmembers_response([], "subcat")
        # pages
        if "Solar System" in title:
            return _make_catmembers_response(["Sun"], "page")
        if "Planets" in title:
            return _make_catmembers_response(["Mars", "Jupiter"], "page")
        return _make_catmembers_response([], "page")

    with patch("fetch._get", side_effect=fake_get):
        titles = collect_titles("Solar System", depth=1)

    assert "Sun" in titles
    assert "Mars" in titles
    assert "Jupiter" in titles


def test_collect_titles_depth_zero_ignores_subcats():
    def fake_get(params):
        if params.get("cmtype") == "page":
            return _make_catmembers_response(["Sun"], "page")
        return _make_catmembers_response(["Category:Planets"], "subcat")

    with patch("fetch._get", side_effect=fake_get):
        titles = collect_titles("Solar System", depth=0)

    assert "Sun" in titles
    assert "Mars" not in titles


# ---------------------------------------------------------------------------
# _fetch_batch (mocked API)
# ---------------------------------------------------------------------------

def test_fetch_batch_parses_basic_fields():
    page = _make_page(19331, "Mars", extract="Mars is a planet. " * 20,
                      revid=12345, timestamp="2025-06-01T12:00:00Z",
                      categories=["Planets"], links=["Jupiter"])
    with patch("fetch._get", return_value={"query": {"pages": [page]}}):
        result = _fetch_batch(["Mars"])

    assert len(result) == 1
    art = result[0]
    assert art["id"] == 19331
    assert art["title"] == "Mars"
    assert art["revision_id"] == 12345
    assert art["last_edited"] == "2025-06-01T12:00:00Z"
    assert "Planets" in art["categories"]
    assert "Jupiter" in art["links_to"]
    assert art["word_count"] > 0


def test_fetch_batch_skips_missing_pages():
    page = {"title": "NonExistent", "missing": True}
    with patch("fetch._get", return_value={"query": {"pages": [page]}}):
        result = _fetch_batch(["NonExistent"])
    assert result == []


def test_fetch_batch_skips_empty_extract():
    page = _make_page(1, "EmptyPage", extract="   ")
    with patch("fetch._get", return_value={"query": {"pages": [page]}}):
        result = _fetch_batch(["EmptyPage"])
    assert result == []


def test_fetch_batch_strips_category_prefix():
    page = _make_page(1, "Mars", categories=["Planets of the Solar System"])
    with patch("fetch._get", return_value={"query": {"pages": [page]}}):
        result = _fetch_batch(["Mars"])
    # categories should NOT have "Category:" prefix
    assert all(not c.startswith("Category:") for c in result[0]["categories"])


def test_fetch_batch_url_uses_underscores():
    page = _make_page(1, "Red Planet")
    with patch("fetch._get", return_value={"query": {"pages": [page]}}):
        result = _fetch_batch(["Red Planet"])
    assert "Red_Planet" in result[0]["url"]


# ---------------------------------------------------------------------------
# build_chunks (mocked embed_batch)
# ---------------------------------------------------------------------------

def _fake_embed_batch(texts):
    return [[0.1] * 384 for _ in texts]


def test_build_chunks_returns_correct_count():
    text = " ".join(f"word{i}" for i in range(400))
    articles = [{"id": 1, "text": text}]
    expected_chunks = len(chunk_text(text))

    with patch("fetch.embed_batch", side_effect=_fake_embed_batch):
        chunks = build_chunks(articles)

    assert len(chunks) == expected_chunks


def test_build_chunks_embedding_dimension():
    text = " ".join(f"word{i}" for i in range(200))
    articles = [{"id": 42, "text": text}]

    with patch("fetch.embed_batch", side_effect=_fake_embed_batch):
        chunks = build_chunks(articles)

    assert all(len(c["embedding"]) == 384 for c in chunks)


def test_build_chunks_chunk_index_sequential():
    text = " ".join(f"word{i}" for i in range(400))
    articles = [{"id": 1, "text": text}]

    with patch("fetch.embed_batch", side_effect=_fake_embed_batch):
        chunks = build_chunks(articles)

    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_build_chunks_empty_text():
    with patch("fetch.embed_batch", side_effect=_fake_embed_batch):
        chunks = build_chunks([{"id": 1, "text": ""}])
    assert chunks == []


def test_build_chunks_multiple_articles():
    text = " ".join(f"word{i}" for i in range(200))
    articles = [{"id": 1, "text": text}, {"id": 2, "text": text}]

    with patch("fetch.embed_batch", side_effect=_fake_embed_batch):
        chunks = build_chunks(articles)

    ids = {c["article_id"] for c in chunks}
    assert ids == {1, 2}
