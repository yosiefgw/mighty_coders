"""Tests for python/load.py with tiny fake data (no files needed: load() takes lists of dicts).
Everything runs inside the test's transaction (commit=False) and is rolled back afterwards.
Test ids are near the INT UNSIGNED maximum, so they never clash with real Wikipedia page ids."""
import pytest

from load import DIM, check, load, read_jsonl, to_datetime
from search import search

A, B, C = 4_294_940_001, 4_294_940_002, 4_294_940_003


def vec(k, dim=DIM):
    return [((i * k) % 11 + 1) / 10 for i in range(1, dim + 1)]


def article(article_id, revision=1, links_to=(), categories=("zz_test_cat",)):
    return {
        "id": article_id, "revision_id": revision, "title": f"zz_test {article_id}",
        "url": f"https://example.org/{article_id}", "word_count": 1000,
        "last_edited": "2025-06-01T12:34:56Z",
        "categories": list(categories), "links_to": list(links_to),
    }


def chunk(article_id, index, k):
    return {"article_id": article_id, "chunk_index": index, "word_count": 150,
            "text": f"chunk {index} of {article_id}", "embedding": vec(k)}


def count(conn, sql, *params):
    cur = conn.cursor()
    cur.execute(sql, params)
    return cur.fetchone()[0]


def chunks_of(conn, article_id):
    return count(conn, "SELECT COUNT(*) FROM chunks WHERE article_id = %s", article_id)


ARTICLES = [
    article(A, links_to=[f"zz_test {B}", "Some page outside the corpus"]),
    article(B, links_to=[f"zz_test {A}"], categories=("zz_test_cat", "zz_test_other")),
]
CHUNKS = [chunk(A, 0, 1), chunk(A, 1, 2), chunk(B, 0, 3)]


# ---------- small helpers ----------

def test_to_datetime():
    assert to_datetime("2025-06-01T12:34:56Z") == "2025-06-01 12:34:56"


def test_read_jsonl(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n\n{"x": 2}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"x": 1}, {"x": 2}]


# ---------- checks before writing ----------

@pytest.mark.parametrize("bad_chunks, message", [
    ([chunk(C, 0, 1)], "is not in the articles file"),
    ([chunk(A, 0, 1), chunk(A, 0, 2)], "appears twice"),
    ([{**chunk(A, 0, 1), "embedding": [0.1, 0.2, 0.3]}], "expected 384"),
])
def test_check_rejects_bad_input(bad_chunks, message):
    with pytest.raises(ValueError, match=message):
        check(ARTICLES, bad_chunks)


def test_bad_input_writes_nothing(conn):
    with pytest.raises(ValueError):
        load(conn, ARTICLES, [chunk(A, 0, 1), chunk(C, 0, 2)], commit=False)
    assert count(conn, "SELECT COUNT(*) FROM articles WHERE id IN (%s, %s)", A, B) == 0


# ---------- first load ----------

def test_first_load_fills_all_tables(conn):
    stats = load(conn, ARTICLES, CHUNKS, commit=False)
    assert stats["articles"] == 2
    assert stats["chunks_inserted"] == 3
    assert chunks_of(conn, A) == 2 and chunks_of(conn, B) == 1
    assert count(conn, "SELECT COUNT(*) FROM article_categories WHERE article_id = %s", B) == 2
    assert count(conn, "SELECT last_edited FROM articles WHERE id = %s", A).isoformat() == "2025-06-01T12:34:56"


def test_links_only_inside_corpus(conn):
    stats = load(conn, ARTICLES, CHUNKS, commit=False)
    assert stats["links"] == 2  # A->B and B->A; the page outside the corpus is skipped
    assert count(conn, "SELECT COUNT(*) FROM links WHERE from_article_id = %s AND to_article_id = %s", A, B) == 1


def test_loaded_chunks_are_searchable(conn):
    load(conn, ARTICLES, CHUNKS, commit=False)
    rows = search(conn, vec(3), k=1, category="zz_test_other")
    assert rows[0]["article_id"] == B and rows[0]["distance"] < 1e-6


# ---------- reload ----------

def test_reload_same_revision_keeps_chunks(conn):
    load(conn, ARTICLES, CHUNKS, commit=False)
    ids_before = count(conn, "SELECT GROUP_CONCAT(id ORDER BY id) FROM chunks WHERE article_id = %s", A)
    stats = load(conn, ARTICLES, CHUNKS, commit=False)
    assert stats["articles_unchanged"] == 2 and stats.get("chunks_inserted", 0) == 0
    assert count(conn, "SELECT GROUP_CONCAT(id ORDER BY id) FROM chunks WHERE article_id = %s", A) == ids_before


def test_reload_new_revision_replaces_chunks(conn):
    load(conn, ARTICLES, CHUNKS, commit=False)
    new_a = article(A, revision=2, links_to=[f"zz_test {B}"])
    stats = load(conn, [new_a, ARTICLES[1]], [chunk(A, 0, 4), chunk(B, 0, 3)], commit=False)
    assert stats["chunks_deleted"] == 2 and stats["chunks_inserted"] == 1
    assert chunks_of(conn, A) == 1
    assert count(conn, "SELECT revision_id FROM articles WHERE id = %s", A) == 2
    # the new vector is found, the old ones are gone
    assert search(conn, vec(4), k=1, category="zz_test_cat")[0]["article_id"] == A
    assert count(conn, "SELECT COUNT(*) FROM chunks WHERE article_id = %s AND chunk_index = 1", A) == 0


def test_force_reloads_unchanged_article(conn):
    load(conn, ARTICLES, CHUNKS, commit=False)
    stats = load(conn, ARTICLES, CHUNKS, force=True, commit=False)
    assert stats["chunks_deleted"] == 3 and stats["chunks_inserted"] == 3


def test_reload_keeps_incoming_links(conn):
    """The article row is updated, never deleted, so links pointing to it (B->A) survive."""
    load(conn, ARTICLES, CHUNKS, commit=False)
    load(conn, [article(A, revision=2)], [chunk(A, 0, 1)], commit=False)
    assert count(conn, "SELECT COUNT(*) FROM links WHERE from_article_id = %s AND to_article_id = %s", B, A) == 1


def test_vector_index_still_finds_all_chunks_after_reload(conn):
    """After reloads (direct DELETE + INSERT), the index search finds every live chunk."""
    load(conn, ARTICLES, CHUNKS, commit=False)
    load(conn, ARTICLES, CHUNKS, force=True, commit=False)
    rows = search(conn, vec(1), k=10, category="zz_test_cat")
    assert sorted((r["article_id"], r["chunk_index"]) for r in rows) == [(A, 0), (A, 1), (B, 0)]
