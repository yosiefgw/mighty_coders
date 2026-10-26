"""Tests for python/search.py (hybrid search) with a small fake dataset.

Fake data (inserted inside the test's transaction, rolled back afterwards):
  3 "near" articles: category zz_test_near, 5000 words, edited 2025-06-01,
                     each chunk has exactly the query vector (distance 0)
  3 "far" articles:  category zz_test_far, 100 words, edited 2020-01-01,
                     chunks point in other directions (distances 0.109, 0.130, 0.217: no ties)
  link: near article 0 -> far article 0
Test ids are near the INT UNSIGNED maximum, so they never clash with real Wikipedia page ids.
"""
import pytest

from search import INDEX_NAME, build_query, search, to_vector_text

DIM = 384
TEST_ID = 4_294_950_000
NEAR = [TEST_ID + i for i in range(3)]
FAR = [TEST_ID + 10 + i for i in range(3)]


def vec(k, dim=DIM):
    """A fixed test vector; different k gives a different direction (same helper as test_schema.py)."""
    return [((i * k) % 11 + 1) / 10 for i in range(1, dim + 1)]


QUERY = vec(1)


def add_article(cur, article_id, words, edited, category, vector):
    cur.execute(
        "INSERT INTO articles (id, revision_id, title, url, word_count, last_edited) "
        "VALUES (%s, 1, %s, 'x', %s, %s)",
        (article_id, f"zz_test {article_id}", words, edited),
    )
    cur.execute(
        "INSERT INTO article_categories (article_id, category_id) "
        "SELECT %s, id FROM categories WHERE name = %s",
        (article_id, category),
    )
    cur.execute(
        "INSERT INTO chunks (article_id, chunk_index, word_count, chunk_text, embedding) "
        "VALUES (%s, 0, 2, 'test chunk', VEC_FromText(%s))",
        (article_id, to_vector_text(vector)),
    )


@pytest.fixture
def data(conn):
    cur = conn.cursor()
    cur.execute("INSERT INTO categories (name) VALUES ('zz_test_near'), ('zz_test_far')")
    for article_id in NEAR:
        add_article(cur, article_id, 5000, "2025-06-01", "zz_test_near", vec(1))
    for article_id, v in zip(FAR, [vec(2), vec(3), vec(5)]):
        add_article(cur, article_id, 100, "2020-01-01", "zz_test_far", v)
    cur.execute("INSERT INTO links (from_article_id, to_article_id) VALUES (%s, %s)", (NEAR[0], FAR[0]))
    return conn


def ids(rows):
    return [r["article_id"] for r in rows]


# ---------- plain search ----------

def test_nearest_are_the_identical_vectors(data):
    rows = search(data, QUERY, k=3)
    assert set(ids(rows)) == set(NEAR)
    assert all(r["distance"] < 1e-6 for r in rows)


def test_result_has_article_columns(data):
    row = search(data, QUERY, k=1)[0]
    assert {"chunk_id", "article_id", "title", "chunk_index", "chunk_text",
            "word_count", "last_edited", "distance"} <= row.keys()


def test_wrong_vector_size_is_rejected():
    with pytest.raises(ValueError):
        build_query([0.1, 0.2, 0.3])


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        build_query(QUERY, mode="fast")


# ---------- each filter ----------

def test_filter_category(data):
    rows = search(data, QUERY, k=3, category="zz_test_far")
    assert ids(rows) == FAR                  # nearest first: distances 0.109, 0.130, 0.217


def test_filter_min_words(data):
    rows = search(data, QUERY, k=10, min_words=1000)
    assert set(ids(rows)) >= set(NEAR) and not set(ids(rows)) & set(FAR)


def test_filter_edited_since(data):
    rows = search(data, QUERY, k=10, edited_since="2025-01-01")
    assert not set(ids(rows)) & set(FAR)


def test_filter_linked_from(data):
    rows = search(data, QUERY, k=5, linked_from=f"zz_test {NEAR[0]}")
    assert ids(rows) == [FAR[0]]


def test_filters_combined(data):
    rows = search(data, QUERY, k=5, category="zz_test_far", linked_from=f"zz_test {NEAR[0]}")
    assert ids(rows) == [FAR[0]]
    rows = search(data, QUERY, k=5, category="zz_test_far", min_words=1000)
    assert rows == []


# ---------- the three modes ----------

@pytest.mark.parametrize("mode", ["auto", "index", "exact"])
def test_all_modes_give_the_same_result_on_small_data(data, mode):
    assert ids(search(data, QUERY, k=3, category="zz_test_far", mode=mode)) == FAR


def explain_keys(conn, sql, params):
    cur = conn.cursor()
    cur.execute("EXPLAIN " + sql, params)
    key_col = [d[0] for d in cur.description].index("key")
    return [row[key_col] for row in cur.fetchall()]


def test_explain_index_mode_uses_vector_index(data):
    sql, params = build_query(QUERY, k=3, category="zz_test_far", mode="index")
    assert INDEX_NAME in explain_keys(data, sql, params)


def test_explain_exact_mode_does_not_use_vector_index(data):
    sql, params = build_query(QUERY, k=3, category="zz_test_far", mode="exact")
    assert INDEX_NAME not in explain_keys(data, sql, params)


# ---------- the docs' "hard cap" warning, measured ----------

def test_index_with_strict_filter_still_returns_k_rows(data):
    """MariaDB docs (Vector Overview) warn that the LIMIT is a hard cap applied BEFORE a WHERE
    filter, so a filtered query could return fewer than k rows. On 11.8.9 we measured the
    opposite: the index search continues until k rows pass the filter.
    Here 300 extra "near" chunks are nearer than the 3 "far" ones; we force the index and use the
    smallest search effort. If this test FAILS, MariaDB now caps before the filter: update
    search.py (its docstring and the query)."""
    cur = data.cursor()
    for j in range(300):
        v = list(QUERY)
        v[j] *= 1.05                     # almost the query vector: much nearer than any "far" chunk
        add_article(cur, TEST_ID + 100 + j, 5000, "2025-06-01", "zz_test_near", v)
    sql, params = build_query(QUERY, k=3, category="zz_test_far", mode="index")
    assert INDEX_NAME in explain_keys(data, sql, params)
    rows = search(data, QUERY, k=3, category="zz_test_far", mode="index", ef_search=1)
    assert ids(rows) == FAR
    cur.execute("SET SESSION mhnsw_ef_search = DEFAULT")


def test_ef_search_is_set(data):
    search(data, QUERY, k=1, ef_search=50)
    cur = data.cursor()
    cur.execute("SELECT @@SESSION.mhnsw_ef_search")
    assert cur.fetchone()[0] == 50
    cur.execute("SET SESSION mhnsw_ef_search = DEFAULT")
