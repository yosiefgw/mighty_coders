"""Hybrid search: vector similarity + ordinary SQL filters in ONE statement, joined back to articles.

    SELECT ... FROM chunks c JOIN articles a ON a.id = c.article_id
    WHERE <filters on articles, categories, links>
    ORDER BY VEC_DISTANCE_COSINE(c.embedding, <question vector>) LIMIT k

Three modes:
  mode="auto"  (default) MariaDB's optimizer picks the plan. In our tests it used the vector index
               when the filter keeps many chunks, and skipped it (exact distance on the few
               filtered chunks) when the filter keeps only a few.
  mode="index" FORCE INDEX: always walk the vector index (approximate HNSW search).
  mode="exact" IGNORE INDEX: no vector index, exact distance for every chunk that passes the filters.
               Used as the "ground truth" when we measure index recall.

Filters and the LIMIT. The docs (page "Vector Overview") warn that the LIMIT is a hard cap applied
BEFORE a WHERE filter, so a filtered query could return fewer than k rows. Our measurement on
MariaDB 11.8.9 (3-300 non-matching chunks nearer than the 3 matching ones, mhnsw_ef_search
1 and 20, with FORCE INDEX) always returned all 3 matching rows: the index search
continues until k rows pass the filter. tests/test_search.py checks this
(test_index_with_strict_filter_still_returns_k_rows). We therefore do NOT use a subquery with a
bigger inner LIMIT: that would itself be a hard cap.
Measured only on small synthetic data so far; to be re-checked on the real corpus.

Docs: https://mariadb.com/docs/server/reference/sql-structure/vectors/vector-overview
      https://mariadb.com/docs/server/reference/sql-structure/vectors/vector-system-variables

Example:
    from db import get_connection
    from search import search
    conn = get_connection()
    rows = search(conn, question_vector, k=5, category="Moons of Saturn", edited_since="2024-01-01")
    for r in rows:
        print(round(r["distance"], 3), r["title"], r["chunk_text"][:80])
"""
import json

DIM = 384                             # must match VECTOR(384) in sql/init/01_vector_schema.sql
INDEX_NAME = "idx_chunks_embedding"   # must match sql/init/01_vector_schema.sql
INDEX_HINTS = {
    "auto": "",
    "index": f"FORCE INDEX ({INDEX_NAME})",
    "exact": f"IGNORE INDEX ({INDEX_NAME})",
}


def to_vector_text(vector):
    """Python list of numbers -> the text form VEC_FromText() expects: '[0.1,0.2,...]'."""
    if len(vector) != DIM:
        raise ValueError(f"the question vector must have {DIM} numbers, got {len(vector)}")
    return json.dumps([float(x) for x in vector])


def _filters(category, min_words, edited_since, linked_from):
    """WHERE parts on the joined articles table `a`, plus their parameters."""
    where, params = [], []
    if category is not None:
        where.append(
            "EXISTS (SELECT 1 FROM article_categories ac JOIN categories cat ON cat.id = ac.category_id "
            "WHERE ac.article_id = a.id AND cat.name = %s)"
        )
        params.append(category)
    if min_words is not None:
        where.append("a.word_count >= %s")
        params.append(min_words)
    if edited_since is not None:
        where.append("a.last_edited >= %s")
        params.append(edited_since)
    if linked_from is not None:
        where.append(
            "EXISTS (SELECT 1 FROM links l JOIN articles src ON src.id = l.from_article_id "
            "WHERE src.title = %s AND l.to_article_id = a.id)"
        )
        params.append(linked_from)
    return where, params


def build_query(vector, k=5, *, category=None, min_words=None, edited_since=None,
                linked_from=None, mode="auto"):
    """Return (sql, params). Separate from search() so tests can run EXPLAIN on the same SQL."""
    if mode not in INDEX_HINTS:
        raise ValueError(f"mode must be one of {sorted(INDEX_HINTS)}, got {mode!r}")
    vec_text = to_vector_text(vector)
    where, filter_params = _filters(category, min_words, edited_since, linked_from)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    # ORDER BY must be the plain VEC_DISTANCE_COSINE(...) call (or its alias), ascending, with a
    # LIMIT, and the same distance as the index (cosine); otherwise the index cannot be used.
    sql = (
        "SELECT c.id AS chunk_id, c.article_id, a.title, c.chunk_index, c.chunk_text, "
        "a.word_count, a.last_edited, "
        "VEC_DISTANCE_COSINE(c.embedding, VEC_FromText(%s)) AS distance "
        f"FROM chunks c {INDEX_HINTS[mode]} "
        "JOIN articles a ON a.id = c.article_id "
        f"{where_sql} "
        "ORDER BY distance LIMIT %s"
    )
    return sql, [vec_text, *filter_params, k]


def search(conn, vector, k=5, *, category=None, min_words=None, edited_since=None,
           linked_from=None, mode="auto", ef_search=None):
    """The k chunks nearest to `vector` (a list of 384 numbers) that pass all given filters.

    Filters (all optional, combined with AND):
      category      only articles in this Wikipedia category (exact name, without "Category:")
      min_words     only articles with at least this many words
      edited_since  only articles last edited on/after this date, e.g. "2024-01-01"
      linked_from   only articles that the article with this title links to
    Tuning:
      mode          "auto" (optimizer decides), "index" (force the vector index), "exact" (no index)
      ef_search     sets mhnsw_ef_search for this connection (default 20; higher = better recall, slower)

    Returns a list of dicts, nearest first: chunk_id, article_id, title, chunk_index,
    chunk_text, word_count, last_edited, distance.
    """
    sql, params = build_query(vector, k, category=category, min_words=min_words,
                              edited_since=edited_since, linked_from=linked_from, mode=mode)
    cur = conn.cursor()
    if ef_search is not None:
        cur.execute("SET SESSION mhnsw_ef_search = %s", (ef_search,))
    cur.execute(sql, params)
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]
