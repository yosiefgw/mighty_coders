"""RAG pipeline: embed → retrieve.

Usage
-----
    from rag import retrieve
    chunks = retrieve("What is Jupiter made of?")
    for c in chunks:
        print(c["article_title"], c["distance"])

The generate step (LLM answer) is not implemented yet — it will be added
once the team has an API key.
"""
from __future__ import annotations

import json
from typing import List

from db import get_connection
from embed import embed

DEFAULT_TOP_K = 5


def _retrieve(query_vec: List[float], top_k: int, conn=None) -> List[dict]:
    """Return the *top_k* closest chunks from MariaDB using VEC_DISTANCE_COSINE().

    Uses VEC_FromText() with a JSON array string — the same format that load.py
    uses for insertion — so the HNSW vector index (DISTANCE=cosine) is used.

    Pass *conn* explicitly in tests to reuse the test transaction (and have it
    rolled back automatically). When *conn* is None a new connection is opened
    and closed after the query.
    """
    vec_json = json.dumps(query_vec)
    sql = """
        SELECT
            a.title,
            c.chunk_index,
            c.chunk_text,
            VEC_DISTANCE_COSINE(c.embedding, VEC_FromText(%s)) AS distance
        FROM chunks c
        JOIN articles a ON a.id = c.article_id
        ORDER BY distance ASC
        LIMIT %s
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (vec_json, top_k))
            rows = cur.fetchall()
    finally:
        if own_conn:
            conn.close()

    return [
        {
            "article_title": r[0],
            "chunk_index": r[1],
            "chunk_text": r[2],
            "distance": float(r[3]),
        }
        for r in rows
    ]


def retrieve(question: str, top_k: int = DEFAULT_TOP_K) -> List[dict]:
    """Embed *question* and return the *top_k* nearest chunks from the DB."""
    query_vec = embed(question)
    return _retrieve(query_vec, top_k)
