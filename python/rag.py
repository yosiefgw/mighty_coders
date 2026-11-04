"""RAG pipeline: embed → retrieve → generate.

Usage
-----
    from rag import retrieve, answer
    # retrieve only:
    chunks = retrieve("What is Jupiter made of?")
    for c in chunks:
        print(c["article_title"], c["distance"])

    # full RAG (retrieve + LLM answer):
    response = answer("What is Jupiter made of?")
    print(response["answer"])
    print(response["sources"])

Requires ANTHROPIC_API_KEY in the environment (copy .env.example → .env).
"""
from __future__ import annotations

import os
from typing import List

import anthropic

from db import get_connection
from embed import embed
from search import search

DEFAULT_TOP_K = 5
LLM_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 512


def _retrieve(query_vec: List[float], top_k: int, conn=None, **kwargs) -> List[dict]:
    """Return the *top_k* closest chunks using search() from search.py.

    Uses the hybrid search function so callers can pass optional filters such as
    ``category``, ``min_words``, ``edited_since``, and ``linked_from``.

    Pass *conn* explicitly in tests to reuse the test transaction (and have it
    rolled back automatically). When *conn* is None a new connection is opened
    and closed after the query.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        rows = search(conn, query_vec, k=top_k, **kwargs)
    finally:
        if own_conn:
            conn.close()

    return [
        {
            "article_title": r["title"],
            "chunk_index": r["chunk_index"],
            "chunk_text": r["chunk_text"],
            "distance": float(r["distance"]),
        }
        for r in rows
    ]


def retrieve(question: str, top_k: int = DEFAULT_TOP_K, **kwargs) -> List[dict]:
    """Embed *question* and return the *top_k* nearest chunks from the DB.

    Optional keyword filters (category, min_words, edited_since, linked_from)
    are forwarded to search() for hybrid search.
    """
    query_vec = embed(question)
    return _retrieve(query_vec, top_k, **kwargs)


def _build_prompt(question: str, chunks: List[dict]) -> str:
    """Format retrieved chunks into a system prompt for the LLM."""
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[{i}] {chunk['article_title']}\n{chunk['chunk_text']}"
        )
    context = "\n\n".join(context_parts)
    return (
        "You are a helpful assistant answering questions about the Solar System "
        "based only on the provided Wikipedia excerpts. "
        "If the excerpts do not contain enough information, say so.\n\n"
        f"Excerpts:\n{context}\n\n"
        f"Question: {question}"
    )


def answer(question: str, top_k: int = DEFAULT_TOP_K, conn=None, **kwargs) -> dict:
    """Full RAG: retrieve relevant chunks, then generate an LLM answer.

    Returns a dict with:
      "answer"  – the model's response string
      "sources" – list of article titles used as context (deduplicated, ordered)
      "chunks"  – the raw list of dicts from _retrieve()
    """
    query_vec = embed(question)
    chunks = _retrieve(query_vec, top_k, conn=conn, **kwargs)

    prompt = _build_prompt(question, chunks)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    llm_answer = message.content[0].text

    seen = set()
    sources = []
    for chunk in chunks:
        t = chunk["article_title"]
        if t not in seen:
            seen.add(t)
            sources.append(t)

    return {"answer": llm_answer, "sources": sources, "chunks": chunks}
