"""Embedding helper using BAAI/bge-small-en-v1.5 (384-dim).

The model is downloaded on first use and cached by sentence-transformers.
All public functions return plain Python lists so they're JSON-serialisable
and easy to pass straight to pymysql / MariaDB VEC_FromText().
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
# BGE models improve retrieval recall when queries are prefixed at search time.
# Apply this to questions only; chunk text is indexed without the prefix.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(text: str) -> List[float]:
    """Return a 384-dimensional embedding for *text*."""
    vec = _model().encode(text, normalize_embeddings=True)
    return vec.tolist()


def embed_query(text: str) -> List[float]:
    """Return a 384-dimensional embedding for a search question.

    Prepends QUERY_PREFIX as recommended for BAAI/bge-small-en-v1.5:
    the prefix is applied to questions at retrieval time only; chunk text
    is indexed with plain embed() so the asymmetry improves recall.
    Measured improvement on our 44-question ground-truth set:
    hit@5 0.86 → 0.91, hit@1 0.45 → 0.55, MRR 0.61 → 0.68.
    """
    return embed(QUERY_PREFIX + text)


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Return embeddings for a list of texts (faster than calling embed() in a loop)."""
    vecs = _model().encode(texts, normalize_embeddings=True, batch_size=64)
    return [v.tolist() for v in vecs]
