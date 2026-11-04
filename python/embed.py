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


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(text: str) -> List[float]:
    """Return a 384-dimensional embedding for *text*."""
    vec = _model().encode(text, normalize_embeddings=True)
    return vec.tolist()


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Return embeddings for a list of texts (faster than calling embed() in a loop)."""
    vecs = _model().encode(texts, normalize_embeddings=True, batch_size=64)
    return [v.tolist() for v in vecs]
