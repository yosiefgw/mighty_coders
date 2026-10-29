
import numpy as np
import pytest

from db import get_connection
from search import DIM, search
from benchmarking import load_embeddings, brute_force_search


@pytest.fixture
def populated_database():
    try:
        conn = get_connection()
    except Exception as exc:
        pytest.skip(f"MariaDB is not available: {exc}")

    try:
        ids, vectors = load_embeddings(conn)

        if len(ids) == 0:
            pytest.skip("Waiting for the real dataset to be loaded")

        yield conn, ids, vectors

    finally:
        conn.close()


def test_real_embeddings_have_expected_dimensions(populated_database):
    _, ids, vectors = populated_database

    assert len(ids) > 0
    assert vectors.shape == (len(ids), DIM)
    assert np.isfinite(vectors).all()


def test_real_database_exact_search_matches_python(populated_database):
    conn, ids, vectors = populated_database

    query_index = len(ids) // 2
    query = vectors[query_index].tolist()
    k = min(5, len(ids))

    python_results = brute_force_search(ids, vectors, query, k=k)
    db_results = search(conn, query, k=k, mode="exact")

    assert db_results, "Exact MariaDB search returned no results"

    # Compare nearest-neighbour IDs. Equal-distance ties may be ordered
    # differently, so use overlap rather than requiring identical ordering.
    python_ids = {chunk_id for chunk_id, _ in python_results}
    db_ids = {row["chunk_id"] for row in db_results}

    assert len(python_ids & db_ids) > 0


def test_real_database_indexed_search_returns_results(populated_database):
    conn, ids, vectors = populated_database

    query = vectors[len(ids) // 2].tolist()
    k = min(5, len(ids))

    results = search(conn, query, k=k, mode="index")

    assert results, "Indexed MariaDB search returned no results"
    assert len(results) <= k
    assert all(np.isfinite(row["distance"]) for row in results)
