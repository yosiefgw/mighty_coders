
import numpy as np
import pytest

from search import DIM
from benchmarking import (
    cosine_distances,
    brute_force_search,
    measure,
)


def test_identical_vector_has_zero_distance():
    query = np.ones(DIM, dtype=np.float32)

    distances = cosine_distances(
        np.array([query, -query]),
        query,
    )

    assert distances[0] == pytest.approx(0.0, abs=1e-6)
    assert distances[1] == pytest.approx(2.0, abs=1e-6)



def test_brute_force_returns_nearest_chunks():
    query = np.zeros(DIM, dtype=np.float32)
    query[0] = 1.0

    vectors = np.zeros((3, DIM), dtype=np.float32)

    # Vector 10 is perpendicular to the query: cosine distance = 1.
    vectors[0, 1] = 1.0

    # Vector 20 is identical to the query: cosine distance = 0.
    vectors[1, 0] = 1.0

    # Vector 30 points in the opposite direction: distance = 2.
    vectors[2, 0] = -1.0

    ids = [10, 20, 30]

    results = brute_force_search(ids, vectors, query, k=2)

    assert results[0][0] == 20
    assert results[0][1] == pytest.approx(0.0)
    assert results[1][0] == 10
    assert results[1][1] == pytest.approx(1.0)



def test_invalid_query_dimension():
    with pytest.raises(ValueError):
        cosine_distances(
            np.ones((2, DIM), dtype=np.float32),
            np.ones(DIM - 1, dtype=np.float32),
        )


def test_zero_vector_is_rejected():
    with pytest.raises(ValueError):
        cosine_distances(
            np.zeros((1, DIM), dtype=np.float32),
            np.ones(DIM, dtype=np.float32),
        )


def test_non_finite_values_are_rejected():
    vectors = np.ones((1, DIM), dtype=np.float32)
    query = np.ones(DIM, dtype=np.float32)
    query[0] = np.nan

    with pytest.raises(ValueError):
        cosine_distances(vectors, query)


def test_measure_records_all_repetitions():
    result, timings = measure(lambda: 42, repeats=3)

    assert result == 42
    assert len(timings) == 3
    assert all(timing >= 0 for timing in timings)


def test_invalid_k_is_rejected():
    with pytest.raises(ValueError):
        brute_force_search(
            [1],
            np.ones((1, DIM), dtype=np.float32),
            np.ones(DIM, dtype=np.float32),
            k=0,
        )
