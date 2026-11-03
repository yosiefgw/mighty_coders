
"""Benchmark Python brute-force cosine search vs MariaDB VECTOR search."""

import json
import statistics
import time
import matplotlib.pyplot as plt
import numpy as np

from db import get_connection
from search import DIM, search, to_vector_text


def cosine_distances(vectors, query):
    """Calculate cosine distance between a query and every document vector."""
    vectors = np.asarray(vectors, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)

    if vectors.ndim != 2 or vectors.shape[1] != DIM:
        raise ValueError(f"Expected vectors with {DIM} dimensions")

    if query.shape != (DIM,):
        raise ValueError(f"Query must contain {DIM} values")

    if not np.isfinite(vectors).all() or not np.isfinite(query).all():
        raise ValueError("Vectors must contain finite values")

    vector_norms = np.linalg.norm(vectors, axis=1)
    query_norm = np.linalg.norm(query)

    if query_norm == 0 or np.any(vector_norms == 0):
        raise ValueError("Zero vectors cannot be compared")

    similarities = (vectors @ query) / (vector_norms * query_norm)
    return 1.0 - similarities


def brute_force_search(ids, vectors, query, k=5):
    """Return the k nearest chunk IDs using brute-force Python search."""
    if k < 1:
        raise ValueError("k must be at least 1")

    distances = cosine_distances(vectors, query)
    indices = np.argsort(distances)[:k]

    return [
        (int(ids[i]), float(distances[i]))
        for i in indices
    ]


def load_embeddings(conn):
    """Load all chunk embeddings from the existing MariaDB database."""
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT id, VEC_ToText(embedding) "
            "FROM chunks ORDER BY id"
        )
        rows = cursor.fetchall()

    if not rows:
        raise RuntimeError(
            "The chunks table is empty. Load the embeddings first."
        )

    ids = np.array([int(row[0]) for row in rows])

    vectors = np.asarray(
        [
            json.loads(
                row[1].decode("utf-8")
                if isinstance(row[1], bytes)
                else row[1]
            )
            for row in rows
        ],
        dtype=np.float32,
    )

    if vectors.shape != (len(ids), DIM):
        raise ValueError(
            f"Expected {DIM}-dimensional embeddings; "
            f"received shape {vectors.shape}"
        )

    return ids, vectors


def measure(function, repeats=10):
    """Return the final result and execution times in milliseconds."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    function()  # Warm-up, excluded from timing.

    timings = []
    result = None

    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        timings.append((time.perf_counter() - start) * 1000)

    return result, timings


def show_latency_chart(python_ms, exact_ms, indexed_ms):
    """Display a latency comparison chart (PNG export can be added later)."""

    methods = [
        "Python brute force",
        "MariaDB exact",
        "MariaDB indexed",
    ]
    latencies = [python_ms, exact_ms, indexed_ms]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(methods, latencies)

    ax.set_title("Median Search Latency Comparison")
    ax.set_ylabel("Latency (milliseconds)")
    ax.set_xlabel("Search method")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, latency in zip(bars, latencies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{latency:.3f} ms",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    plt.show()


def main(repeats=10, k=5, query_count=5):
    """Benchmark the current database contents without hardcoded dataset sizes."""
    if repeats < 1 or k < 1 or query_count < 1:
        raise ValueError("repeats, k and query_count must be positive")

    conn = get_connection()

    try:
        ids, vectors = load_embeddings(conn)

        # Adapt k and query count to however much data is available.
        k = min(k, len(ids))
        query_count = min(query_count, len(ids))

        # Choose evenly spaced real embeddings for reproducible queries.
        query_indices = np.linspace(
            0, len(ids) - 1, num=query_count, dtype=int
        )

        methods = {
            "Python brute force": lambda query: brute_force_search(
                ids, vectors, query, k
            ),
            "MariaDB exact": lambda query: [
                (row["chunk_id"], row["distance"])
                for row in search(conn, query, k=k, mode="exact")
            ],
            "MariaDB indexed": lambda query: [
                (row["chunk_id"], row["distance"])
                for row in search(conn, query, k=k, mode="index")
            ],
        }

        print(f"Corpus size: {len(ids):,} chunks")
        print(f"Embedding dimensions: {vectors.shape[1]}")
        print(f"Queries: {query_count} | Top-k: {k} | Repeats: {repeats}")
        print()

        all_times = {name: [] for name in methods}
        recalls = []

        for query_number, query_index in enumerate(query_indices, start=1):
            query = vectors[query_index].tolist()
            query_results = {}

            for name, function in methods.items():
                result, timings = measure(
                    lambda f=function, q=query: f(q),
                    repeats=repeats,
                )
                query_results[name] = result
                all_times[name].extend(timings)

            exact_ids = {
                chunk_id
                for chunk_id, _ in query_results["MariaDB exact"]
            }
            indexed_ids = {
                chunk_id
                for chunk_id, _ in query_results["MariaDB indexed"]
            }

            recall = (
                len(exact_ids & indexed_ids) / len(exact_ids)
                if exact_ids else 1.0
            )
            recalls.append(recall)

            print(
                f"Query {query_number}/{query_count}: "
                f"indexed recall@{k} = {recall:.1%}"
            )

        print("\n--- Overall Benchmark ---")
        medians = {}

        for name, timings in all_times.items():
            medians[name] = statistics.median(timings)
            print(
                f"{name:<24} "
                f"median={medians[name]:.3f} ms, "
                f"mean={statistics.mean(timings):.3f} ms"
            )

        python_time = medians["Python brute force"]

        for name in ("MariaDB exact", "MariaDB indexed"):
            db_time = medians[name]
            speedup = python_time / db_time if db_time > 0 else float("inf")
            print(f"Python / {name} speed ratio: {speedup:.2f}x")

        print(f"Average indexed recall@{k}: {statistics.mean(recalls):.1%}")
        print(
            "\nNote: Python vectors are preloaded in memory; MariaDB "
            "measurements include database query and network overhead."
        )
        show_latency_chart(
            medians["Python brute force"],
            medians["MariaDB exact"],
            medians["MariaDB indexed"],
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()
