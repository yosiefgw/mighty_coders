"""
SYNTHETIC HNSW ef_search experiment for MariaDB 11.8.

Creates an independent synthetic vector table. Does not modify chunks.
All generated data and results are SYNTHETIC.
"""

import csv
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from db import get_connection


# ---------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------

SEED = 42
NUM_VECTORS = 100_000
NUM_QUERIES = 100
DIMENSIONS = 384
TOP_K = 10

EF_SEARCH_VALUES = [10, 20, 50, 100, 200, 400]

BATCH_SIZE = 500
REPEATS = 1

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"

TABLE_NAME = "synth_vectors"


# ---------------------------------------------------------
# Database helpers
# ---------------------------------------------------------

def execute(connection, sql, params=None):
    with connection.cursor() as cursor:
        cursor.execute(sql, params)


def fetch_all(connection, sql, params=None):
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        return cursor.fetchall()


def vector_to_sql(vector):
    """Convert a NumPy vector into MariaDB vector text format."""
    return "[" + ",".join(format(float(x), ".9g") for x in vector) + "]"


# ---------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------

def generate_vectors():
    """Generate reproducible normalized float32 vectors and queries."""
    rng = np.random.default_rng(SEED)

    vectors = rng.standard_normal(
        (NUM_VECTORS, DIMENSIONS)
    ).astype(np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= norms

    queries = rng.standard_normal(
        (NUM_QUERIES, DIMENSIONS)
    ).astype(np.float32)

    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries /= query_norms

    return vectors, queries


# ---------------------------------------------------------
# Synthetic table creation and insertion
# ---------------------------------------------------------

def create_synthetic_table(connection):
    """
    Create the independent synthetic table if it does not exist.

    Existing complete datasets are reused. A partial non-empty table
    is not truncated or modified; the script stops instead.
    """
    print("\nChecking SYNTHETIC table...")

    execute(connection, f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INT UNSIGNED NOT NULL AUTO_INCREMENT,
            embedding VECTOR({DIMENSIONS}) NOT NULL,
            PRIMARY KEY (id),
            VECTOR KEY idx_synth_embedding (embedding)
                M=8 DISTANCE=cosine
        ) ENGINE=InnoDB
    """)

    count = fetch_all(
        connection,
        f"SELECT COUNT(*) FROM {TABLE_NAME}"
    )[0][0]

    if count == NUM_VECTORS:
        print(
            f"Found {count:,} existing SYNTHETIC vectors. "
            "Reusing dataset; skipping insertion."
        )
        return False

    if count > 0:
        raise RuntimeError(
            f"Found {count:,} vectors, expected either 0 or "
            f"{NUM_VECTORS:,}. Refusing to truncate existing data."
        )

    print("Synthetic table is empty and ready for insertion.")
    return True


def insert_vectors(connection, vectors):
    """Insert generated vectors in batches into synth_vectors."""
    print(f"\nInserting {NUM_VECTORS:,} SYNTHETIC vectors...")

    insert_sql = f"""
        INSERT INTO {TABLE_NAME} (embedding)
        VALUES (VEC_FromText(%s))
    """

    start = time.perf_counter()

    with connection.cursor() as cursor:
        for start_index in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[start_index:start_index + BATCH_SIZE]

            rows = [
                (vector_to_sql(vector),)
                for vector in batch
            ]

            cursor.executemany(insert_sql, rows)

            inserted = min(start_index + len(batch), len(vectors))
            print(
                f"\rInserted {inserted:,}/{NUM_VECTORS:,}",
                end=""
            )

    print()
    print(f"Insertion time: {time.perf_counter() - start:.2f} seconds")


# ---------------------------------------------------------
# Query execution
# ---------------------------------------------------------

def exact_search(connection, query_vector):
    """
    Full scan: IGNORE INDEX prevents use of the HNSW vector index.
    Returns exact top-K nearest IDs.
    """
    sql = f"""
        SELECT id
        FROM {TABLE_NAME} IGNORE INDEX (idx_synth_embedding)
        ORDER BY VEC_DISTANCE_COSINE(
            embedding, VEC_FromText(%s)
        )
        LIMIT {TOP_K}
    """

    rows = fetch_all(connection, sql, (vector_to_sql(query_vector),))
    return [row[0] for row in rows]


def set_ef_search(connection, ef_search):
    """Set and verify ef_search once before timing queries for this value."""
    with connection.cursor() as cursor:
        cursor.execute(
            "SET SESSION mhnsw_ef_search = %s",
            (int(ef_search),)
        )
        cursor.execute("SELECT @@session.mhnsw_ef_search")
        actual_value = cursor.fetchone()[0]

    if actual_value != ef_search:
        raise RuntimeError(
            f"Requested ef_search={ef_search}, "
            f"but MariaDB reports {actual_value}"
        )


def indexed_search(connection, query_vector):
    """
    Search using the vector index.

    Returns integer IDs only, so they can be compared directly with
    the integer IDs returned by exact_search().
    """
    sql = f"""
        SELECT id
        FROM {TABLE_NAME}
        ORDER BY VEC_DISTANCE_COSINE(
            embedding, VEC_FromText(%s)
        )
        LIMIT {TOP_K}
    """

    rows = fetch_all(connection, sql, (vector_to_sql(query_vector),))
    return [row[0] for row in rows]


# ---------------------------------------------------------
# Verify execution plans
# ---------------------------------------------------------

def verify_execution_plans(connection, query_vector):
    print("\nChecking execution plans...")

    vector_text = vector_to_sql(query_vector)

    exact_plan = fetch_all(
        connection,
        f"""
        EXPLAIN
        SELECT id
        FROM {TABLE_NAME} IGNORE INDEX (idx_synth_embedding)
        ORDER BY VEC_DISTANCE_COSINE(
            embedding, VEC_FromText(%s)
        )
        LIMIT {TOP_K}
        """,
        (vector_text,),
    )

    indexed_plan = fetch_all(
        connection,
        f"""
        EXPLAIN
        SELECT id
        FROM {TABLE_NAME}
        ORDER BY VEC_DISTANCE_COSINE(
            embedding, VEC_FromText(%s)
        )
        LIMIT {TOP_K}
        """,
        (vector_text,),
    )

    print("\nExact search plan:")
    for row in exact_plan:
        print(row)

    print("\nIndexed search plan:")
    for row in indexed_plan:
        print(row)

    print(
        "\nIMPORTANT: Inspect the plans to confirm the exact query "
        "does not use idx_synth_embedding and the indexed query does."
    )


# ---------------------------------------------------------
# Benchmark
# ---------------------------------------------------------

def run_experiment(connection, queries):
    results = []

    print("\nCalculating exact reference results...")

    # Calculate exact top-10 only once per query.
    exact_results = []

    for query_number, query_vector in enumerate(queries, start=1):
        exact_ids = exact_search(connection, query_vector)
        exact_results.append(set(exact_ids))

        if query_number % 10 == 0:
            print(f"  Exact queries: {query_number}/{NUM_QUERIES}")

    print("\nStarting SYNTHETIC benchmark...")

    for ef_search in EF_SEARCH_VALUES:
        print(f"\nTesting ef_search = {ef_search}")

        # Keep session-setting overhead outside the timed SELECT.
        set_ef_search(connection, ef_search)

        recalls = []
        latencies_ms = []

        for query_number, query_vector in enumerate(queries):
            exact_set = exact_results[query_number]

            # Warm-up indexed search (not timed).
            indexed_search(connection, query_vector)

            for _ in range(REPEATS):
                start = time.perf_counter()
                indexed_ids = indexed_search(connection, query_vector)
                elapsed_ms = (time.perf_counter() - start) * 1000

                latencies_ms.append(elapsed_ms)

                indexed_set = set(indexed_ids)
                recall = len(exact_set.intersection(indexed_set)) / TOP_K
                recalls.append(recall)

            if (query_number + 1) % 10 == 0:
                print(
                    f"  Completed {query_number + 1}/{NUM_QUERIES} queries"
                )

        median_latency = statistics.median(latencies_ms)
        mean_recall = statistics.mean(recalls) * 100

        result = {
            "data_type": "SYNTHETIC",
            "ef_search": ef_search,
            "recall_at_10_percent": round(mean_recall, 2),
            "median_latency_ms": round(median_latency, 4),
            "num_vectors": NUM_VECTORS,
            "num_queries": NUM_QUERIES,
            "dimensions": DIMENSIONS,
        }

        results.append(result)

        print(
            f"  Recall@10: {mean_recall:.2f}% | "
            f"Median latency: {median_latency:.4f} ms"
        )

    return results


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

def save_results(results):
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DOCS_DIR / "ef_search_results.csv"
    fieldnames = list(results[0].keys())

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nCSV saved: {csv_path}")


def create_chart(results):
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    ef_values = [r["ef_search"] for r in results]
    recalls = [r["recall_at_10_percent"] for r in results]
    latencies = [r["median_latency_ms"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(
        ef_values,
        recalls,
        marker="o",
        label="Recall@10 (%)"
    )
    ax1.set_xlabel("mhnsw_ef_search")
    ax1.set_ylabel("Recall@10 (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title("SYNTHETIC DATA — HNSW Search Trade-off")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(
        ef_values,
        latencies,
        marker="s",
        linestyle="--",
        label="Median latency (ms)"
    )
    ax2.set_ylabel("Median latency (ms)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.tight_layout()

    chart_path = DOCS_DIR / "ef_search_tradeoff.png"
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)

    print(f"Chart saved: {chart_path}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print("=" * 65)
    print("SYNTHETIC MARIADB HNSW ef_search EXPERIMENT")
    print("=" * 65)

    print(f"Vectors: {NUM_VECTORS:,}")
    print(f"Queries: {NUM_QUERIES}")
    print(f"Dimensions: {DIMENSIONS}")
    print(f"Top-K: {TOP_K}")
    print(f"ef_search values: {EF_SEARCH_VALUES}")
    print("Existing chunks table will NOT be modified.")

    vectors, queries = generate_vectors()

    connection = get_connection(autocommit=True)

    try:
        needs_insertion = create_synthetic_table(connection)

        if needs_insertion:
            insert_vectors(connection, vectors)
        else:
            print("Using previously generated SYNTHETIC dataset.")

        count = fetch_all(
            connection,
            f"SELECT COUNT(*) FROM {TABLE_NAME}"
        )[0][0]

        print(f"\nVerified synthetic row count: {count:,}")

        if count != NUM_VECTORS:
            raise RuntimeError(
                f"Expected {NUM_VECTORS} rows but found {count}"
            )

        verify_execution_plans(connection, queries[0])

        results = run_experiment(connection, queries)

        save_results(results)
        create_chart(results)

        print("\n" + "=" * 65)
        print("SYNTHETIC EXPERIMENT COMPLETE")
        print("=" * 65)

        for result in results:
            print(
                f"ef_search={result['ef_search']:>3} | "
                f"Recall@10={result['recall_at_10_percent']:>6.2f}% | "
                f"Median={result['median_latency_ms']:>9.4f} ms"
            )

    finally:
        connection.close()


if __name__ == "__main__":
    main()
