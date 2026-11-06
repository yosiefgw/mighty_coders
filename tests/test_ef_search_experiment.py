"""
Read-only validation checks for the MariaDB mHNSW ef_search cache experiment.

Run from the repository root:
    python -m pytest tests/test_ef_search_experiment.py -v

Synthetic-data tests are skipped when the optional synth_vectors table does
not exist. The cache-size test is skipped unless the global cache is configured
to 1 GiB. These tests do not insert, update, delete, drop, or rebuild objects.
"""

import sys
from pathlib import Path

import pytest

# Make python/db.py importable when pytest is launched from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from db import get_connection  # noqa: E402


EXPECTED_SYNTHETIC_ROWS = 100_000
EXPECTED_CACHE_BYTES = 1_073_741_824
EXPECTED_VECTOR_INDEX = "idx_synth_embedding"


@pytest.fixture(scope="module")
def connection():
    conn = get_connection(autocommit=True)
    try:
        yield conn
    finally:
        conn.close()


def fetch_one(connection, sql, params=None):
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        return cursor.fetchone()


@pytest.fixture
def synthetic_table(connection):
    """Skip synthetic-only checks when the optional experiment table is absent."""
    row = fetch_one(
        connection,
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = DATABASE()
          AND table_name = 'synth_vectors'
        """,
    )
    if not row or row[0] == 0:
        pytest.skip(
            "Optional synth_vectors table is absent; "
            "run python/ef_search_experiment.py to create the experiment dataset"
        )


def test_database_version_is_mariadb_118(connection):
    row = fetch_one(connection, "SELECT VERSION()")
    version = row[0]
    assert "11.8" in version, f"Expected MariaDB 11.8.x, got {version}"


def test_synthetic_dataset_has_expected_row_count(connection, synthetic_table):
    row = fetch_one(connection, "SELECT COUNT(*) FROM synth_vectors")
    assert row[0] == EXPECTED_SYNTHETIC_ROWS


def test_synthetic_vector_index_exists_and_is_enabled(connection, synthetic_table):
    with connection.cursor() as cursor:
        cursor.execute("SHOW INDEX FROM synth_vectors")
        indexes = cursor.fetchall()

    matching = [
        row for row in indexes
        if row[2] == EXPECTED_VECTOR_INDEX
    ]
    assert matching, f"Missing vector index {EXPECTED_VECTOR_INDEX}"

    # SHOW INDEX columns: Table, Non_unique, Key_name, ..., Index_type, ..., Ignored
    assert matching[0][10] == "VECTOR", (
        f"Expected VECTOR index, got {matching[0][10]}"
    )
    assert matching[0][-1] == "NO", "Vector index is marked as ignored"


def test_global_hnsw_cache_is_one_gibibyte(connection):
    row = fetch_one(connection, "SELECT @@GLOBAL.mhnsw_max_cache_size")
    actual_cache_bytes = int(row[0])
    if actual_cache_bytes != EXPECTED_CACHE_BYTES:
        pytest.skip(
            "Requires mhnsw_max_cache_size to be 1 GiB; "
            f"current value is {actual_cache_bytes} bytes"
        )


def test_exact_query_plan_avoids_vector_index(connection, synthetic_table):
    sql = """
        EXPLAIN
        SELECT id
        FROM synth_vectors IGNORE INDEX (idx_synth_embedding)
        ORDER BY VEC_DISTANCE_COSINE(
            embedding,
            VEC_FromText(%s)
        )
        LIMIT 10
    """
    vector_text = "[" + ",".join(["0.0"] * 384) + "]"
    with connection.cursor() as cursor:
        cursor.execute(sql, (vector_text,))
        plan = cursor.fetchall()

    access_type = str(plan[0][3]).upper()
    assert access_type == "ALL", (
        f"Expected full table scan for exact query, got access type {access_type}"
    )


def test_indexed_query_plan_uses_vector_index(connection, synthetic_table):
    sql = """
        EXPLAIN
        SELECT id
        FROM synth_vectors
        ORDER BY VEC_DISTANCE_COSINE(
            embedding,
            VEC_FromText(%s)
        )
        LIMIT 10
    """
    vector_text = "[" + ",".join(["0.0"] * 384) + "]"
    with connection.cursor() as cursor:
        cursor.execute(sql, (vector_text,))
        plan = cursor.fetchall()

    key_name = plan[0][5]
    assert key_name == EXPECTED_VECTOR_INDEX, (
        f"Expected {EXPECTED_VECTOR_INDEX}, got {key_name}"
    )


def test_indexed_search_returns_ten_rows(connection, synthetic_table):
    row = fetch_one(
        connection,
        "SELECT VEC_ToText(embedding) FROM synth_vectors WHERE id = 1"
    )
    assert row and row[0], "Could not read a sample synthetic vector"

    with connection.cursor() as cursor:
        cursor.execute("SET SESSION mhnsw_ef_search = 400")
        cursor.execute(
            """
            SELECT id
            FROM synth_vectors
            ORDER BY VEC_DISTANCE_COSINE(
                embedding,
                VEC_FromText(%s)
            )
            LIMIT 10
            """,
            (row[0],),
        )
        results = cursor.fetchall()

    assert len(results) == 10
    assert len({item[0] for item in results}) == 10
