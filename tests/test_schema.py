"""Tests for sql/init/01_vector_schema.sql (MariaDB 11.8).

Run from a clean start:  docker compose down -v && docker compose up -d   (wait ~30 s), then  pytest -q
Test rows use very high ids (near the INT UNSIGNED maximum), so they never clash with real Wikipedia page ids.
"""
import pymysql
import pytest

from db import get_connection

DIM = 384
TEST_ID = 4_294_960_000  # high test ids: TEST_ID, TEST_ID + 1, ...


def vec(k, dim=DIM):
    """A fixed test vector with `dim` numbers; different k gives a different direction."""
    return "[" + ",".join(str(((i * k) % 11 + 1) / 10) for i in range(1, dim + 1)) + "]"


def add_article(cur, article_id):
    cur.execute(
        "INSERT INTO articles (id, revision_id, title, url, word_count, last_edited) "
        "VALUES (%s, 1, %s, %s, 100, '2025-01-01')",
        (article_id, f"Test {article_id}", f"https://example.org/{article_id}"),
    )


def add_chunk(cur, article_id, vector, chunk_index=0):
    cur.execute(
        "INSERT INTO chunks (article_id, chunk_index, word_count, chunk_text, embedding) "
        "VALUES (%s, %s, 2, 'test chunk', VEC_FromText(%s))",
        (article_id, chunk_index, vector),
    )


def search(cur, vector, k=5):
    """Article ids of the k nearest chunks, using the vector index (same pattern as the real queries)."""
    cur.execute(
        "SELECT article_id FROM chunks "
        "ORDER BY VEC_DISTANCE_COSINE(embedding, VEC_FromText(%s)) LIMIT %s",
        (vector, k),
    )
    return [row[0] for row in cur.fetchall()]


def error_code(exc_info):
    return exc_info.value.args[0]


# ---------- the schema itself ----------

def test_all_tables_exist(conn):
    cur = conn.cursor()
    cur.execute("SHOW TABLES")
    tables = {row[0] for row in cur.fetchall()}
    assert {"articles", "categories", "article_categories", "links", "chunks"} <= tables


def test_vector_index_definition(conn):
    cur = conn.cursor()
    cur.execute("SHOW CREATE TABLE chunks")
    ddl = cur.fetchone()[1]
    assert "`embedding` vector(384) NOT NULL" in ddl
    assert "VECTOR KEY `idx_chunks_embedding` (`embedding`) `M`='8' `DISTANCE`='cosine'" in ddl


def test_chunks_foreign_key_has_no_cascade(conn):
    """On purpose: a cascade delete breaks the vector index (see test_known_bug_cascade_breaks_vector_index)."""
    cur = conn.cursor()
    cur.execute("SHOW CREATE TABLE chunks")
    fk_line = next(line for line in cur.fetchone()[1].splitlines() if "fk_chunks_article" in line)
    assert "CASCADE" not in fk_line


# ---------- behaviour of the vector index ----------

def test_explain_uses_vector_index(conn):
    cur = conn.cursor()
    add_article(cur, TEST_ID)
    add_chunk(cur, TEST_ID, vec(1))
    cur.execute(
        "EXPLAIN SELECT id FROM chunks ORDER BY VEC_DISTANCE_COSINE(embedding, VEC_FromText(%s)) LIMIT 5",
        (vec(1),),
    )
    columns = [d[0] for d in cur.description]
    row = dict(zip(columns, cur.fetchone()))
    assert row["key"] == "idx_chunks_embedding"


def test_nearest_chunk_is_the_same_vector(conn):
    cur = conn.cursor()
    add_article(cur, TEST_ID)
    add_article(cur, TEST_ID + 1)
    add_chunk(cur, TEST_ID, vec(1))
    add_chunk(cur, TEST_ID + 1, vec(2))
    assert search(cur, vec(1))[0] == TEST_ID


def test_delete_article_with_chunks_is_refused(conn):
    cur = conn.cursor()
    add_article(cur, TEST_ID)
    add_chunk(cur, TEST_ID, vec(1))
    with pytest.raises(pymysql.MySQLError) as exc:
        cur.execute("DELETE FROM articles WHERE id = %s", (TEST_ID,))
    assert error_code(exc) == 1451  # foreign key: delete the chunks first


def test_delete_chunks_then_article_keeps_search_correct(conn):
    cur = conn.cursor()
    add_article(cur, TEST_ID)
    add_article(cur, TEST_ID + 1)
    add_chunk(cur, TEST_ID, vec(1))
    add_chunk(cur, TEST_ID + 1, vec(2))
    cur.execute("DELETE FROM chunks WHERE article_id = %s", (TEST_ID,))
    cur.execute("DELETE FROM articles WHERE id = %s", (TEST_ID,))
    assert search(cur, vec(1)) == [TEST_ID + 1]


def test_rollback_removes_chunk_from_search(conn):
    cur = conn.cursor()
    add_article(cur, TEST_ID)
    add_chunk(cur, TEST_ID, vec(1))
    assert TEST_ID in search(cur, vec(1))      # visible inside the transaction
    conn.rollback()
    assert TEST_ID not in search(cur, vec(1))  # gone after rollback


@pytest.mark.parametrize("title_a, title_b", [("Titan", "TITAN"), ("Mone", "Moné")])
def test_titles_differing_only_in_case_or_accent_are_allowed(conn, title_a, title_b):
    """Wikipedia titles are exact: 'Titan' and 'TITAN' are two pages. The title column uses
    COLLATE utf8mb4_bin; with the default utf8mb4_unicode_ci this insert failed (error 1062)."""
    cur = conn.cursor()
    for offset, title in enumerate([title_a, title_b]):
        cur.execute(
            "INSERT INTO articles (id, revision_id, title, url, word_count, last_edited) "
            "VALUES (%s, 1, %s, 'x', 1, '2025-01-01')",
            (TEST_ID + offset, title),
        )
    cur.execute("INSERT INTO categories (name) VALUES (%s), (%s)", (title_a, title_b))


def test_wrong_vector_size_is_rejected(conn):
    cur = conn.cursor()
    add_article(cur, TEST_ID)
    with pytest.raises(pymysql.MySQLError) as exc:
        add_chunk(cur, TEST_ID, "[0.1,0.2,0.3]")
    assert error_code(exc) == 1292


# ---------- rules from the MariaDB docs ----------

@pytest.mark.parametrize(
    "name, ddl, expected_error",
    [
        ("two_indexes", "(a VECTOR(3) NOT NULL, b VECTOR(3) NOT NULL, VECTOR INDEX (a), VECTOR INDEX (b))", 1235),
        ("nullable", "(a VECTOR(3), VECTOR INDEX (a))", 1252),
        ("m_2", "(a VECTOR(3) NOT NULL, VECTOR INDEX (a) M=2)", 1912),
        ("m_201", "(a VECTOR(3) NOT NULL, VECTOR INDEX (a) M=201)", 1912),
        ("m_3", "(a VECTOR(3) NOT NULL, VECTOR INDEX (a) M=3)", None),
        ("m_200", "(a VECTOR(3) NOT NULL, VECTOR INDEX (a) M=200)", None),
    ],
)
def test_docs_rules_for_vector_index(name, ddl, expected_error):
    table = f"t_docs_rule_{name}"
    c = get_connection(autocommit=True)
    cur = c.cursor()
    try:
        if expected_error is None:
            cur.execute(f"CREATE TABLE {table} {ddl}")
        else:
            with pytest.raises(pymysql.MySQLError) as exc:
                cur.execute(f"CREATE TABLE {table} {ddl}")
            assert error_code(exc) == expected_error
    finally:
        cur.execute(f"DROP TABLE IF EXISTS {table}")
        c.close()


# ---------- known MariaDB problem ----------

@pytest.mark.parametrize(
    "distance, distance_fn",
    [("cosine", "VEC_DISTANCE_COSINE"), ("euclidean", "VEC_DISTANCE_EUCLIDEAN")],
)
def test_known_bug_cascade_breaks_vector_index(distance, distance_fn):
    """Rows deleted by ON DELETE CASCADE break the vector index (seen on 11.8.9, cosine and euclidean).
    A direct DELETE works. If this test FAILS, MariaDB may have fixed the bug: check, then update the schema."""
    c = get_connection(autocommit=True)
    cur = c.cursor()
    tables = ["x_cos_cascade", "x_cos_direct", "x_parent"]
    try:
        cur.execute("DROP TABLE IF EXISTS " + ", ".join(tables))
        cur.execute("CREATE TABLE x_parent (id INT UNSIGNED PRIMARY KEY) ENGINE=InnoDB")
        for table, on_delete in [("x_cos_cascade", "ON DELETE CASCADE"), ("x_cos_direct", "")]:
            cur.execute(
                f"CREATE TABLE {table} (id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY, "
                f"parent_id INT UNSIGNED NOT NULL, v VECTOR({DIM}) NOT NULL, "
                f"VECTOR INDEX vi (v) M=8 DISTANCE={distance}, "
                f"FOREIGN KEY (parent_id) REFERENCES x_parent (id) {on_delete}) ENGINE=InnoDB"
            )
        for k in range(1, 21):
            cur.execute("INSERT INTO x_parent VALUES (%s)", (k,))
            for table in ["x_cos_cascade", "x_cos_direct"]:
                cur.execute(f"INSERT INTO {table} (parent_id, v) VALUES (%s, VEC_FromText(%s))", (k, vec(k)))

        cur.execute("DELETE FROM x_cos_direct WHERE parent_id <= 5")  # direct delete
        cur.execute("DELETE FROM x_parent WHERE id <= 5")             # cascade for x_cos_cascade

        def found_via_index(table):
            cur.execute(
                f"SELECT COUNT(*) FROM (SELECT id FROM {table} "
                f"ORDER BY {distance_fn}(v, VEC_FromText(%s)) LIMIT 20) s",
                (vec(3),),
            )
            return cur.fetchone()[0]

        assert found_via_index("x_cos_direct") >= 14       # correct (approximate index may miss one)
        assert found_via_index("x_cos_cascade") < 14, (
            "The cascade no longer breaks the vector index: MariaDB may have fixed the bug."
        )
    finally:
        cur.execute("DROP TABLE IF EXISTS " + ", ".join(tables))
        c.close()
