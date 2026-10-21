"""Smoke test: the database is reachable and is new enough for the VECTOR type (11.7+)."""


def test_server_version_supports_vector(conn):
    cur = conn.cursor()
    cur.execute("SELECT VERSION()")
    version = cur.fetchone()[0]
    major, minor = (int(x) for x in version.split("-")[0].split(".")[:2])
    assert (major, minor) >= (11, 7), f"VECTOR needs MariaDB 11.7+, got {version}"
