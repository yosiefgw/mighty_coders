"""Shared database connection. Settings come from environment variables or the .env file (see .env.example)."""
import os
from pathlib import Path

import pymysql

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env_file(path):
    """Copy the DB_* and MARIADB_* lines of .env into os.environ.
    Variables that are already set in the environment win (so CI can set its own)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key.startswith(("DB_", "MARIADB_")):
            os.environ.setdefault(key, value.strip("\"'"))


def get_connection(autocommit=True):
    load_env_file(ENV_FILE)
    password = os.getenv("MARIADB_PASSWORD")
    if not password:
        raise RuntimeError("MARIADB_PASSWORD is not set. Copy .env.example to .env (README, Setup step 2).")
    return pymysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3308")),
        user=os.getenv("MARIADB_USER", "mcuser"),
        password=password,
        database=os.getenv("MARIADB_DATABASE", "mighty_coders"),
        autocommit=autocommit,
    )
