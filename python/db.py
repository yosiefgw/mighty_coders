"""Shared database connection. Settings come from environment variables (see .env.example)."""
import os
import pymysql


def get_connection(autocommit=True):
    return pymysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3308")),
        user=os.getenv("MARIADB_USER", "mcuser"),
        password=os.getenv("MARIADB_PASSWORD", "mcpassword"),
        database=os.getenv("MARIADB_DATABASE", "mighty_coders"),
        autocommit=autocommit,
    )
