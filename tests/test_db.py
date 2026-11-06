"""Tests for the .env handling in python/db.py (no database needed)."""
import os

import pytest

import db


def test_load_env_file_reads_db_settings_only(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nDB_PORT=3999\nMARIADB_PASSWORD='secret'\nANTHROPIC_API_KEY=xyz\n")
    for key in ("DB_PORT", "MARIADB_PASSWORD", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    db.load_env_file(env)
    assert os.environ["DB_PORT"] == "3999"
    assert os.environ["MARIADB_PASSWORD"] == "secret"     # quotes removed
    assert "ANTHROPIC_API_KEY" not in os.environ          # only DB_* and MARIADB_* are read


def test_environment_wins_over_env_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("MARIADB_PASSWORD=from_file\n")
    monkeypatch.setenv("MARIADB_PASSWORD", "from_env")
    db.load_env_file(env)
    assert os.environ["MARIADB_PASSWORD"] == "from_env"


def test_missing_password_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("MARIADB_PASSWORD", raising=False)
    monkeypatch.setattr(db, "ENV_FILE", tmp_path / "missing.env")
    with pytest.raises(RuntimeError, match="MARIADB_PASSWORD is not set"):
        db.get_connection()
