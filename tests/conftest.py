import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from db import get_connection  # noqa: E402


@pytest.fixture
def conn():
    """One connection per test, rolled back afterwards.
    Note: DDL commits implicitly, so tests that create tables must drop them themselves."""
    c = get_connection(autocommit=False)
    yield c
    c.rollback()
    c.close()
