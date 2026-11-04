import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from db import get_connection  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_torch: mark test as requiring a working torch/sentence-transformers installation",
    )
    """Stub torch/sentence_transformers early if they cannot be loaded.

    This prevents collection-time ImportErrors in modules that import embed.py.
    When the stub is active, tests that require real embeddings are skipped via
    the ``requires_torch`` marker (see pytest_collection_modifyitems below).
    """
    try:
        import torch  # noqa: F401
        config._torch_available = True
    except (ImportError, OSError):
        from unittest.mock import MagicMock
        for mod in (
            "torch",
            "sentence_transformers",
            "sentence_transformers.SentenceTransformer",
        ):
            sys.modules.setdefault(mod, MagicMock())
        config._torch_available = False


def pytest_collection_modifyitems(session, config, items):
    """Skip tests marked ``requires_torch`` when torch is not properly installed."""
    if getattr(config, "_torch_available", True):
        return
    skip = pytest.mark.skip(reason="torch not available in this environment")
    for item in items:
        if item.get_closest_marker("requires_torch"):
            item.add_marker(skip)


@pytest.fixture
def conn():
    """One connection per test, rolled back afterwards.
    Note: DDL commits implicitly, so tests that create tables must drop them themselves."""
    c = get_connection(autocommit=False)
    yield c
    c.rollback()
    c.close()
