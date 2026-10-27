"""Shared setup: every test works on an empty temporary inbox/ and data/, never on the real ones.

No test calls the OpenAI API or the internet - these check the Python side only.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


@pytest.fixture(autouse=True)
def temp_data(tmp_path, monkeypatch):
    inbox, data = tmp_path / "inbox", tmp_path / "data"
    inbox.mkdir()
    data.mkdir()
    monkeypatch.setattr(config, "INBOX_DIR", inbox)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "TRANSACTIONS_FILE", data / "transactions.json")
    monkeypatch.setattr(config, "MEMORY_FILE", data / "memory.json")
    monkeypatch.setattr(config, "SKIPPED_FILE", data / "skipped.json")
    monkeypatch.setattr(config, "READ_CACHE_FILE", data / "read_cache.json")
    monkeypatch.setattr(config, "USE_ONLINE_LOOKUPS", False)
    return inbox


@pytest.fixture
def add_file(temp_data):
    """Put a small file in the temporary inbox/ (its content decides whether it is a copy)."""
    def add(name: str, content: bytes = b"receipt") -> str:
        (temp_data / name).write_bytes(content)
        return name
    return add
