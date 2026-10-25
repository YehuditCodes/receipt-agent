"""Reading and writing the agent's JSON files.

transactions.json - what was spent / earned (list of transactions).
memory.json       - what the agent learned (e.g. business -> category).
They are separate so a month can be reset without the agent forgetting what it learned.
"""

import json
import os
import re
from pathlib import Path

import config


class StorageError(Exception):
    """A data file is missing, unreadable or damaged. The message is shown to the user."""


def _read_json(path: Path, default, expected_type: type):
    """Return the file's content, or `default` if the file does not exist yet."""
    if not path.exists():
        return default
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        # Never overwrite a damaged file - that could lose real data.
        raise StorageError(
            f"הקובץ {path.name} פגום ולא ניתן לקרוא אותו (שורה {e.lineno}). "
            f"יש לתקן או למחוק אותו ולהריץ שוב. הקובץ לא שונה."
        ) from e
    except OSError as e:
        raise StorageError(f"לא ניתן לפתוח את הקובץ {path.name}: {e}") from e

    if not isinstance(data, expected_type):
        raise StorageError(
            f"הקובץ {path.name} במבנה לא צפוי. יש לתקן או למחוק אותו ולהריץ שוב. הקובץ לא שונה."
        )
    return data


def _write_json(path: Path, data) -> None:
    """Write safely: to a temp file first, then replace, so a crash mid-write can't corrupt the file."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except OSError as e:
        raise StorageError(f"לא ניתן לשמור את הקובץ {path.name}: {e}") from e


# --- Transactions ---

def load_transactions() -> list[dict]:
    return _read_json(config.TRANSACTIONS_FILE, default=[], expected_type=list)


def save_transactions(transactions: list[dict]) -> None:
    _write_json(config.TRANSACTIONS_FILE, transactions)


def next_transaction_id(transactions: list[dict]) -> str:
    """Running id: t001, t002, ... (one past the highest existing id)."""
    numbers = [
        int(m.group(1))
        for t in transactions
        if (m := re.fullmatch(r"t(\d+)", str(t.get("id", ""))))
    ]
    return f"t{max(numbers, default=0) + 1:03d}"


# --- Memory ---

def load_memory() -> dict:
    memory = _read_json(config.MEMORY_FILE, default={}, expected_type=dict)
    memory.setdefault("business_categories", {})
    return memory


def save_memory(memory: dict) -> None:
    _write_json(config.MEMORY_FILE, memory)
