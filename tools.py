"""The agent's tools that need no AI: listing files, saving, duplicate check, monthly summary.

Every tool returns a dict and never raises for bad input: {"ok": True, ...} or
{"ok": False, "error": "..."}. The model reads the error and decides what to do next
(fix the values, re-read the image, ask the user). Only StorageError (a damaged data
file) is raised, because then the agent must stop.
"""

import math
# Aliased: "date" is also a tool parameter name (the model sends it by that name)
from datetime import date as calendar_date, datetime

import config
import storage


def _error(message: str) -> dict:
    return {"ok": False, "error": message}


def _round_money(amount: float) -> float:
    return round(amount, 2)


# --- Validation helpers: return an error message, or None if the value is fine ---

def _check_date(value) -> str | None:
    if not isinstance(value, str):
        return "התאריך חייב להיות מחרוזת בפורמט YYYY-MM-DD"
    try:
        parsed = calendar_date.fromisoformat(value)
    except ValueError:
        return f"התאריך '{value}' לא תקין – נדרש פורמט YYYY-MM-DD"
    if parsed > calendar_date.today():
        return f"התאריך {value} עתידי – כנראה טעות בקריאה"
    return None


def _check_amount(value) -> str | None:
    # bool is a subclass of int in Python, so it is excluded explicitly
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return f"הסכום חייב להיות מספר (התקבל: {value!r})"
    if _round_money(value) == 0:
        return "הסכום לא יכול להיות 0"
    return None


def _categories_for(kind: str) -> list[str]:
    return config.EXPENSE_CATEGORIES if kind == "expense" else config.INCOME_CATEGORIES


# --- Tools ---

def list_documents() -> dict:
    """Images in inbox/, each marked whether it was already recorded (by file name)."""
    if not config.INBOX_DIR.is_dir():
        return _error(f"תיקיית הקלט {config.INBOX_DIR} לא קיימת")

    recorded = {t.get("source_file") for t in storage.load_transactions()}
    documents, unsupported = [], []
    for path in sorted(config.INBOX_DIR.iterdir()):
        if not path.is_file() or path.name.startswith("."):  # skip folders and .gitkeep
            continue
        if path.suffix.lower() in config.SUPPORTED_IMAGE_EXTENSIONS:
            documents.append({"file": path.name, "already_recorded": path.name in recorded})
        else:
            unsupported.append(path.name)

    result = {
        "ok": True,
        "documents": documents,
        "new_count": sum(not d["already_recorded"] for d in documents),
        "unsupported_files": unsupported,
    }
    if not documents:
        result["note"] = "אין תמונות בתיקיית inbox"
    return result


def add_transaction(type: str, date: str, amount: float, category: str, business: str,
                    source_file: str, items: list[str] | None = None) -> dict:
    """Validate and save one expense or income.

    A refund is an expense with a negative amount (not income). For a payslip, amount is the net pay.
    """
    if type not in ("expense", "income"):
        return _error(f"type חייב להיות 'expense' או 'income' (התקבל: {type!r})")
    if problem := _check_date(date):
        return _error(problem)
    if problem := _check_amount(amount):
        return _error(problem)
    if type == "income" and amount < 0:
        return _error("הכנסה חייבת להיות חיובית. זיכוי נרשם כהוצאה עם סכום שלילי")
    allowed = _categories_for(type)
    if category not in allowed:
        return _error(f"הקטגוריה '{category}' לא קיימת. אפשריות: {', '.join(allowed)}")
    if not isinstance(business, str) or not business.strip():
        return _error("חסר שם העסק / המעסיק")
    if not isinstance(source_file, str) or not source_file.strip():
        return _error("חסר שם קובץ המקור")
    if items is None:
        items = []
    if not isinstance(items, list) or not all(isinstance(i, str) for i in items):
        return _error("items חייב להיות רשימה של מחרוזות")

    transactions = storage.load_transactions()
    transaction = {
        "id": storage.next_transaction_id(transactions),
        "type": type,
        "date": date,
        "amount": _round_money(amount),
        "category": category,
        "business": business.strip(),
        "source_file": source_file.strip(),
        "items": items,
        "added_at": datetime.now().isoformat(timespec="seconds"),
    }
    transactions.append(transaction)
    storage.save_transactions(transactions)
    return {"ok": True, "transaction": transaction}


def find_duplicate(date: str, amount: float, source_file: str | None = None) -> dict:
    """Return already-recorded transactions that may be the same document.

    Candidates are matched by date + amount (business names read from images vary,
    e.g. "שופרסל" / "שופרסל דיל"), or by the same file name. The agent compares
    the businesses and decides whether it really is a duplicate.
    """
    if problem := _check_date(date):
        return _error(problem)
    if problem := _check_amount(amount):
        return _error(problem)

    candidates = []
    for t in storage.load_transactions():
        same_doc = t.get("date") == date and t.get("amount") == _round_money(amount)
        same_file = source_file is not None and t.get("source_file") == source_file
        if same_doc or same_file:
            candidates.append({
                "id": t.get("id"),
                "date": t.get("date"),
                "amount": t.get("amount"),
                "business": t.get("business"),
                "source_file": t.get("source_file"),
                "matched_by": "same_file" if same_file else "date_and_amount",
            })
    return {"ok": True, "candidates": candidates, "possible_duplicate": bool(candidates)}


def remember_business(business: str, category: str) -> dict:
    """Remember which category a business belongs to. Call only after the user confirmed it."""
    if not isinstance(business, str) or not business.strip():
        return _error("חסר שם העסק")
    if category not in config.EXPENSE_CATEGORIES + config.INCOME_CATEGORIES:
        return _error(f"הקטגוריה '{category}' לא קיימת")

    memory = storage.load_memory()
    memory["business_categories"][business.strip()] = category
    storage.save_memory(memory)
    return {"ok": True, "business": business.strip(), "category": category}


def get_monthly_summary(month: str) -> dict:
    """Income, expenses by category, balance and budget status for a month ("YYYY-MM")."""
    try:
        datetime.strptime(month, "%Y-%m")
    except (TypeError, ValueError):
        return _error(f"החודש '{month}' לא תקין – נדרש פורמט YYYY-MM")

    in_month = [t for t in storage.load_transactions() if str(t.get("date", "")).startswith(month + "-")]

    income_total = 0.0
    expenses_by_category: dict[str, float] = {}
    for t in in_month:
        if t.get("type") == "income":
            income_total += t["amount"]
        elif t.get("type") == "expense":
            expenses_by_category[t["category"]] = expenses_by_category.get(t["category"], 0.0) + t["amount"]
    expenses_by_category = {c: _round_money(a) for c, a in expenses_by_category.items()}
    expense_total = sum(expenses_by_category.values(), 0.0)

    budgets = []
    for category, budget in config.BUDGETS.items():
        spent = expenses_by_category.get(category, 0.0)
        ratio = spent / budget
        if ratio > 1:
            status = "over"
        elif ratio >= config.BUDGET_WARNING_RATIO:
            status = "near"
        else:
            status = "ok"
        budgets.append({
            "category": category,
            "budget": budget,
            "spent": spent,
            "percent": round(ratio * 100),
            "status": status,
        })

    result = {
        "ok": True,
        "month": month,
        "transaction_count": len(in_month),
        "income_total": _round_money(income_total),
        "expense_total": _round_money(expense_total),
        "balance": _round_money(income_total - expense_total),
        "expenses_by_category": expenses_by_category,
        "budgets": budgets,
        "currency": config.CURRENCY,
    }
    if not in_month:
        result["note"] = f"אין תנועות בחודש {month}"
    return result
