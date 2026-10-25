"""The agent's tools: listing files, reading an image, saving, duplicate check, monthly summary.

Only read_document uses AI (it sends the image to the model); the rest is plain Python.

Every tool returns a dict and never raises for bad input: {"ok": True, ...} or
{"ok": False, "error": "..."}. The model reads the error and decides what to do next
(fix the values, re-read the image, ask the user). Only StorageError (a damaged data
file) is raised, because then the agent must stop.
"""

import base64
import functools
import json
import math
import re
# Aliased: "date" is also a tool parameter name (the model sends it by that name)
from datetime import date as calendar_date, datetime

import openai

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


# --- read_document: the image-reading prompt and the exact JSON shape the model must return ---

READ_PROMPT = """You read ONE image of an Israeli financial document and extract its data.
Report only what is printed. Never guess: if a field cannot be read, use null.

document_type:
- "receipt": a purchase at a store or restaurant
- "bill": a utility / municipal / phone / internet bill (חשבון)
- "payslip": a salary slip (תלוש שכר)
- "refund": a credit note or returned goods (זיכוי)
- "other": not a financial document

readable is about image quality only:
- false if the image is too blurry, dark, small or cut off to read every digit of the total with
  certainty. Then set unreadable_reason (short, in Hebrew). Never estimate or guess a number from
  a blurry image - a wrong amount is worse than no amount.
- true for a clear image even if it is not a financial document (then document_type is "other").

business: who issued the document (store, company, employer), as printed, in its original language,
without labels or punctuation around it. Not the customer or the employee.

date_printed: the document's date copied exactly as printed (e.g. "03/03/10"), without the time.
Do not convert or reorder it.
- receipt / refund: the purchase date
- bill: the issue date (תאריך הפקה / תאריך עריכה), not the billing period or the due date
- payslip: the issue date (הודפס בתאריך), not the salary month

billing_period_end_printed: bills only - the last day of the billing period (תקופת החשבון), exactly as
printed. null for other documents.

total: the amount actually paid or received, as a positive number.
- receipt: לתשלום / סה"כ - not the cash handed over (מזומן) or the change (עודף)
- bill: סה"כ לתשלום
- payslip: the final לתשלום line (after all deductions), not gross pay; use שכר נטו only if there is no לתשלום line
- refund: the refunded amount, still positive
Lines such as "you saved" (חסכת / הנחה), VAT (מע"מ) or points are not the total.

payments: how the total was paid, as printed in the payment section - one entry per payment line
(credit card, cash, voucher), each with its amount. A purchase split between several cards has
several entries. Empty if no payment lines are printed. change: the change given back (עודף), or null.

currency: ISO code; ₪ / ש"ח / NIS = "ILS".

items: each purchased line with its price, for receipts and refunds only (empty for bills and payslips).
Copy the names as printed; leave out a line rather than invent it. price is null if it cannot be read.

notes: anything unusual worth knowing, in Hebrew, or null."""

_NULLABLE_STRING = {"type": ["string", "null"]}
_NULLABLE_NUMBER = {"type": ["number", "null"]}

# Structured output: the API guarantees the reply matches this schema, so it is always valid JSON.
READ_SCHEMA = {
    "type": "object",
    "properties": {
        "readable": {"type": "boolean"},
        "unreadable_reason": _NULLABLE_STRING,
        "document_type": {"type": "string", "enum": ["receipt", "bill", "payslip", "refund", "other"]},
        "business": _NULLABLE_STRING,
        "date_printed": _NULLABLE_STRING,
        "billing_period_end_printed": _NULLABLE_STRING,
        "total": _NULLABLE_NUMBER,
        "payments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"method": {"type": "string"}, "amount": {"type": "number"}},
                "required": ["method", "amount"],
                "additionalProperties": False,
            },
        },
        "change": _NULLABLE_NUMBER,
        "currency": _NULLABLE_STRING,
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "price": _NULLABLE_NUMBER},
                "required": ["name", "price"],
                "additionalProperties": False,
            },
        },
        "notes": _NULLABLE_STRING,
    },
    "required": ["readable", "unreadable_reason", "document_type", "business",
                 "date_printed", "billing_period_end_printed", "total", "payments", "change", "currency", "items", "notes"],
    "additionalProperties": False,
}

_MIME_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


@functools.cache
def _openai_client() -> openai.OpenAI:
    # Created on first use; reads OPENAI_API_KEY from the environment (raises OpenAIError if missing)
    return openai.OpenAI(timeout=config.READ_TIMEOUT_SECONDS)


def _ask_model_to_read(path) -> dict:
    """Send the image to the model and return its parsed reply. Raises on API problems."""
    image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    image_url = f"data:{_MIME_TYPES[path.suffix.lower()]};base64,{image_b64}"
    response = _openai_client().chat.completions.create(
        model=config.READ_MODEL,  # no temperature: gpt-5 models accept only the default
        reasoning_effort=config.READ_REASONING_EFFORT,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "document", "schema": READ_SCHEMA, "strict": True},
        },
        messages=[
            {"role": "system", "content": READ_PROMPT},
            {"role": "user", "content": [
                # Always high detail: in tests, low detail invented items that are not on the receipt
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
            ]},
        ],
    )
    message = response.choices[0].message
    if message.refusal:
        raise ValueError(f"המודל סירב לקרוא את התמונה: {message.refusal}")
    if response.choices[0].finish_reason != "stop":
        raise ValueError("התשובה של המודל נקטעה")
    return json.loads(message.content)


def _parse_printed_date(printed: str | None) -> str | None:
    """Israeli day-first date as printed ("03/03/10", "27.02.19", "10/07/2020") -> "YYYY-MM-DD".

    Done in Python, not by the model: in tests the model converted dates inconsistently.
    """
    match = re.search(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4}|\d{2})\b", printed or "")
    if not match:
        return None
    day, month, year = (int(g) for g in match.groups())
    if year < 100:
        year += 2000  # receipts are from this century
    try:
        return calendar_date(year, month, day).isoformat()
    except ValueError:  # e.g. 31/02
        return None


def _check_reading(reading: dict) -> tuple[list[str], list[str]]:
    """Checks done in Python on what the model read. Adds computed fields.

    Returns (warnings, info): a warning means "do not record blindly - ask the user";
    info is worth knowing but does not block recording.
    """
    warnings, info = [], []
    # Names sometimes come back split over lines ("מגה בול\nבול סגולה")
    reading["business"] = " ".join((reading["business"] or "").split()).strip(" :;,.-") or None
    if reading["document_type"] == "other":
        warnings.append("זה לא נראה כמו קבלה, חשבון או תלוש – לא לרשום בלי לשאול את המשתמשת")
    if reading["total"] is None:
        warnings.append("לא נמצא סכום כולל")

    reading["date"] = _parse_printed_date(reading["date_printed"])
    if reading["date_printed"] is None:
        warnings.append("לא נמצא תאריך על המסמך")
    elif reading["date"] is None:
        warnings.append(f"לא הצלחתי להבין את התאריך '{reading['date_printed']}' – לשאול את המשתמשת")
    elif problem := _check_date(reading["date"]):
        warnings.append(f"{problem} (מודפס: {reading['date_printed']})")

    # Cross-check: a bill is issued after its billing period ends. In tests the model
    # sometimes returned the period end instead of the issue date.
    period_end = _parse_printed_date(reading.pop("billing_period_end_printed"))
    if reading["date"] and period_end and reading["date"] <= period_end:
        warnings.append(
            f"תאריך ההפקה ({reading['date']}) לא אחרי סוף תקופת החיוב ({period_end}) – "
            "כנראה נקרא תאריך לא נכון"
        )

    # Cross-check: the total must equal what was paid. In tests the model once took the
    # "you saved" line as the total, while the card line showed the real amount.
    if reading["payments"] and reading["total"] is not None:
        paid = _round_money(sum(p["amount"] for p in reading["payments"]) - (reading["change"] or 0))
        reading["paid_total"] = paid
        if abs(paid - reading["total"]) > config.AMOUNT_TOLERANCE:
            warnings.append(
                f"הסה\"כ שנקרא ({reading['total']}) לא תואם לסכום ששולם ({paid}) – "
                "אחד מהם נקרא לא נכון"
            )

    if reading["currency"] not in (None, "ILS"):
        warnings.append(f"המטבע הוא {reading['currency']} ולא ₪ – הסכום לא הומר")

    prices = [i["price"] for i in reading["items"] if i["price"] is not None]
    if prices and reading["total"] is not None:
        items_sum = _round_money(sum(prices))
        reading["items_sum"] = items_sum
        reading["items_match_total"] = abs(items_sum - reading["total"]) <= config.AMOUNT_TOLERANCE
        # Info only: on long receipts the model often misses lines, while the big total is reliable
        if not reading["items_match_total"]:
            info.append(
                f"סכום הפריטים ({items_sum}) לא תואם לסה\"כ ({reading['total']}) – "
                "רשימת הפריטים חלקית; הסה\"כ אמין, הפריטים רק רמז לקטגוריה"
            )
    return warnings, info


def read_document(file: str) -> dict:
    """Read one image from inbox/ with the model and return structured data plus warnings.

    `file` is a file name as returned by list_documents (not a path).
    """
    if not isinstance(file, str) or not file.strip():
        return _error("חסר שם קובץ")
    path = config.INBOX_DIR / file
    # Only plain names inside inbox/ - blocks paths such as "../data/transactions.json"
    if path.name != file:
        return _error(f"'{file}' אינו שם קובץ תקין – יש לשלוח שם קובץ מתוך list_documents")
    if not path.is_file():
        return _error(f"הקובץ '{file}' לא נמצא בתיקיית inbox")
    if path.suffix.lower() not in config.SUPPORTED_IMAGE_EXTENSIONS:
        return _error(f"סוג הקובץ '{path.suffix}' לא נתמך. נתמכים: {', '.join(sorted(config.SUPPORTED_IMAGE_EXTENSIONS))}")
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > config.MAX_IMAGE_MB:
        return _error(f"הקובץ גדול מדי ({size_mb:.1f}MB, מקסימום {config.MAX_IMAGE_MB}MB) – יש להקטין את התמונה")

    try:
        reading = _ask_model_to_read(path)
    except openai.BadRequestError as e:
        # Usually a damaged file or a file that is not really an image
        return _error(f"המודל לא הצליח לפתוח את '{file}' – ייתכן שהקובץ פגום או שאינו תמונה ({e.message})")
    except openai.OpenAIError as e:
        return _error(f"שגיאה בפנייה למודל: {e}")
    except ValueError as e:  # also json.JSONDecodeError
        return _error(f"התקבלה תשובה לא תקינה מהמודל: {e}")

    if not reading["readable"]:
        reason = reading["unreadable_reason"] or "התמונה לא ברורה"
        return _error(f"{reason}. התמונה לא ברורה – יש לבקש מהמשתמשת להעלות אותה שוב")

    warnings, info = _check_reading(reading)
    return {"ok": True, "file": file, **reading, "warnings": warnings, "info": info}


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
