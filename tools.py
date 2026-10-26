"""The agent's tools: listing files, reading an image, saving, duplicate check, monthly summary.

Only read_document uses AI (it sends the image to the model); the rest is plain Python.

Every tool returns a dict and never raises for bad input: {"ok": True, ...} or
{"ok": False, "error": "..."}. The model reads the error and decides what to do next
(fix the values, re-read the image, ask the user). Only StorageError (a damaged data
file) is raised, because then the agent must stop.
"""

import base64
import copy
import functools
import hashlib
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


def _valid_tax_id(value) -> str | None:
    """An Israeli business number (ח.פ. / ע.מ. / ת.ז.) as 9 digits, or None if it is not valid.

    The last digit is a check digit (digits weighted 1,2,1,2..., two-digit products summed,
    total divisible by 10), so a number with one misread digit fails and is not used.
    """
    digits = re.sub(r"\D", "", str(value or ""))
    if not 8 <= len(digits) <= 9:
        return None
    digits = digits.zfill(9)
    total = 0
    for i, d in enumerate(digits):
        product = int(d) * (1 if i % 2 == 0 else 2)
        total += product - 9 if product > 9 else product
    return digits if total % 10 == 0 else None


def _known_business(tax_id: str | None, business: str | None) -> dict | None:
    """A business recorded before: first by its number (reliable), then by its name."""
    memory = storage.load_memory()
    if tax_id and tax_id in memory["tax_ids"]:
        return {**memory["tax_ids"][tax_id], "matched_by": "tax_id"}
    if business and business in memory["business_categories"]:
        return {"business": business, "category": memory["business_categories"][business], "matched_by": "name"}
    return None


# --- Tools ---

def list_documents() -> dict:
    """Images in inbox/, each marked whether it was already recorded or skipped on purpose (by file name)."""
    if not config.INBOX_DIR.is_dir():
        return _error(f"תיקיית הקלט {config.INBOX_DIR} לא קיימת")

    recorded = {t.get("source_file") for t in storage.load_transactions()}
    skipped = storage.load_skipped()
    documents, unsupported = [], []
    for path in sorted(config.INBOX_DIR.iterdir()):
        if not path.is_file() or path.name.startswith("."):  # skip folders and .gitkeep
            continue
        if path.suffix.lower() in config.SUPPORTED_IMAGE_EXTENSIONS:
            document = {"file": path.name, "already_recorded": path.name in recorded}
            if path.name in skipped:
                document["skipped"] = skipped[path.name]["reason"]
            documents.append(document)
        else:
            unsupported.append(path.name)

    result = {
        "ok": True,
        "documents": documents,
        "new_count": sum(not d["already_recorded"] and "skipped" not in d for d in documents),
        "unsupported_files": unsupported,
    }
    if not documents:
        result["note"] = "אין תמונות בתיקיית inbox"
    return result


# --- read_document: the image-reading prompt and the exact JSON shape the model must return ---

READ_PROMPT = """You read ONE image of an Israeli financial document and extract its data.
Report only what is printed. Never guess: if a field cannot be read, use null.

document_type:
- "receipt": any proof of money paid (קבלה / חשבונית מס קבלה / חשבונית מס) - a store, restaurant,
  online order, service, lesson, tuition (שכר לימוד) or donation (קבלה על תרומה)
- "bill": a utility / municipal / phone / internet bill (חשבון)
- "payslip": a salary slip (תלוש שכר)
- "refund": a credit note or returned goods (זיכוי)
- "other": not a proof of payment - e.g. a bank statement, a price quote or an order that does
  not say it was paid

readable is about image quality only:
- false if the image is too blurry, dark, small or cut off to read every digit of the total with
  certainty. Then set unreadable_reason (short, in Hebrew). Never estimate or guess a number from
  a blurry image - a wrong amount is worse than no amount.
- true for a clear image even if it is not a financial document (then document_type is "other").

business: who issued the document (store, company, employer), as printed, in its original language,
without labels or punctuation around it. Not the customer or the employee. Use the name in the
header at the TOP of the document - prefer the trade name there (e.g. "H&M", "הבירה בית ספר לנהיגה")
over the legal company name in small print (e.g. "מאצי ריטייל בע"מ"). Ignore logos and names in the
background, in a watermark, in an advertisement or at the bottom of the document.

tax_id: the issuer's Israeli business number - ח.פ. / ע.מ. / עוסק מורשה / עמותה רשומה / ע.ר. (usually
9 digits, may be printed with dashes, e.g. 51-420070-8). For a payslip: the employer's number (ח.פ.,
or תיק ניכויים if there is no ח.פ.). Not the customer's ID (ת.ז. after לכבוד), not a phone, receipt or
terminal number. null if not printed or not fully readable.

date_printed: the document's date copied exactly as printed (e.g. "03/03/10"), without the time.
Do not convert or reorder it.
- receipt / refund: the purchase date
- bill: the issue date, labeled תאריך הפקה / תאריך עריכה / תאריך עריכת החשבון (the label may be on
  the line above the date). It is a separate date, usually a few weeks AFTER the billing period:
  never take a date from the period line (מ- ... עד ... / תקופת החשבון) or the due date (לשלם עד)
- payslip: the issue date (הודפס בתאריך) if printed; otherwise the last day of the salary month
  (e.g. "תלוש לחודש יוני 2026" -> "30/06/2026")

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
If there are no item lines, use the line that says what was paid for (עבור / פרטים / תיאור, often
handwritten) as one item - it tells what the money was for.

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
        "tax_id": _NULLABLE_STRING,
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
    "required": ["readable", "unreadable_reason", "document_type", "business", "tax_id",
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


def _cached_reading(path) -> dict:
    """The model's reading of an image, from the cache if this exact image was read before.

    The key is a fingerprint of the file's content (not its name - a copy is the same image)
    plus the prompt and the model, so changing the prompt never reuses old readings.
    Only the raw reading is cached; the checks in _check_reading run again every time.
    """
    if not config.USE_READ_CACHE:
        return _ask_model_to_read(path)
    fingerprint = hashlib.sha256(path.read_bytes())
    fingerprint.update(f"{config.READ_MODEL}|{config.READ_REASONING_EFFORT}|{READ_PROMPT}|{json.dumps(READ_SCHEMA)}".encode())
    key = fingerprint.hexdigest()
    cache = storage.load_read_cache()
    if key not in cache:
        cache[key] = _ask_model_to_read(path)
        storage.save_read_cache(cache)
    return copy.deepcopy(cache[key])  # _check_reading changes the reading it gets


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


_MISSING_BUSINESS = "לא נמצא שם עסק – לשאול את המשתמשת (לא לרשום שם כמו 'לא ידוע')"


def _likely_misread_year(date: str) -> str | None:
    """The same day and month in the current year, if that is recent - the year was probably misread.

    In tests a photo of 22/10/26 was read as 22/10/24. An old date that does not fit this
    pattern (e.g. 17/12/25, read correctly) is not questioned.
    """
    today = calendar_date.today()
    read = calendar_date.fromisoformat(date)
    if read.year == today.year:
        return None
    try:
        this_year = read.replace(year=today.year)
    except ValueError:  # 29/02 in a non-leap year
        return None
    if 0 <= (today - this_year).days <= config.YEAR_MISREAD_DAYS:
        return this_year.isoformat()
    return None


def _check_reading(reading: dict) -> tuple[list[str], list[str]]:
    """Checks done in Python on what the model read. Adds computed fields.

    Returns (warnings, info): a warning means "do not record blindly - ask the user";
    info is worth knowing but does not block recording.
    """
    warnings, info = [], []
    # Names sometimes come back split over lines ("מגה בול\nבול סגולה")
    reading["business"] = " ".join((reading["business"] or "").split()).strip(" :;,.-") or None
    printed_tax_id = reading["tax_id"]
    reading["tax_id"] = _valid_tax_id(printed_tax_id)
    if printed_tax_id and not reading["tax_id"]:
        info.append(f"מספר העסק '{printed_tax_id}' לא עבר בדיקת ספרת ביקורת – הזיהוי יהיה לפי השם")
    if reading["document_type"] == "other":
        warnings.append("זה לא נראה כמו קבלה, חשבון או תלוש – לא לרשום בלי לשאול את המשתמשת")
    if reading["total"] is None:
        warnings.append("לא נמצא סכום כולל")
    if reading["business"] is None:
        warnings.append(_MISSING_BUSINESS)

    reading["date"] = _parse_printed_date(reading["date_printed"])
    if reading["date_printed"] is None:
        warnings.append("לא נמצא תאריך על המסמך")
    elif reading["date"] is None:
        warnings.append(f"לא הצלחתי להבין את התאריך '{reading['date_printed']}' – לשאול את המשתמשת")
    elif problem := _check_date(reading["date"]):
        warnings.append(f"{problem} (מודפס: {reading['date_printed']})")
    elif likely := _likely_misread_year(reading["date"]):
        warnings.append(
            f"ייתכן שהשנה נקראה לא נכון: נקרא {reading['date']}, אולי {likely} (מודפס: {reading['date_printed']}) – "
            "לוודא עם המשתמשת"
        )

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
    Raises openai.OpenAIError for problems that are not about this image (connection, credit, key).
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
        reading = _cached_reading(path)
    except openai.BadRequestError as e:
        # Usually a damaged file or a file that is not really an image
        return _error(f"המודל לא הצליח לפתוח את '{file}' – ייתכן שהקובץ פגום או שאינו תמונה ({e.message})")
    except openai.APITimeoutError:
        return _error(f"קריאת '{file}' לקחה יותר מ-{config.READ_TIMEOUT_SECONDS} שניות – אפשר לנסות שוב בהמשך")
    # Other API errors (no internet, no credit, wrong key) are not about this image: they would
    # fail for every image, so they are not caught here - they stop the run (see agent.run).
    except ValueError as e:  # also json.JSONDecodeError
        return _error(f"התקבלה תשובה לא תקינה מהמודל: {e}")

    if not reading["readable"]:
        reason = reading["unreadable_reason"] or "התמונה לא ברורה"
        return _error(f"{reason}. התמונה לא ברורה – יש לבקש מהמשתמשת להעלות אותה שוב")

    warnings, info = _check_reading(reading)
    known = _known_business(reading["tax_id"], reading["business"])
    if known and reading["business"] is None:  # the number identified it although the name was not read
        reading["business"] = known["business"]
        warnings.remove(_MISSING_BUSINESS)
    return {"ok": True, "file": file, **reading, "known_business": known, "warnings": warnings, "info": info}


def add_transaction(type: str, date: str, amount: float, category: str, business: str,
                    source_file: str, items: list[str] | None = None, tax_id: str | None = None) -> dict:
    """Validate and save one expense or income, and remember the business's category for next time.

    A refund is an expense with a negative amount (not income). For a payslip, amount is the net pay.
    tax_id (the business number) is optional; an invalid one is ignored, not an error.
    Remembering is done here, not left to the model: in run 2 it remembered only 1 of 16 businesses.
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
    # The same file is never recorded twice (a test run recorded a file it had skipped as a duplicate)
    source_file = source_file.strip()
    if recorded := next((t for t in transactions if t.get("source_file") == source_file), None):
        return _error(f"הקובץ '{source_file}' כבר נרשם ({recorded.get('id')}) – אין לרשום אותו שוב")
    if skipped := storage.load_skipped().get(source_file):
        return _error(f"הקובץ '{source_file}' סומן כדולג ({skipped['reason']}) – אין לרשום אותו")
    transaction = {
        "id": storage.next_transaction_id(transactions),
        "type": type,
        "date": date,
        "amount": _round_money(amount),
        "category": category,
        "business": business.strip(),
        "tax_id": _valid_tax_id(tax_id),
        "source_file": source_file.strip(),
        "items": items,
        "added_at": datetime.now().isoformat(timespec="seconds"),
    }
    transactions.append(transaction)
    storage.save_transactions(transactions)
    learned = remember_business(transaction["business"], category, transaction["tax_id"]).get("changed", False)
    return {"ok": True, "transaction": transaction, "learned_business": learned}


def find_duplicate(date: str, amount: float, source_file: str | None = None, tax_id: str | None = None) -> dict:
    """Return already-recorded transactions that may be the same document.

    Candidates are matched by date + amount (business names read from images vary,
    e.g. "שופרסל" / "שופרסל דיל"), by the same file name, or by the same business number +
    amount + day and month in another year (a misread year: 22/10/24 vs 22/10/26). Only the
    year may differ: a monthly payment of the same amount (rent, a donation) is not a duplicate.
    The agent compares and decides whether it really is a duplicate.
    """
    if problem := _check_date(date):
        return _error(problem)
    if problem := _check_amount(amount):
        return _error(problem)

    tax_id = _valid_tax_id(tax_id)
    candidates = []
    for t in storage.load_transactions():
        same_amount = t.get("amount") == _round_money(amount)
        if source_file is not None and t.get("source_file") == source_file:
            matched_by = "same_file"
        elif same_amount and t.get("date") == date:
            matched_by = "date_and_amount"
        elif same_amount and tax_id and t.get("tax_id") == tax_id and str(t.get("date", ""))[5:] == date[5:]:
            matched_by = "business_number_amount_and_day"
        else:
            continue
        candidates.append({
            "id": t.get("id"),
            "date": t.get("date"),
            "amount": t.get("amount"),
            "business": t.get("business"),
            "source_file": t.get("source_file"),
            "matched_by": matched_by,
        })
    return {"ok": True, "candidates": candidates, "possible_duplicate": bool(candidates)}


def skip_document(file: str, reason: str) -> dict:
    """Mark a file as handled without recording it (a duplicate, or she said not to record it).

    The next run's list_documents marks it as skipped, so it is not handled again.
    Not for unreadable images - those should come back as a new photo.
    """
    if not isinstance(file, str) or (config.INBOX_DIR / file).name != file or not (config.INBOX_DIR / file).is_file():
        return _error(f"הקובץ '{file}' לא נמצא בתיקיית inbox")
    if not isinstance(reason, str) or not reason.strip():
        return _error("חסרה סיבה לדילוג")
    skipped = storage.load_skipped()
    skipped[file] = {"reason": reason.strip(), "skipped_at": datetime.now().isoformat(timespec="seconds")}
    storage.save_skipped(skipped)
    return {"ok": True, "file": file, "reason": reason.strip()}


def remember_business(business: str, category: str, tax_id: str | None = None) -> dict:
    """Remember a business's category by name, and by number if it has a valid one (called by add_transaction).

    "changed" says whether the memory learned something new.
    """
    if not isinstance(business, str) or not business.strip():
        return _error("חסר שם העסק")
    if category not in config.EXPENSE_CATEGORIES + config.INCOME_CATEGORIES:
        return _error(f"הקטגוריה '{category}' לא קיימת")

    business = business.strip()
    memory = storage.load_memory()
    before = json.dumps(memory, sort_keys=True)
    memory["business_categories"][business] = category
    if tax_id := _valid_tax_id(tax_id):
        memory["tax_ids"][tax_id] = {"business": business, "category": category}
    changed = json.dumps(memory, sort_keys=True) != before
    if changed:
        storage.save_memory(memory)
    return {"ok": True, "business": business, "category": category, "changed": changed}


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
