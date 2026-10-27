"""The agent's tools: listing files, reading an image, saving, duplicate check, monthly summary.

Only read_document uses AI (it sends the image to the model); the rest is plain Python.
read_document also looks up company numbers in the public companies registry (no AI).

Every tool returns a dict and never raises for bad input: {"ok": True, ...} or
{"ok": False, "error": "..."}. The model reads the error and decides what to do next
(fix the values, re-read the image, ask the user). Only StorageError (a damaged data
file) is raised, because then the agent must stop.
"""

import base64
import copy
import csv
import difflib
import functools
import hashlib
import io
import json
import math
import re
import urllib.parse
import urllib.request
# Aliased: "date" is also a tool parameter name (the model sends it by that name)
from datetime import date as calendar_date, datetime, timedelta

import openai
from PIL import Image

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
    """Images in inbox/, each marked whether it was already recorded or skipped on purpose.

    Already recorded = by file name, or an exact copy of a recorded file (same content). A test run:
    two re-uploaded copies (x-2, x-3) were read and questioned before the copy was noticed.
    """
    if not config.INBOX_DIR.is_dir():
        return _error(f"תיקיית הקלט {config.INBOX_DIR} לא קיימת")

    transactions = storage.load_transactions()
    recorded = {t.get("source_file") for t in transactions}
    recorded_content = {}  # fingerprint of a recorded file -> its transaction id
    for t in transactions:
        if (content := _file_content(t.get("source_file"))) is not None:
            recorded_content.setdefault(hashlib.sha256(content).hexdigest(), t.get("id"))
    skipped = storage.load_skipped()
    documents, unsupported = [], []
    for path in sorted(config.INBOX_DIR.iterdir()):
        if not path.is_file() or path.name.startswith("."):  # skip folders and .gitkeep
            continue
        if path.suffix.lower() in config.SUPPORTED_EXTENSIONS:
            document = {"file": path.name, "already_recorded": path.name in recorded}
            if not document["already_recorded"]:
                copy_of = recorded_content.get(hashlib.sha256(path.read_bytes()).hexdigest())
                if copy_of:
                    document.update(already_recorded=True, copy_of=copy_of)
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
Report only what is written on the document - printed or handwritten (many receipts are filled in by
hand: amount, date, what was paid for). Never guess: if a field cannot be read with certainty, use null.

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

date_printed: the document's date copied exactly as written - printed or handwritten (e.g. "03/03/10",
"14.3.2025") - without the time.
Do not convert or reorder it.
- receipt / refund: the purchase date
- bill: the issue date, labeled תאריך הפקה / תאריך עריכה / תאריך עריכת החשבון (the label may be on
  the line above the date). It is a separate date, usually a few weeks AFTER the billing period:
  never take a date from the period line (מ- ... עד ... / תקופת החשבון) or the due date (לשלם עד)
- payslip: the issue date (הודפס בתאריך) if printed; otherwise the last day of the salary month
  (e.g. "תלוש לחודש יוני 2026" -> "30/06/2026")

date_handwritten: true if the date is written by hand, false if it is printed.
total_handwritten: true if the total is written by hand, false if it is printed.

billing_period_end_printed: bills only - the last day of the billing period (תקופת החשבון), exactly as
printed. null for other documents.

total: the amount actually paid or received, as a positive number.
- receipt: לתשלום / סה"כ - not the cash handed over (מזומן) or the change (עודף)
- bill: סה"כ לתשלום
- payslip: the final לתשלום line (after all deductions), not gross pay; use שכר נטו only if there is no לתשלום line
- refund: the refunded amount, still positive
- other (e.g. an invoice that was not paid yet, a price quote): the document's total as printed, so she
  can decide about it
Lines such as "you saved" (חסכת / הנחה), VAT (מע"מ) or points are not the total.

payments: how the total was paid, as printed in the payment section - one entry per payment line
(credit card, cash, voucher), each with its amount. A purchase split between several cards has
several entries. Empty if no payment lines are printed. change: the change given back (עודף), or null.

currency: ISO code; ₪ / ש"ח / NIS = "ILS".
total_ils_printed: only for a total in another currency - the ₪ equivalent of that total if the
document prints one (e.g. "Equivalent ... NIS", "שווה ערך בש\"ח"), otherwise null.

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
        "date_handwritten": {"type": "boolean"},
        "total_handwritten": {"type": "boolean"},
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
        "total_ils_printed": _NULLABLE_NUMBER,
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
                 "date_printed", "date_handwritten", "total_handwritten", "billing_period_end_printed", "total", "payments", "change", "currency", "total_ils_printed", "items", "notes"],
    "additionalProperties": False,
}

_MIME_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
               ".pdf": "application/pdf"}


@functools.cache
def _openai_client() -> openai.OpenAI:
    # Created on first use; reads OPENAI_API_KEY from the environment (raises OpenAIError if missing)
    return openai.OpenAI(timeout=config.READ_TIMEOUT_SECONDS)


_ROTATION_PROMPT = """This is a photo of a receipt or another document. By how many degrees must the
photo be turned CLOCKWISE so that its text reads normally, upright (0 if it is already upright)?"""
_ROTATION_SCHEMA = {
    "type": "object",
    "properties": {"rotation": {"type": "integer", "enum": [0, 90, 180, 270]}},
    "required": ["rotation"],
    "additionalProperties": False,
}


_UPRIGHT_PROMPT = """Two copies of the same photo, turned differently. In which one does the text read
normally, the right way up (not upside down)?"""
_UPRIGHT_SCHEMA = {
    "type": "object",
    "properties": {"upright": {"type": "string", "enum": ["A", "B"]}},
    "required": ["upright"],
    "additionalProperties": False,
}


def _small_image_part(image: Image.Image) -> dict:
    small = image.convert("RGB")
    small.thumbnail((config.ROTATION_CHECK_PIXELS, config.ROTATION_CHECK_PIXELS))
    buffer = io.BytesIO()
    small.save(buffer, format="JPEG", quality=85)  # without EXIF: the model sees the pixels that are turned
    data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}}


def _ask_small(prompt: str, schema: dict, content: list) -> dict:
    response = _openai_client().chat.completions.create(
        model=config.READ_MODEL,
        reasoning_effort=config.ROTATION_REASONING_EFFORT,
        response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": schema, "strict": True}},
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": content}],
    )
    return json.loads(response.choices[0].message.content)


def _text_rotation(image: Image.Image) -> int:
    """Degrees to turn the image clockwise so its text is upright. Raises on API problems.

    Measured (5 tries per image): "already upright" was always right, but about 1 in 10 sideways
    answers mixed up 90 and 270, and the receipt was read upside down ("פיניש ווש" became
    "הולנדיה"). Choosing which of two turned copies is upright was right 18 of 18 times - so for
    a sideways answer, that easier question decides the direction.
    """
    rotation = _ask_small(_ROTATION_PROMPT, _ROTATION_SCHEMA, [_small_image_part(image)])["rotation"]
    if rotation not in (90, 270):
        return rotation
    choice = _ask_small(_UPRIGHT_PROMPT, _UPRIGHT_SCHEMA, [
        {"type": "text", "text": "A:"}, _small_image_part(image.rotate(-90, expand=True)),
        {"type": "text", "text": "B:"}, _small_image_part(image.rotate(-270, expand=True)),
    ])["upright"]
    return 90 if choice == "A" else 270


def _upright_content(path) -> bytes:
    """The file's bytes, turned upright first if it is a landscape photo whose text lies sideways.

    Portrait photos and PDFs are sent exactly as before. A landscape photo is not always sideways
    (a test water bill was photographed landscape and upright), so the model is asked - and its
    answer is cached like a reading. Unchanged bytes keep their old read-cache entries.
    """
    content = path.read_bytes()
    if path.suffix.lower() == ".pdf":
        return content
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except OSError:  # not an image Pillow knows - the model gets it as is and says if it cannot read it
        return content
    if image.width <= image.height:
        return content
    key = "rotation|" + hashlib.sha256(content).hexdigest() + f"|{config.READ_MODEL}|{_ROTATION_PROMPT}|{_UPRIGHT_PROMPT}"
    cache = storage.load_read_cache() if config.USE_READ_CACHE else {}
    if key not in cache:
        try:
            cache[key] = _text_rotation(image)
        except (ValueError, TypeError, KeyError):  # a refusal or an odd answer - read it as it is
            return content
        if config.USE_READ_CACHE:
            storage.save_read_cache(cache)
    if cache[key] == 0:
        return content
    buffer = io.BytesIO()
    # rotate() turns counterclockwise; quality is ignored by formats without it (PNG)
    image.rotate(-cache[key], expand=True).save(buffer, format=image.format or "PNG", quality=95)
    return buffer.getvalue()


def _ask_model_to_read(path, content: bytes) -> dict:
    """Send the image (or PDF) to the model and return its parsed reply. Raises on API problems."""
    file_b64 = base64.b64encode(content).decode("ascii")
    data_url = f"data:{_MIME_TYPES[path.suffix.lower()]};base64,{file_b64}"
    if path.suffix.lower() == ".pdf":
        document_part = {"type": "file", "file": {"filename": path.name, "file_data": data_url}}
    else:
        # Always high detail: in tests, low detail invented items that are not on the receipt
        document_part = {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}
    response = _openai_client().chat.completions.create(
        model=config.READ_MODEL,  # no temperature: gpt-5 models accept only the default
        reasoning_effort=config.READ_REASONING_EFFORT,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "document", "schema": READ_SCHEMA, "strict": True},
        },
        messages=[
            {"role": "system", "content": READ_PROMPT},
            {"role": "user", "content": [document_part]},
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
    content = _upright_content(path)
    if not config.USE_READ_CACHE:
        return _ask_model_to_read(path, content)
    fingerprint = hashlib.sha256(content)
    fingerprint.update(f"{config.READ_MODEL}|{config.READ_REASONING_EFFORT}|{READ_PROMPT}|{json.dumps(READ_SCHEMA)}".encode())
    key = fingerprint.hexdigest()
    cache = storage.load_read_cache()
    if key not in cache:
        cache[key] = _ask_model_to_read(path, content)
        storage.save_read_cache(cache)
    return copy.deepcopy(cache[key])  # _check_reading changes the reading it gets


_MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def normalize_answer(answer: str, answer_type: str) -> str | None:
    """Her answer to a question about a missing date or amount, in the form the tools use - or None
    if it is not a valid one. A test run: she typed "מ" and "222222" for a date and an amount.
    "date" -> "YYYY-MM-DD" (not in the future); "amount" -> a number above 0, e.g. "200.0".
    """
    answer = answer.strip()
    if answer_type == "date":
        date = answer if _check_date(answer) is None else _parse_printed_date(answer)
        return date if date and _check_date(date) is None else None
    if answer_type == "amount":
        try:
            amount = float(re.sub(r"[₪,\s]|ש\"?ח|שקל(ים)?", "", answer))
        except ValueError:
            return None
        return str(_round_money(amount)) if math.isfinite(amount) and amount > 0 else None
    return answer or None


def _parse_printed_date(printed: str | None) -> str | None:
    """Israeli day-first date as printed ("03/03/10", "27.02.19", "10/07/2020") -> "YYYY-MM-DD".

    Also an English month name, day first or month first ("08-Sep-26", "8 September 2026",
    "Sep 8, 2026") - a travel agency receipt had "08-Sep-26" and the agent asked her for the date.
    Done in Python, not by the model: in tests the model converted dates inconsistently.
    """
    printed = printed or ""
    if match := re.search(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4}|\d{2})\b", printed):
        day, month, year = (int(g) for g in match.groups())
    elif match := re.search(r"\b(\d{1,2})[\s./-]*([A-Za-z]{3})[A-Za-z]*\.?[\s./,-]*(\d{4}|\d{2})\b", printed):
        day, name, year = int(match[1]), match[2].lower(), int(match[3])
        if name not in _MONTH_NAMES:
            return None
        month = _MONTH_NAMES.index(name) + 1
    elif match := re.search(r"\b([A-Za-z]{3})[A-Za-z]*\.?\s+(\d{1,2}),?\s+(\d{4})\b", printed):
        name, day, year = match[1].lower(), int(match[2]), int(match[3])
        if name not in _MONTH_NAMES:
            return None
        month = _MONTH_NAMES.index(name) + 1
    else:
        return None
    if year < 100:
        year += 2000  # receipts are from this century
    try:
        return calendar_date(year, month, day).isoformat()
    except ValueError:  # e.g. 31/02
        return None


# Warnings state facts in her words: what to do about them is in the agent's prompt (and its Python guard)
_MISSING_BUSINESS = "לא הצלחתי לזהות שם עסק במסמך"


def _likely_misread_year(date: str, printed: str) -> str | None:
    """The same day and month in the current year, if that is recent - the year was probably misread.

    In tests a photo of 22/10/26 was read as 22/10/24. Only a two-digit year is suspected: a
    four-digit year is printed clearly (a 2022 bill dated 19/10/2022 was wrongly questioned).
    An old date that does not fit this pattern (e.g. 17/12/25, read correctly) is not questioned.
    """
    if re.search(r"\d{4}", printed):
        return None
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


@functools.lru_cache(maxsize=256)
def _exchange_rate(currency: str, day: str) -> tuple[float, str] | None:
    """Bank of Israel official rate of `currency` in ₪ on `day`, or on the last business day before it.

    Returns (rate, the rate's date), or None if there is no rate. Raises OSError / ValueError /
    KeyError if the bank cannot be reached or answers badly (errors are not cached).
    """
    start = (calendar_date.fromisoformat(day) - timedelta(days=10)).isoformat()
    query = urllib.parse.urlencode({"startperiod": start, "endperiod": day, "format": "csv"})
    url = config.EXCHANGE_RATE_URL.format(currency=currency) + "?" + query
    with urllib.request.urlopen(url, timeout=config.EXCHANGE_RATE_TIMEOUT_SECONDS) as response:
        rows = list(csv.DictReader(io.StringIO(response.read().decode("utf-8-sig"))))
    if not rows:
        return None
    last = max(rows, key=lambda row: row["TIME_PERIOD"])
    return float(last["OBS_VALUE"]), last["TIME_PERIOD"]


def _convert_to_ils(reading: dict, warnings: list[str], info: list[str]) -> None:
    """Convert the amounts of a document in a foreign currency to ₪.

    The rate: the document's own ₪ equivalent if it prints one (what was really charged), otherwise
    the Bank of Israel official rate of its date. The original total, currency and rate go to
    reading["original"], and add_transaction keeps them.
    Called after the cross-checks, which compare amounts in the document's own currency.
    """
    currency = str(reading["currency"]).upper()
    total, printed_ils = reading["total"], reading.pop("total_ils_printed", None)
    if total and printed_ils and printed_ils > 0:
        rate, source = printed_ils / total, "לפי הסכום בשקלים שמודפס במסמך"
        rate_date, ils_total = reading["date"], printed_ils
    else:
        not_converted = f"המטבע הוא {currency} ולא ₪ – הסכום לא הומר"
        if not config.USE_ONLINE_LOOKUPS or not re.fullmatch(r"[A-Z]{3}", currency) or not reading["date"]:
            warnings.append(not_converted)
            return
        try:
            found = _exchange_rate(currency, reading["date"])
        except (OSError, ValueError, KeyError):  # no internet, unknown currency (404), unexpected answer
            found = None
        if found is None:
            warnings.append(f"{not_converted} (לא נמצא שער יציג של בנק ישראל)")
            return
        (rate, rate_date), source = found, "לפי השער היציג של בנק ישראל"
        ils_total = total * rate if total is not None else None
    reading["original"] = {"total": total, "currency": currency, "rate": round(rate, 4), "rate_date": rate_date}
    if ils_total is not None:
        reading["total"] = _round_money(ils_total)
    for payment in reading["payments"]:
        payment["amount"] = _round_money(payment["amount"] * rate)
    for key in ("change", "paid_total", "items_sum"):
        if reading.get(key) is not None:
            reading[key] = _round_money(reading[key] * rate)
    for item in reading["items"]:
        if item["price"] is not None:
            item["price"] = _round_money(item["price"] * rate)
    reading["currency"] = "ILS"
    info.append(f"הסכום הומר מ-{currency} {source} (שער {round(rate, 4)}): {total} {currency} = {reading['total']} ₪")


def _check_reading(reading: dict) -> tuple[list[str], list[str]]:
    """Checks done in Python on what the model read. Adds computed fields.

    Returns (warnings, info): a warning means "do not record blindly - ask the user";
    info is worth knowing but does not block recording.
    """
    warnings, info = [], []
    reading["confirm"] = []  # values she must confirm herself ("date", "amount") - checked by the agent loop
    # Names sometimes come back split over lines ("מגה בול\nבול סגולה")
    reading["business"] = " ".join((reading["business"] or "").split()).strip(" :;,.-") or None
    printed_tax_id = reading["tax_id"]
    reading["tax_id"] = _valid_tax_id(printed_tax_id)
    if printed_tax_id and not reading["tax_id"]:
        info.append(f"מספר העסק '{printed_tax_id}' לא עבר בדיקת ספרת ביקורת – הזיהוי יהיה לפי השם")
    if reading["document_type"] == "other":
        # The reason goes into her question: "it does not look like a receipt" alone did not help her decide
        warnings.append("זו לא הוכחת תשלום (לא קבלה, חשבון או תלוש)"
                        + (f": {reading['notes']}" if reading.get("notes") else "")
                        + ". הוצאה נרשמת לפי הקבלה כשמשלמים – אם נרשום גם את המסמך הזה וגם את הקבלה, "
                        "הסכום ייספר פעמיים")
    if reading["total"] is None:
        warnings.append("לא הצלחתי לזהות סכום כולל במסמך")
    elif reading.get("total_handwritten"):
        # A test run: a handwritten 320 was read as 300 - Python cannot check handwritten digits
        warnings.append(f"הסכום כתוב ביד ונקרא {reading['total']} ₪ – ספרות בכתב יד עלולות להיקרא לא נכון")
        reading["confirm"].append("amount")
    if reading["business"] is None:
        warnings.append(_MISSING_BUSINESS)

    reading["date"] = _parse_printed_date(reading["date_printed"])
    # No usable date: it must come from her. A real run: a gift-voucher email with no date was
    # recorded with today's date after she said "record anyway" - the prompt alone did not stop it.
    if reading["date_printed"] is None:
        warnings.append("לא הצלחתי לזהות תאריך במסמך")
        reading["confirm"].append("date")
    elif reading["date"] is None:
        warnings.append(f"לא הצלחתי להבין את התאריך '{reading['date_printed']}'")
        reading["confirm"].append("date")
    elif problem := _check_date(reading["date"]):
        warnings.append(f"{problem} (מודפס: {reading['date_printed']})")
        reading["confirm"].append("date")
    elif likely := _likely_misread_year(reading["date"], reading["date_printed"]):
        warnings.append(
            f"ייתכן שהשנה נקראה לא נכון – התאריך הסביר: {likely} (נקרא {reading['date']}, מודפס: {reading['date_printed']})"
        )
        reading["confirm"].append("date")
    elif reading.get("date_handwritten") and (likely := _handwritten_year(reading["date"], reading["date_printed"])):
        warnings.append(
            f"השנה כתובה ביד – התאריך הסביר: {likely} (נקרא {reading['date']}, כתוב: {reading['date_printed']})"
        )
        reading["confirm"].append("date")

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
        _convert_to_ils(reading, warnings, info)
    reading.pop("total_ils_printed", None)

    prices = [i["price"] for i in reading["items"] if i["price"] is not None]
    if prices and reading["total"] is not None:
        items_sum = _round_money(sum(prices))
        reading["items_sum"] = items_sum
        reading["items_match_total"] = any(
            abs(items_sum * (1 + vat) - reading["total"]) <= config.AMOUNT_TOLERANCE
            for vat in (0, *config.VAT_RATES)
        )
        # Info only: on long receipts the model often misses lines, while the big total is reliable
        if not reading["items_match_total"]:
            info.append(
                f"סכום הפריטים ({items_sum}) לא תואם לסה\"כ ({reading['total']}) – "
                "רשימת הפריטים חלקית; הסה\"כ אמין, הפריטים רק רמז לקטגוריה"
            )
    return warnings, info


def _handwritten_year(date: str, printed: str) -> str | None:
    """For a handwritten two-digit year that is not this year: the latest past date with the same day and month.

    A test run: "15/3/26" written by hand was read as 15/3/22 and recorded without a question
    (_likely_misread_year covers only the last few months). A printed year is not questioned -
    an old printed receipt (a 2013 fuel receipt) must not get a question.
    """
    if re.search(r"\d{4}", printed):
        return None
    today = calendar_date.today()
    read = calendar_date.fromisoformat(date)
    for year in (today.year, today.year - 1):
        try:
            candidate = read.replace(year=year)
        except ValueError:  # 29/02 in a non-leap year
            continue
        if candidate <= today:
            return candidate.isoformat() if candidate != read else None
    return None


@functools.lru_cache(maxsize=256)
def _registry_name(tax_id: str) -> str | None:
    """A company's legal name from the companies registry, or None if the number is not there.

    Raises OSError / ValueError / KeyError if the registry cannot be reached or answers badly
    (errors are not cached, so the next document tries again).
    """
    query = urllib.parse.urlencode({
        "resource_id": config.REGISTRY_RESOURCE_ID,
        "filters": json.dumps({"מספר חברה": int(tax_id)}),
        "fields": "שם חברה",
    })
    with urllib.request.urlopen(f"{config.REGISTRY_URL}?{query}", timeout=config.REGISTRY_TIMEOUT_SECONDS) as response:
        records = json.loads(response.read())["result"]["records"]
    if not records:
        return None
    # The registry writes ~ for the quote mark ("בע~מ"), and sometimes leaves it out ("בעמ")
    name = " ".join(str(records[0]["שם חברה"]).replace("~", '"').split())
    return re.sub(r"\bבעמ$", 'בע"מ', name) or None


def _name_letters(text: str | None) -> str:
    return re.sub(r"[\s\"'׳״`-]", "", text or "")


def _with_familiar_name(printed: str | None, legal: str) -> str:
    """The name to record: "אושר עד (מרב - מזון כל בע"מ)" - the name people know, the legal one beside it.

    Her request: the legal name alone was unfamiliar (H&M -> "מאצ' ריטייל"). A printed name close to
    the legal one, or to a part of it, is probably a misreading of it ("שוקי פרר" / "שוקי פון בע"מ";
    a sample: "הפינה" / "הבירה- בי"ס לנהיגה בע"מ") - then the legal name alone.
    """
    printed_letters, legal_letters = _name_letters(printed), _name_letters(legal)
    if not printed_letters:
        return legal
    # Every part of the legal name as long as the printed one (the whole name if it is shorter)
    size = len(printed_letters)
    parts = [legal_letters[i:i + size] for i in range(max(1, len(legal_letters) - size + 1))]
    if max(difflib.SequenceMatcher(None, printed_letters, part).ratio() for part in parts) >= 0.5:
        return legal
    return f"{printed} ({legal})"


def _apply_registry_name(reading: dict, warnings: list[str], info: list[str]) -> str | None:
    """Add the company's legal name to the name read from the image. Returns the legal name, or None.

    The legal name is trusted over the photo - a misread one ("שוקי פרר" instead of "שוקי פון") is
    replaced; a brand ("אושר עד") is kept beside it (see _with_familiar_name).
    """
    tax_id = reading["tax_id"]
    if not config.USE_ONLINE_LOOKUPS or not tax_id or not tax_id.startswith(("51", "52")):
        return None
    try:
        name = _registry_name(tax_id)
    except (OSError, ValueError, KeyError, TypeError):  # no internet, timeout, unexpected answer
        info.append("מאגר רשם החברות לא זמין כרגע – השם נשאר כפי שנקרא מהתמונה")
        return None
    if name is None:
        return None
    if _name_letters(name) == _name_letters(reading["business"]):  # the same name, only spaces or quote marks differ
        name = reading["business"]
    elif name != reading["business"]:
        reading["business_printed"] = reading["business"]
        reading["business"] = _with_familiar_name(reading["business"], name)
        info.append(f"השם לפי רשם החברות (ח.פ. {tax_id}): {name}. מהתמונה נקרא: {reading['business_printed'] or 'שום שם'}")
    if _MISSING_BUSINESS in warnings:
        warnings.remove(_MISSING_BUSINESS)
    return name


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
    if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        return _error(f"סוג הקובץ '{path.suffix}' לא נתמך. נתמכים: {', '.join(sorted(config.SUPPORTED_EXTENSIONS))}")
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > config.MAX_IMAGE_MB:
        return _error(f"הקובץ גדול מדי ({size_mb:.1f}MB, מקסימום {config.MAX_IMAGE_MB}MB) – יש להקטין את התמונה")

    try:
        reading = _cached_reading(path)
    except openai.BadRequestError as e:
        # Usually a damaged file or a file that is not really an image
        return _error(f"המודל לא הצליח לפתוח את '{file}' – ייתכן שהקובץ פגום או שאינו תמונה או PDF ({e.message})")
    except openai.APITimeoutError:
        return _error(f"קריאת '{file}' לקחה יותר מ-{config.READ_TIMEOUT_SECONDS} שניות – אפשר לנסות שוב בהמשך")
    # Other API errors (no internet, no credit, wrong key) are not about this image: they would
    # fail for every image, so they are not caught here - they stop the run (see agent.run).
    except ValueError as e:  # also json.JSONDecodeError
        return _error(f"התקבלה תשובה לא תקינה מהמודל: {e}")

    if not reading["readable"]:
        reason = reading["unreadable_reason"] or "התמונה לא ברורה"
        return _error(f"{reason}. התמונה לא ברורה – צריך לצלם אותה שוב")

    warnings, info = _check_reading(reading)
    legal_name = _apply_registry_name(reading, warnings, info)
    known = _known_business(reading["tax_id"], reading["business"])
    # Memory's name wins (the first recording, or her ✏️ fix): a real run read an אושר עד receipt as
    # "מגה בעיר", which would have replaced it. Only a name without the legal one in it is replaced -
    # it is from before the registry check and may itself be a misreading ("שוקי פרר").
    if known and legal_name and _name_letters(legal_name) not in _name_letters(known["business"]):
        known["business"] = reading["business"]
    # A fuller name that keeps memory's name whole only adds to it: a sideways 08.jpg was remembered as
    # "מרב - מזון כל בע"מ", then 01.jpg read "אושר עד גבעת שאול (מרב - מזון כל בע"מ)" and lost the brand
    if known and reading["business"] and _name_letters(known["business"]) != _name_letters(reading["business"]) \
            and _name_letters(known["business"]) in _name_letters(reading["business"]):
        known["business"] = reading["business"]
    if known and reading["business"] is None:  # the number identified it although the name was not read
        reading["business"] = known["business"]
        warnings.remove(_MISSING_BUSINESS)
    return {"ok": True, "file": file, **reading, "known_business": known, "warnings": warnings, "info": info}


def add_transaction(type: str, date: str, amount: float, category: str, business: str,
                    source_file: str, items: list[str] | None = None, tax_id: str | None = None,
                    original: dict | None = None) -> dict:
    """Validate and save one expense or income, and remember the business's category for next time.

    A refund is an expense with a negative amount (not income). For a payslip, amount is the net pay.
    tax_id (the business number) is optional; an invalid one is ignored, not an error.
    original: the foreign-currency total and rate from read_document - added by the agent loop,
    not by the model, so the converted amount can always be traced back.
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
    # In a test run the model mistyped a file name; the record then looked like a new file next run
    if (config.INBOX_DIR / source_file.strip()).name != source_file.strip() or not (config.INBOX_DIR / source_file.strip()).is_file():
        return _error(f"הקובץ '{source_file}' לא נמצא בתיקיית inbox – יש להעתיק את השם בדיוק מ-list_documents")
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
    if original:
        transaction["original"] = original
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
    # A copy of a recorded image under another name (a test run uploaded "x-2.jpeg" and recorded it twice)
    content = _file_content(source_file)
    candidates = []
    for t in storage.load_transactions():
        same_amount = t.get("amount") == _round_money(amount)
        if source_file is not None and t.get("source_file") == source_file:
            matched_by = "same_file"
        elif content is not None and _file_content(t.get("source_file")) == content:
            matched_by = "same_file_content"
        elif same_amount and t.get("date") == date:
            matched_by = "date_and_amount"
        elif same_amount and tax_id and t.get("tax_id") == tax_id and str(t.get("date", ""))[5:] == date[5:]:
            matched_by = "business_number_amount_and_day"
        elif tax_id and t.get("tax_id") == tax_id and t.get("date") == date:
            # Another photo of the same handwritten receipt can come back with another amount
            # (630 / 650) - the same business on the same day may be the same receipt: she decides
            matched_by = "same_business_and_date"
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


def recorded_copy(file) -> str | None:
    """The id of a recorded transaction whose image has exactly the same content as this file, or None.

    A test run: a re-uploaded copy (x-2.jpeg) was found by find_duplicate, yet the model asked about
    the amount and recorded it again - the same bytes are the same document, for certain.
    """
    content = _file_content(file)
    if content is None:
        return None
    for t in storage.load_transactions():
        if t.get("source_file") != file and _file_content(t.get("source_file")) == content:
            return t.get("id")
    return None


def _file_content(file) -> bytes | None:
    """The bytes of a file directly inside inbox/, or None."""
    if not isinstance(file, str) or (config.INBOX_DIR / file).name != file:
        return None
    path = config.INBOX_DIR / file
    return path.read_bytes() if path.is_file() else None


def skip_document(file: str, reason: str) -> dict:
    """Mark a file as handled without recording it (a duplicate, or she said not to record it).

    The next run's list_documents marks it as skipped, so it is not handled again.
    Not for unreadable images - those should come back as a new photo.
    """
    if not isinstance(file, str) or (config.INBOX_DIR / file).name != file or not (config.INBOX_DIR / file).is_file():
        return _error(f"הקובץ '{file}' לא נמצא בתיקיית inbox")
    if not isinstance(reason, str) or not reason.strip():
        return _error("חסרה סיבה לדילוג")
    # A test run recorded 08.jpg, then "skipped" it as a duplicate of a receipt recorded after it
    if recorded := next((t for t in storage.load_transactions() if t.get("source_file") == file), None):
        return _error(f"הקובץ '{file}' כבר נרשם ({recorded.get('id')}) – אין לדלג עליו")
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


def correct_transaction(transaction_id: str, business: str, category: str,
                        date: str | None = None, amount: float | None = None) -> dict:
    """Her correction of a recorded transaction (from the web dashboard) - not a tool the model can call.

    date and amount too (optional): a handwritten amount can be misread, and before this she could
    only fix the name and category. The amount keeps its sign - a refund stays negative.

    The image reading can be wrong in a way the agent cannot notice (a test read "שרותי מזון" as
    "שירותי מיזוג"), and the wrong name was then remembered by business number for every later
    receipt. So the correction also replaces what was remembered: next time the agent uses her name.
    """
    if not isinstance(business, str) or not business.strip():
        return _error("חסר שם העסק")
    business = business.strip()
    transactions = storage.load_transactions()
    transaction = next((t for t in transactions if t.get("id") == transaction_id), None)
    if transaction is None:
        return _error(f"התנועה {transaction_id} לא נמצאה")
    allowed = _categories_for(transaction.get("type"))
    if category not in allowed:
        return _error(f"הקטגוריה '{category}' לא קיימת. אפשריות: {', '.join(allowed)}")
    if date is not None and (problem := _check_date(date)):
        return _error(problem)
    if amount is not None:
        if problem := _check_amount(amount):
            return _error(problem)
        if amount < 0:
            return _error("הסכום חייב להיות חיובי (זיכוי נשאר זיכוי)")

    old_business = transaction.get("business")
    transaction["business"], transaction["category"] = business, category
    if date is not None:
        transaction["date"] = date
    if amount is not None:
        transaction["amount"] = -_round_money(amount) if transaction.get("amount", 0) < 0 else _round_money(amount)
    transaction["corrected_at"] = datetime.now().isoformat(timespec="seconds")
    storage.save_transactions(transactions)

    # The old (misread) name is forgotten unless another transaction still uses it
    memory = storage.load_memory()
    if old_business != business and not any(t.get("business") == old_business for t in transactions):
        memory["business_categories"].pop(old_business, None)
        storage.save_memory(memory)
    remember_business(business, category, transaction.get("tax_id"))
    return {"ok": True, "transaction": transaction}
