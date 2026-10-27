"""The agent loop: the model decides the next step, Python runs the tool, the result goes back.

The agent never prints or reads input itself - everything goes through an AgentUI, so the
same loop can run with a terminal or a web UI.

The model gets a goal and rules (SYSTEM_PROMPT), not a fixed order of steps: it decides
which tool to call, reads the result and decides what to do next. It stops when it answers
without calling a tool (goal reached) or after config.MAX_STEPS model calls.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import date as calendar_date

import openai

import config
import storage
import tools
from ui.base import AgentUI

SYSTEM_PROMPT = """You are an expense-tracking agent for one person in Israel. Today is {today}.

GOAL: process every new document (receipt, bill, payslip, refund) in her inbox, record each one
correctly exactly once, then show her the monthly summary and report what happened.

WORK ONE DOCUMENT AT A TIME: finish a document - read it, check for a duplicate, ask her if
needed, then record it (add_transaction) or skip it (skip_document) - before you read the next
one. Within a document you decide which steps are needed. (Reading everything first made you
forget documents she had approved, and she could not tell which document a late question was about.)

RULES
- Documents marked already_recorded or skipped are done - do not read them again.
- Before recording a document, check that it is not already recorded (find_duplicate, with the
  document's tax_id). The candidates may be the same document with a slightly different business
  name ("שופרסל" / "שופרסל דיל", "מיכל עטר" / "מיכל עזר") or date - compare and decide. If you are
  not sure, ask her.
- skip_document: call it when you decide for good not to record a document - a duplicate, or she
  said not to record it - with a short reason in Hebrew. Not for an unreadable image (she should
  upload it again) and not when she gave no answer (ask again next time).
- read_document "warnings" mean the reading may be wrong: do not record blindly. Ask her, say
  what looks wrong, and record only what she confirms (or skip if she says so). add_transaction
  refuses a file with warnings until she has answered a question about that file.
  "info" is for your knowledge only and does not block recording - it is never a reason to ask.
  Items that do not add up to the total are normal (lines missed, prices before VAT): the total
  is what is recorded, the items only hint at the category.
- Questions: ask about ONE document per ask_user call and pass its file name in "file" (her screen
  then shows the document's business, date and amount above your question). Never mention the file
  name in the question itself - she does not know file names. Ask about the specific problem (the
  unclear date, amount, name or category), not whether to record it: a receipt or bill is obviously an
  expense, so never offer "לרשום כהוצאה" for it (only a document of type "other" gets that question).
  Options must be complete answers she can choose without thinking, e.g. likely categories or the
  likely value, plus "לא לרשום" - never a yes/no question such as "לדלג?". When you need a value only
  she can type (e.g. a business name, a date), ask without options. Act on her answer right away. If she gives no answer, do not record that
  document and mention it in the final report.
- If read_document returns an error (e.g. a blurry image), do not record anything for that file;
  tell her in the final report which files to upload again.
- A warning that the year may be misread gives "התאריך הסביר": ask with two options - that likely date
  first with " (מומלץ)", then the date that was read (she then clicks instead of typing).
- A warning that the total is handwritten: ask her to confirm it, with answer_type "amount" and the
  option "<amount> ₪ – נכון" (one click if right; otherwise she types the amount).
- Options are shown as buttons: do not list them again in the question text.
- A date and a handwritten amount that both need her: two questions, one for each (add_transaction
  records them only as she gave them).
- find_duplicate matched_by "same_file_content": the very same image is already recorded - skip it
  as a duplicate without asking.
- find_duplicate matched_by "same_business_and_date" (same business and day, another amount): maybe
  another photo of the same receipt, maybe a second purchase - ask her, naming both amounts, with the
  options "כן, אותה קבלה (לא לרשום)" and "לא, קבלה נפרדת". Same receipt -> skip_document.
- If the business name, the date or the amount is missing, ask her for that value directly (no
  options except "לא לרשום") - one question per missing value; for a date or an amount pass
  answer_type "date" / "amount" (the answer then comes back as YYYY-MM-DD / a number). Never invent a value: no placeholder such as "לא ידוע", no name guessed
  from a slogan or logo, and never today's date for a document without a date.
- Type: receipt and bill -> expense; a bill's category is "חשבונות בית". payslip -> income, category
  "משכורת". refund -> expense with a
  NEGATIVE amount, in the category of what was returned. "other" (not a proof of payment) ->
  ask her whether to record it, and explain why you ask: say what the document is and its amount,
  and why recording it may be wrong (e.g. an invoice is a demand for payment - the expense is
  recorded from the receipt when she pays, so recording both counts it twice; mention it if this
  business is already known). Options e.g. "לא לרשום – אעלה את הקבלה כשאשלם" / "לרשום בכל זאת".
- An "other" document telling her she RECEIVED something (a gift voucher, a gift card, a prize,
  cashback) is income in category "מתנות והטבות" - no money left her. Ask whether to record it,
  options e.g. "לרשום כהכנסה" / "לא לרשום". A purchase later paid with it is a normal expense, so
  the two cancel out. A receipt is never income - even for flowers or a gift she bought.
- Known business: if read_document returns "known_business", this business was recorded before
  (matched by its business number, or by its name) - record it with that business name and category,
  without asking. Pass the document's tax_id to add_transaction.
- Registry name: if read_document returns "business_printed", the business name was checked against
  the companies registry (found by its number): "business" is the legal name, or the familiar name
  with the legal name in brackets. Record "business" exactly as returned - it is more reliable than
  a name read from a photo. It may also tell what kind of business it is.
- Category: a known business (list below) gets its category without asking. Otherwise choose by
  WHAT WAS PAID FOR (the items / "עבור" line), not by words in a legal company name - e.g. "א.ג.
  נכסים והשקעות בע"מ" with the item "שיעור פסנתר" is a lesson, not housing. Use the category
  descriptions below. Choose without asking only if the document itself shows it: then pass
  category_evidence - a word copied exactly from the business name or an item name that shows the
  category (e.g. "פון" in "שוקי פון" -> electronics, "חלב" in the items -> food; any language
  counts - "Milk" in an English item -> food). Who the document is addressed to is not evidence.
  No such word, or the name and the items point to different categories -> ask (add_transaction
  refuses a new business without evidence found in the document).
  When you ask, offer 2-4 likely categories, your best guess first with " (מומלץ)" after it (record
  the category without that mark). "אחר" is not a way to avoid asking - add_transaction refuses
  "אחר" unless she chose it for that file.
  Every recorded business is remembered automatically for next time.
- If a tool returns ok=false, read the error: fix the values and try again, or ask her.
- Never calculate totals or budgets yourself - get_monthly_summary does that.
- When all documents are handled, call get_monthly_summary for each month (any year) in which you
  recorded a transaction during this run. If you recorded nothing, no summary.
- Explain your decisions, not your routine: before a tool call that is a decision, write one short
  sentence in Hebrew saying why (e.g. why you ask her, skip a file, read again, or chose a category
  without asking). Routine steps (listing the inbox, reading a document, checking duplicates,
  recording an obvious document, the summaries) need no sentence - the step itself shows what happens.
  "No sentence" is not "skip": every rule above still applies (always find_duplicate before recording).
  The sentence is shown to her: speak to her in the second person, feminine ("אשאל אותך על הקטגוריה,
  כי..."), never about her ("לשאול אותה", "שלה").

FINAL ANSWER (when you are done, answer without calling a tool), in Hebrew, usually one or two sentences:
- what was recorded (businesses and categories) - only if something was recorded; never say a
  document was recorded when it was skipped.
- ONLY IF a document was skipped: which and why.
- ONLY IF a file could not be read: which to upload again.
Leave out every part that did not happen - never write that nothing was skipped or no file needs uploading. Do not describe your process (the questions you asked) and do not repeat the
summary table - she already sees it.
Do not count documents or transactions and do not mention budgets - Python adds the exact numbers and
the budget warnings under your answer.

Expense categories (what belongs in each):
{expense_categories}
Income categories: {income_categories}
Monthly budgets ({currency}): {budgets}

Known businesses (business -> category), learned from earlier runs:
{known_businesses}"""

# Layer 2: the "menu" of tools. Parameter names are the exact names of the functions in tools.py.
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List the images in the inbox; each is marked already_recorded if it was recorded before "
                           "(copy_of: an exact copy of that recorded transaction's file).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Read one image and return its data: document_type, business, date (YYYY-MM-DD), "
                           "total, payments, items, plus 'warnings' (do not record blindly - ask) and 'info'.",
            "parameters": {
                "type": "object",
                "properties": {"file": {"type": "string", "description": "File name exactly as returned by list_documents"}},
                "required": ["file"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_duplicate",
            "description": "Return already-recorded transactions that may be the same document (same date and "
                           "amount, or same file). Compare the business names and decide if it is a duplicate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "amount": {"type": "number"},
                    "source_file": {"type": "string"},
                    "tax_id": {"type": "string", "description": "The business number (tax_id) from read_document, if any"},
                },
                "required": ["date", "amount", "source_file"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_transaction",
            "description": "Record one expense or income. A refund is an expense with a negative amount.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["expense", "income"]},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "amount": {"type": "number", "description": "In ₪. Payslip: the amount paid (לתשלום)"},
                    "category": {"type": "string", "description": "One of the categories for this type"},
                    "business": {"type": "string", "description": "Store, company or employer"},
                    "source_file": {"type": "string", "description": "The image file name"},
                    "items": {"type": "array", "items": {"type": "string"}, "description": "Item names (optional)"},
                    "tax_id": {"type": "string", "description": "The business number (tax_id) from read_document, if any"},
                    "category_evidence": {"type": "string", "description": "For a business that is not known, when she "
                                          "did not choose the category: a word copied from the business name or an "
                                          "item that shows the category"},
                },
                "required": ["type", "date", "amount", "category", "business", "source_file"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skip_document",
            "description": "Mark a document as handled without recording it (a duplicate, or she said not to record it), "
                           "so the next run does not handle it again.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "reason": {"type": "string", "description": "Short reason in Hebrew, e.g. 'כפילות של t003'"},
                },
                "required": ["file", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_monthly_summary",
            "description": "Income, expenses by category, balance and budget status (ok / near / over) for one month. "
                           "The summary is shown to the user.",
            "parameters": {
                "type": "object",
                "properties": {"month": {"type": "string", "description": "YYYY-MM"}},
                "required": ["month"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Ask the user a question (in Hebrew) when something is unclear. Returns her answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "file": {"type": "string", "description": "The document this question is about (one per question)"},
                    "options": {"type": "array", "items": {"type": "string"},
                                "description": "Optional likely answers (e.g. 2-4 categories); she picks one by number"},
                    "answer_type": {"type": "string", "enum": ["text", "date", "amount"],
                                    "description": "'date' or 'amount' when asking for a missing date or amount: "
                                                   "her screen then accepts only a valid one. Default 'text'"},
                },
                "required": ["question"],
            },
        },
    },
]

# Tools that are plain functions in tools.py (ask_user needs the UI, so it is handled in the loop)
_TOOL_FUNCTIONS = {
    "list_documents": tools.list_documents,
    "read_document": tools.read_document,
    "find_duplicate": tools.find_duplicate,
    "add_transaction": tools.add_transaction,
    "skip_document": tools.skip_document,
    "get_monthly_summary": tools.get_monthly_summary,
}


@dataclass
class _RunState:
    """What Python remembers during one run, to enforce rules the prompt alone did not keep."""
    recorded: int = 0
    skipped: int = 0
    sent_back: bool = False  # the model said "done" too early once and was sent back
    recorded_months: set[str] = field(default_factory=set)  # "YYYY-MM" of what was recorded
    summarized: set[str] = field(default_factory=set)       # months whose summary was shown
    warned: set[str] = field(default_factory=set)    # files whose reading had warnings
    # file -> the names read from a document of an unknown business (category evidence must be in them)
    evidence_text: dict[str, list[str]] = field(default_factory=dict)
    converted: dict[str, dict] = field(default_factory=dict)  # file -> {"total" in ₪, "original": ...}
    full_names: dict[str, str] = field(default_factory=dict)  # file -> business name with its legal name
    document_types: dict[str, str] = field(default_factory=dict)  # file -> document_type from read_document
    answers: dict[str, list[str]] = field(default_factory=dict)  # file -> her answers about it
    documents: dict[str, str] = field(default_factory=dict)  # file -> "business · date · amount"
    duplicate_checked: set[tuple] = field(default_factory=set)  # (date, amount) passed to find_duplicate
    recorded_categories: set[tuple] = field(default_factory=set)  # (month, category) of recorded expenses
    summaries: dict[str, dict] = field(default_factory=dict)  # month -> the summary she was shown
    copies: list[str] = field(default_factory=list)  # "file (copy of tNNN)" marked skipped in this run
    confirm: dict[str, list[str]] = field(default_factory=dict)  # file -> values she must confirm ("date", "amount")
    maybe_same: set[str] = field(default_factory=set)  # files that may be another photo of a recorded receipt
    undated: set[str] = field(default_factory=set)  # files on which no date at all was read


# Steps where the model decides something; its sentence before other steps only repeats the step title
_DECISION_TOOLS = {"add_transaction", "skip_document", "ask_user"}


def _is_decision(call, state: _RunState) -> bool:
    """Whether the model's sentence before this call is shown: a decision, or reading a document again."""
    if call.function.name in _DECISION_TOOLS:
        return True
    if call.function.name == "read_document":
        try:
            return json.loads(call.function.arguments or "{}").get("file") in state.documents
        except (json.JSONDecodeError, AttributeError):
            return False
    return False


def _system_prompt() -> str:
    """Layer 1 (goal and rules) + layer 1a (what the agent learned, from memory.json)."""
    today = calendar_date.today()
    known = storage.load_memory()["business_categories"]
    return SYSTEM_PROMPT.format(
        today=today.isoformat(),
        this_month=today.strftime("%Y-%m"),
        expense_categories="\n".join(f"- {c}: {config.CATEGORY_DESCRIPTIONS.get(c, '')}" for c in config.EXPENSE_CATEGORIES),
        income_categories=", ".join(config.INCOME_CATEGORIES),
        currency=config.CURRENCY,
        budgets=", ".join(f"{c}: {a}" for c, a in config.BUDGETS.items()),
        known_businesses="\n".join(f"- {b} -> {c}" for b, c in known.items()) or "(none yet)",
    )


def _describe_document(reading: dict) -> str:
    """One line that tells her which document a question is about (she does not know file names)."""
    parts = [reading.get("business") or "ללא שם עסק", reading.get("date") or "ללא תאריך"]
    if reading.get("total") is not None:
        # Not converted (no rate found): show its own currency, not ₪
        currency = config.CURRENCY if reading.get("currency") in (None, "ILS") else reading["currency"]
        parts.append(f"{reading['total']:,.2f} {currency}")
    return " · ".join(parts)


def _evidence_found(evidence, names: list[str]) -> bool:
    """Whether the model's category evidence is really a word from the document's business name or items."""
    def clean(text):
        return re.sub(r"[\s\"'׳״`.,()-]", "", str(text or "")).casefold()
    evidence = clean(evidence)
    return len(evidence) >= 2 and any(evidence in clean(name) for name in names)


def _date_answered(date, answers: list[str]) -> bool:
    """Whether this date is one she gave: a chosen option ("2026-05-29 (מומלץ)") or a typed date."""
    return isinstance(date, str) and any(date in a or tools.normalize_answer(a, "date") == date for a in answers)


def _amount_answered(amount, answers: list[str]) -> bool:
    """Whether this amount is one she gave: a chosen option ("640 ₪ – נכון") or a typed number."""
    if not isinstance(amount, (int, float)):
        return False
    said = [float(n.replace(",", "")) for a in answers for n in re.findall(r"\d[\d,]*(?:\.\d+)?", a)]
    return any(abs(abs(amount) - n) <= config.AMOUNT_TOLERANCE for n in said)


def _record_refusal(args: dict, state: _RunState) -> tuple[str, str] | None:
    """Rules checked in Python before recording (the model broke the first three in test runs).

    Returns (error for the model - what to do next, notice for her - one line in her words), or None.
    """
    file = args.get("source_file")
    if not isinstance(file, str):
        return None
    answers = state.answers.get(file, [])
    # A test run: after the gift-voucher rule was added, a flower-shop receipt was offered as income
    if args.get("type") == "income" and state.document_types.get(file) in ("receipt", "refund", "bill"):
        return ("קבלה, זיכוי או חשבון הם תמיד הוצאה (זיכוי – הוצאה בסכום שלילי), גם על מתנה שנקנתה. "
                "הכנסה היא רק תלוש שכר או הודעה על הטבה שהתקבלה. יש לרשום כהוצאה",
                "הרישום נעצר: קבלה על קנייה היא הוצאה, לא הכנסה")
    if copy_of := tools.recorded_copy(file):
        return (f"הקובץ '{file}' זהה בדיוק לקובץ של {copy_of} – זה אותו מסמך. יש לדלג עליו "
                f"(skip_document, סיבה: 'כפילות של {copy_of}') בלי לשאול",
                f"הרישום נעצר: זה אותו קובץ בדיוק כמו {copy_of}")
    if file in state.maybe_same and not answers:
        return ("ייתכן שזו אותה קבלה כמו תנועה רשומה (אותו עסק ואותו תאריך, סכום אחר) – יש לשאול את המשתמשת "
                "(ask_user עם file, האפשרויות 'כן, אותה קבלה (לא לרשום)' ו-'לא, קבלה נפרדת')",
                "הרישום נעצר: קודם שאלה אם זו אותה קבלה")
    # A test run: two warnings (handwritten amount and year), one question about the amount, and the
    # model chose the date itself - each value she must confirm has to come from her answers
    needed = state.confirm.get(file, [])
    if "date" in needed and not _date_answered(args.get("date"), answers):
        return ("התאריך חסר או לא ודאי – יש לשאול את המשתמשת על התאריך (ask_user עם file, answer_type 'date', "
                "התאריך הסביר ראשון עם ' (מומלץ)' אם יש כזה) ולרשום את התאריך שבחרה. אין להשתמש בתאריך של היום",
                "הרישום נעצר: קודם שאלה על התאריך")
    business = args.get("business")
    if "business" in needed and not (isinstance(business, str) and business.strip()
                                     and any(business.strip() in a or a.strip() in business for a in answers if a.strip())):
        return ("שם העסק לא נמצא במסמך – יש לשאול את המשתמשת מה שם העסק (ask_user עם file) ולרשום את השם "
                "שכתבה. אין להמציא שם",
                "הרישום נעצר: קודם שאלה על שם העסק")
    if "amount" in needed and not _amount_answered(args.get("amount"), answers):
        return ("הסכום כתוב ביד – יש לשאול את המשתמשת על הסכום (ask_user עם file, answer_type 'amount', "
                "האפשרות '<סכום> ₪ – נכון') ולרשום את הסכום שאישרה",
                "הרישום נעצר: קודם שאלה על הסכום")
    # A test run skipped find_duplicate and recorded a receipt twice - the prompt alone is not enough
    amount = args.get("amount")
    if isinstance(amount, (int, float)) and (args.get("date"), round(amount, 2)) not in state.duplicate_checked:
        return ("לפני הרישום יש לבדוק כפילות: find_duplicate עם אותו תאריך, סכום, source_file ו-tax_id",
                "הרישום נעצר: קודם בודקים שהמסמך לא נרשם כבר")
    if file in state.warned and not answers:
        return (f"בקריאת '{file}' היו אזהרות – יש לשאול את המשתמשת על הקובץ הזה "
                "(ask_user עם file) לפני הרישום",
                "הרישום נעצר: היו אזהרות בקריאה, ולכן קודם שאלה")
    # The prompt said "ask if not obvious", yet the model was always sure: "שוקי פרר" (no items) became
    # leisure, a travel agency became studies. Now it must point to a word in the document; asked at
    # most once per business - after it is recorded, memory knows its category.
    # Her answers count: a category she chose, or a business name she typed (evidence may come from it)
    chose_category = any(str(args.get("category")) in a for a in answers)
    if file in state.evidence_text and not chose_category \
            and not _evidence_found(args.get("category_evidence"), state.evidence_text[file] + answers):
        # A real run: she answered "רכב", the model recorded תחבורה without category_evidence and was
        # refused - then asked her the same question again. Her answer is the evidence; never re-ask.
        if answers:
            return (f"המשתמשת כבר ענתה על המסמך הזה ({' | '.join(answers)}). אין לשאול אותה שוב: יש לקרוא "
                    "שוב ל-add_transaction עם מילה מהתשובה שלה ב-category_evidence",
                    "הרישום נעצר: חסרה ראיה לקטגוריה, התשובה שלך תשמש כראיה")
        # A test run: a fish store was refused for a missing category_evidence and the model asked her
        # instead of sending the word "דגים" - the error now sends it back to look for the word first
        return (f"לא נמצאה ראיה לקטגוריה '{args.get('category')}': category_evidence חייב להיות מילה שמופיעה בשם "
                "העסק או בפריט ומראה את הקטגוריה, בכל שפה (גם מילה באנגלית בשם פריט). השם והפריטים במסמך: "
                f"{' | '.join(state.evidence_text[file])}. אם יש ביניהם מילה כזו – "
                "לקרוא שוב ל-add_transaction עם המילה "
                "ב-category_evidence. רק אם אין – לשאול את המשתמשת (ask_user עם file ו-2-4 קטגוריות, "
                "הניחוש שלך ראשון עם ' (מומלץ)')",
                "הרישום נעצר: הקטגוריה צריכה להתבסס על מילה מהמסמך")
    # A document in dollars: the ₪ amount from read_document must be recorded, not the printed one
    converted = state.converted.get(file)
    if converted and isinstance(amount, (int, float)) and converted["original"]["total"] is not None \
            and abs(abs(amount) - abs(converted["original"]["total"])) <= config.AMOUNT_TOLERANCE:
        original = converted["original"]
        return (f"הסכום {amount} הוא ב-{original['currency']}. יש לרשום בשקלים: {converted['total']} "
                f"(לפי שער {original['rate']})",
                f"הרישום נעצר: הסכום היה ב-{original['currency']}, צריך לרשום בשקלים")
    if args.get("category") == "אחר" and not any("אחר" in a for a in answers):
        return ("'אחר' רק אם המשתמשת בחרה בו – יש לשאול אותה על הקטגוריה (ask_user עם file "
                "ו-2-4 קטגוריות אפשריות)",
                "הרישום נעצר: 'אחר' רק אם בחרת בו")
    return None


def _run_tool(name: str, raw_args: str, ui: AgentUI, state: _RunState) -> dict:
    """Run one tool the model asked for. Mistakes by the model come back as an error it can read."""
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        return {"ok": False, "error": f"הפרמטרים של {name} אינם JSON תקין"}
    if not isinstance(args, dict):
        return {"ok": False, "error": f"הפרמטרים של {name} חייבים להיות אובייקט"}
    # A run showed an empty summary of a month with nothing recorded - refused before she sees it
    if name == "get_monthly_summary" and args.get("month") not in _wanted_summaries(state):
        # "hidden": the refusal is for the model only - she would see a red error about a summary she never asked for
        return {"ok": False, "hidden": True,
                "error": "אין צורך בסיכום של החודש הזה – רק החודש הנוכחי וחודשים אחרונים "
                         f"שנרשם בהם משהו בריצה הזו: {', '.join(sorted(_wanted_summaries(state)))}"}
    ui.tool_call(name, args)

    if name == "ask_user":
        question = args.get("question")
        if not isinstance(question, str) or not question.strip():
            return {"ok": False, "error": "חסרה שאלה"}
        file = args.get("file") if isinstance(args.get("file"), str) else None
        if file in state.documents:
            question = f"מסמך: {state.documents[file]}\n\n{question}"
        answer_type = args.get("answer_type") if args.get("answer_type") in ("date", "amount") else "text"
        options = [o for o in (args.get("options") or []) if isinstance(o, str) and o.strip()]
        # A real run: no date was read on a handwritten receipt, and the model offered today's date (also
        # in the screenshot's file name) twice, once as "(מומלץ)" - one click would record a wrong date.
        # With nothing read there is nothing to suggest: only options without digits ("לא לרשום") stay.
        if answer_type == "date" and file in state.undated:
            options = [o for o in options if not re.search(r"\d", o)]
        unique, seen = [], set()  # the same option twice ("X (מומלץ)" and "X") is shown once
        for option in options:
            if (key := option.replace("(מומלץ)", "").strip()) not in seen:
                seen.add(key)
                unique.append(option)
        options = unique
        answer = ui.ask(question, options, answer_type)
        # The screen accepts only a valid date / amount; checked here too, for any UI. An option
        # (e.g. "לא לרשום") is taken as is.
        if answer and answer_type != "text" and answer not in options:
            answer = tools.normalize_answer(answer, answer_type)
            if answer is None:
                return {"ok": True, "answer": "", "note": "התשובה לא הייתה תאריך / סכום תקין"}
        if not answer:
            return {"ok": True, "answer": "", "note": "המשתמשת לא ענתה"}
        if file:
            state.answers.setdefault(file, []).append(answer)
        return {"ok": True, "answer": answer}

    if name == "add_transaction" and (refusal := _record_refusal(args, state)):
        return {"ok": False, "error": refusal[0], "notice": refusal[1]}
    if name == "add_transaction":
        args = {k: v for k, v in args.items() if k != "category_evidence"}  # checked above, not saved
    # A real run: read_document gave "אושר עד גבעת שאול (מרב - מזון כל בע"מ)" and the model recorded
    # only "אושר עד גבעת שאול". A shortened name gets its full form back; a different one (her answer) stays.
    full_name = state.full_names.get(args.get("source_file"))
    if name == "add_transaction" and full_name and isinstance(args.get("business"), str) \
            and args["business"].strip() and args["business"].strip() in full_name:
        args = {**args, "business": full_name}
    if name == "add_transaction" and (converted := state.converted.get(args.get("source_file"))):
        args = {**args, "original": converted["original"]}  # kept in the transaction, not the model's job

    function = _TOOL_FUNCTIONS.get(name)
    if function is None:
        return {"ok": False, "error": f"אין כלי בשם '{name}'. הכלים: {', '.join(_TOOL_FUNCTIONS)}, ask_user"}
    try:
        result = function(**args)
    except TypeError as e:  # missing or unknown parameter name
        return {"ok": False, "error": f"פרמטרים שגויים ל-{name}: {e}"}
    if name == "find_duplicate" and result.get("ok") and isinstance(args.get("amount"), (int, float)):
        state.duplicate_checked.add((args.get("date"), round(args["amount"], 2)))
        if any(c["matched_by"] == "same_business_and_date" for c in result["candidates"]):
            state.maybe_same.add(args.get("source_file"))
    if name == "list_documents" and result.get("ok"):
        # An exact copy of a recorded file is skipped by Python, once - no reading, no question
        for document in result["documents"]:
            if document.get("copy_of") and "skipped" not in document:
                tools.skip_document(document["file"], f"עותק של {document['copy_of']}")
                document["skipped"] = f"עותק של {document['copy_of']}"
                state.copies.append(f"{document['file']} (עותק של {document['copy_of']}, שכבר רשום)")
                result.setdefault("copies_now", []).append(document)
        result["new_count"] = sum(not d["already_recorded"] and "skipped" not in d for d in result["documents"])
    if name == "read_document" and result.get("ok"):
        state.documents[result["file"]] = _describe_document(result)
        state.document_types[result["file"]] = result.get("document_type")
        if result.get("date_printed") is None:
            state.undated.add(result["file"])
        if result.get("warnings"):
            state.warned.add(result["file"])
        if result.get("confirm"):
            state.confirm[result["file"]] = result["confirm"]
        if not result.get("business"):  # after the registry and memory had their chance
            # A test run: no name was read, she answered "לרשום בכל זאת", and "שובר מתנה" was recorded as the business
            state.confirm.setdefault(result["file"], []).append("business")
        if result.get("document_type") in ("receipt", "refund") and not result.get("known_business"):
            names = [result.get("business"), result.get("business_printed")] + [i["name"] for i in result.get("items", [])]
            state.evidence_text[result["file"]] = [n for n in names if n]
        if result.get("original"):
            state.converted[result["file"]] = {"total": result["total"], "original": result["original"]}
        if result.get("business_printed") or result.get("known_business"):
            state.full_names[result["file"]] = (result.get("known_business") or {}).get("business") or result["business"]
    return result


class AgentStopped(Exception):
    """The run stopped before its goal (API problem or Ctrl+C). The message says why and what was saved."""


def _api_error_message(error: openai.OpenAIError) -> str:
    """A Hebrew explanation of an API problem, with what to do about it."""
    if isinstance(error, openai.APITimeoutError):  # before APIConnectionError: it is a subclass
        return "המודל לא ענה בזמן – ייתכן שהחיבור לאינטרנט איטי. נסי שוב בעוד כמה דקות."
    if isinstance(error, openai.APIConnectionError):
        return "אין חיבור לשרת של OpenAI. בדקי את החיבור לאינטרנט ונסי שוב."
    if isinstance(error, openai.AuthenticationError):
        return "מפתח ה-API לא תקין. בדקי את משתנה הסביבה OPENAI_API_KEY (ראו README)."
    if isinstance(error, openai.RateLimitError):
        if error.code == "insufficient_quota":
            return "נגמר הקרדיט בחשבון OpenAI. יש להטעין יתרה ב-platform.openai.com ולהריץ שוב."
        return "יותר מדי בקשות ל-OpenAI בזמן קצר. נסי שוב בעוד דקה."
    if isinstance(error, (openai.NotFoundError, openai.PermissionDeniedError)):
        return f"אין גישה למודל ({config.MODEL} / {config.READ_MODEL}). בדקי את MODEL ו-READ_MODEL ב-config.py."
    if isinstance(error, openai.InternalServerError):
        return "תקלה זמנית בשרתים של OpenAI. נסי שוב מאוחר יותר."
    return f"שגיאה בפנייה למודל: {error}"


def _not_handled() -> list[str]:
    """Inbox files neither recorded nor skipped on purpose - computed by Python, not the model."""
    documents = tools.list_documents()
    if not documents.get("ok"):
        return []
    return [d["file"] for d in documents["documents"] if not d["already_recorded"] and "skipped" not in d]


def _progress_report(recorded: int) -> str:
    """What was done before a stop."""
    text = f"נרשמו בריצה הזו: {recorded} תנועות (הן שמורות)."
    if left := _not_handled():
        text += "\nלא טופלו עדיין: " + ", ".join(left) + "\nאפשר להריץ שוב כדי להמשיך."
    return text


def _wanted_summaries(state: _RunState) -> set[str]:
    """The months in which something was recorded in this run (any year); none if nothing was.

    Her choice (a test run recorded a 2022 receipt and then showed October 2026): the summary is
    about what was just added.
    """
    return set(state.recorded_months)


def _show_missing_summaries(ui: AgentUI, state: _RunState) -> None:
    """Show the summaries the model should have asked for (a test run showed only 1 of the months)."""
    for month in sorted(_wanted_summaries(state) - state.summarized):
        summary = tools.get_monthly_summary(month)
        if summary.get("ok"):
            ui.show_summary(summary)
            state.summaries[month] = summary


def _nothing_new_message(state: _RunState) -> str:
    """The final message of a run that recorded and skipped nothing - written by Python."""
    if state.copies:  # she did upload a file - say what it is first
        return "\n".join(f"הקובץ {c} – לכן לא נרשם שוב." for c in state.copies)
    return "אין מסמכים חדשים בתיבה – לא נרשם דבר."


def _run_facts(state: _RunState) -> str:
    """The exact numbers of this run - from Python, not the model (a test run said 8 when 11 were recorded)."""
    # The count only when more than one document was handled - for one, the model's sentence already says it
    text = ""
    if state.recorded + state.skipped > 1:
        text = f"נרשמו בריצה הזו: {state.recorded}" + (f" · דולגו: {state.skipped}" if state.skipped else "")
    # Budgets from Python, only where something was recorded now (the model mentioned an old
    # clothing overrun after recording a lesson)
    for month, summary in sorted(state.summaries.items()):
        for budget in summary.get("budgets", []):
            if budget["status"] in ("near", "over") and (month, budget["category"]) in state.recorded_categories:
                label = "חריגה מהתקציב" if budget["status"] == "over" else "קרוב לתקציב"
                text += f"\n{label}: {budget['category']} ({budget['percent']}% ב-{month[5:]}/{month[:4]})"
    # A safety net: in run 2 the model forgot documents she had approved
    if left := _not_handled():
        text += "\nקבצים שלא נרשמו ולא דולגו (לבדוק): " + ", ".join(left)
    return text


def run(ui: AgentUI) -> None:
    """Run the agent until it reaches its goal or the step limit.

    Raises AgentStopped (API problem or Ctrl+C) and storage.StorageError (damaged data file);
    the caller shows them to the user.
    """
    state = _RunState()  # outside _loop, so the count survives an exception
    try:
        _loop(ui, state)
    except (KeyboardInterrupt, EOFError):  # Ctrl+C (or Ctrl+Z at a question)
        raise AgentStopped("עצרת את הריצה.\n" + _progress_report(state.recorded)) from None
    except openai.OpenAIError as e:
        raise AgentStopped(_api_error_message(e) + "\n" + _progress_report(state.recorded)) from e


def _loop(ui: AgentUI, state: _RunState) -> None:
    client = openai.OpenAI(timeout=config.MODEL_TIMEOUT_SECONDS)
    messages = [{"role": "system", "content": _system_prompt()}]

    for _ in range(config.MAX_STEPS):
        response = client.chat.completions.create(model=config.MODEL, messages=messages, tools=TOOL_DEFINITIONS)
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:  # no tool chosen: the model says it is done
            # Stop condition checked by Python: in a test run the model "finished" after 2 of 15
            # documents. Sent back once - some files rightly stay open (unreadable, no answer).
            left = _not_handled()
            if left and not state.sent_back:
                state.sent_back = True
                messages.append({"role": "user", "content": (
                    f"עוד לא סיימת: {len(left)} קבצים לא נרשמו ולא דולגו: {', '.join(left)}. "
                    "המשך לטפל בהם. אם אי אפשר לטפל באחד מהם עכשיו (תמונה לא קריאה, לא התקבלה תשובה) – "
                    "כתוב זאת בהודעת הסיום.")})
                continue
            _show_missing_summaries(ui, state)
            facts = _run_facts(state).strip()
            if state.recorded or state.skipped:
                ui.final((message.content or "סיימתי.") + (f"\n\n{facts}" if facts else ""))
            else:
                # Nothing new: a test run's model then described every older document as if just recorded
                ui.final(_nothing_new_message(state) + (f"\n\n{facts}" if facts else ""))
            return

        if any(_is_decision(call, state) for call in message.tool_calls):
            ui.thinking(message.content or "")
        for call in message.tool_calls:
            name = call.function.name
            result = _run_tool(name, call.function.arguments, ui, state)
            notice = result.pop("notice", None)
            if notice:
                ui.guard(notice)
            elif not result.pop("hidden", False):
                ui.tool_result(name, result)
            if result.get("ok") and name == "get_monthly_summary":
                ui.show_summary(result)
                state.summarized.add(result["month"])
                state.summaries[result["month"]] = result
            if result.get("ok") and name == "add_transaction":
                state.recorded += 1
                state.recorded_months.add(result["transaction"]["date"][:7])
                if result["transaction"]["type"] == "expense":
                    state.recorded_categories.add((result["transaction"]["date"][:7], result["transaction"]["category"]))
            if result.get("ok") and name == "skip_document":
                state.skipped += 1
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    ui.final(f"עצרתי: הגעתי למגבלה של {config.MAX_STEPS} צעדים לפני שסיימתי.\n" + _progress_report(state.recorded))
