"""The agent loop: the model decides the next step, Python runs the tool, the result goes back.

The agent never prints or reads input itself - everything goes through an AgentUI, so the
same loop can run with a terminal or a web UI.

The model gets a goal and rules (SYSTEM_PROMPT), not a fixed order of steps: it decides
which tool to call, reads the result and decides what to do next. It stops when it answers
without calling a tool (goal reached) or after config.MAX_STEPS model calls.
"""

import json
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
  "info" is for your knowledge only and does not block recording.
- Questions: ask about ONE document per ask_user call and pass its file name in "file" (her screen
  then shows the document's business, date and amount above your question). Never mention the file
  name in the question itself - she does not know file names. Options must be complete actions she
  can choose without thinking, e.g. "לרשום כהוצאה" / "לא לרשום", or likely categories - never a yes/no
  question such as "לדלג?". When you need a value only she can type (e.g. a business name, a date),
  ask without options. Act on her answer right away. If she gives no answer, do not record that
  document and mention it in the final report.
- If read_document returns an error (e.g. a blurry image), do not record anything for that file;
  tell her in the final report which files to upload again.
- If the business name, the date or the amount is missing, ask her for that value directly (no
  options except "לא לרשום"). Never invent a value: no placeholder such as "לא ידוע", no name guessed
  from a slogan or logo, and never today's date for a document without a date.
- Type: receipt and bill -> expense; a bill's category is "חשבונות בית". payslip -> income, category
  "משכורת". refund -> expense with a
  NEGATIVE amount, in the category of what was returned. "other" (not a proof of payment) ->
  ask her whether to record it.
- Known business: if read_document returns "known_business", this business was recorded before
  (matched by its business number, or by its name) - record it with that business name and category,
  without asking. Pass the document's tax_id to add_transaction.
- Category: a known business (list below) gets its category without asking. Otherwise choose by
  WHAT WAS PAID FOR (the items / "עבור" line), not by words in a legal company name - e.g. "א.ג.
  נכסים והשקעות בע"מ" with the item "שיעור פסנתר" is a lesson, not housing. Use the category
  descriptions below. Choose without asking only if it is obvious (a donation receipt -> "תרומות");
  if the name and the items point to different categories, or a short name could mean several
  things and there are no items, ask.
  If not obvious, ask her with ask_user and offer 2-4 likely categories as options. "אחר" is not a
  way to avoid asking - add_transaction refuses "אחר" unless she chose it for that file.
  Every recorded business is remembered automatically for next time.
- If a tool returns ok=false, read the error: fix the values and try again, or ask her.
- Never calculate totals or budgets yourself - get_monthly_summary does that.
- When all documents are handled, call get_monthly_summary for the current month ({this_month}) and
  for each of these recent months in which you recorded a transaction during this run: {recent_months}.
  Older documents are recorded as usual, without a summary.
- Before each tool call, write one short sentence in Hebrew saying what you are doing and why.

FINAL ANSWER (when you are done, answer without calling a tool), in Hebrew, short:
what was recorded, what was skipped and why, files to upload again, and any budget whose status
is near (say "קרוב לתקציב") or over (say "חריגה מהתקציב"). Do not repeat the whole summary table - she already sees it.
Do not count documents or transactions - Python adds the exact numbers under your answer.

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
            "description": "List the images in the inbox; each is marked already_recorded if it was recorded before.",
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
    answers: dict[str, list[str]] = field(default_factory=dict)  # file -> her answers about it
    documents: dict[str, str] = field(default_factory=dict)  # file -> "business · date · amount"


def _recent_months() -> list[str]:
    """The current month and the ones before it ("YYYY-MM"), config.SUMMARY_RECENT_MONTHS in all."""
    today = calendar_date.today()
    months = []
    for back in range(config.SUMMARY_RECENT_MONTHS):
        year, month = divmod(today.year * 12 + today.month - 1 - back, 12)
        months.append(f"{year}-{month + 1:02d}")
    return months


def _system_prompt() -> str:
    """Layer 1 (goal and rules) + layer 1a (what the agent learned, from memory.json)."""
    today = calendar_date.today()
    known = storage.load_memory()["business_categories"]
    return SYSTEM_PROMPT.format(
        today=today.isoformat(),
        this_month=today.strftime("%Y-%m"),
        recent_months=", ".join(_recent_months()),
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
        parts.append(f"{reading['total']:,.2f} {config.CURRENCY}")
    return " · ".join(parts)


def _record_refusal(args: dict, state: _RunState) -> str | None:
    """Rules checked in Python before recording: the model broke both of them in test runs."""
    file = args.get("source_file")
    if not isinstance(file, str):
        return None
    answers = state.answers.get(file, [])
    if file in state.warned and not answers:
        return (f"בקריאת '{file}' היו אזהרות – יש לשאול את המשתמשת על הקובץ הזה "
                "(ask_user עם file) לפני הרישום")
    if args.get("category") == "אחר" and not any("אחר" in a for a in answers):
        return ("'אחר' רק אם המשתמשת בחרה בו – יש לשאול אותה על הקטגוריה (ask_user עם file "
                "ו-2-4 קטגוריות אפשריות)")
    return None


def _run_tool(name: str, raw_args: str, ui: AgentUI, state: _RunState) -> dict:
    """Run one tool the model asked for. Mistakes by the model come back as an error it can read."""
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        return {"ok": False, "error": f"הפרמטרים של {name} אינם JSON תקין"}
    if not isinstance(args, dict):
        return {"ok": False, "error": f"הפרמטרים של {name} חייבים להיות אובייקט"}
    ui.tool_call(name, args)

    if name == "ask_user":
        question = args.get("question")
        if not isinstance(question, str) or not question.strip():
            return {"ok": False, "error": "חסרה שאלה"}
        file = args.get("file") if isinstance(args.get("file"), str) else None
        if file in state.documents:
            question = f"מסמך: {state.documents[file]}\n\n{question}"
        answer = ui.ask(question, args.get("options"))
        if not answer:
            return {"ok": True, "answer": "", "note": "המשתמשת לא ענתה"}
        if file:
            state.answers.setdefault(file, []).append(answer)
        return {"ok": True, "answer": answer}

    if name == "add_transaction" and (refusal := _record_refusal(args, state)):
        return {"ok": False, "error": refusal}

    function = _TOOL_FUNCTIONS.get(name)
    if function is None:
        return {"ok": False, "error": f"אין כלי בשם '{name}'. הכלים: {', '.join(_TOOL_FUNCTIONS)}, ask_user"}
    try:
        result = function(**args)
    except TypeError as e:  # missing or unknown parameter name
        return {"ok": False, "error": f"פרמטרים שגויים ל-{name}: {e}"}
    if name == "read_document" and result.get("ok"):
        state.documents[result["file"]] = _describe_document(result)
        if result.get("warnings"):
            state.warned.add(result["file"])
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


def _show_missing_summaries(ui: AgentUI, state: _RunState) -> None:
    """Show the summaries the model should have asked for (a test run showed only 1 of the months)."""
    recent = _recent_months()
    wanted = {recent[0]} | (state.recorded_months & set(recent))
    for month in sorted(wanted - state.summarized):
        summary = tools.get_monthly_summary(month)
        if summary.get("ok"):
            ui.show_summary(summary)


def _run_facts(state: _RunState) -> str:
    """The exact numbers of this run - from Python, not the model (a test run said 8 when 11 were recorded)."""
    text = f"נרשמו בריצה הזו: {state.recorded} · דולגו: {state.skipped}"
    if older := sorted(state.recorded_months - set(_recent_months())):
        text += "\nנרשמו גם מסמכים מחודשים קודמים (בלי טבלת סיכום): " + ", ".join(older)
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
            ui.final((message.content or "סיימתי.") + "\n\n" + _run_facts(state))
            return

        ui.thinking(message.content or "")
        for call in message.tool_calls:
            name = call.function.name
            result = _run_tool(name, call.function.arguments, ui, state)
            ui.tool_result(name, result)
            if result.get("ok") and name == "get_monthly_summary":
                ui.show_summary(result)
                state.summarized.add(result["month"])
            if result.get("ok") and name == "add_transaction":
                state.recorded += 1
                state.recorded_months.add(result["transaction"]["date"][:7])
            if result.get("ok") and name == "skip_document":
                state.skipped += 1
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    ui.final(f"עצרתי: הגעתי למגבלה של {config.MAX_STEPS} צעדים לפני שסיימתי.\n" + _progress_report(state.recorded))
