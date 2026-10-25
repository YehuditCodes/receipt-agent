"""The agent loop: the model decides the next step, Python runs the tool, the result goes back.

The agent never prints or reads input itself - everything goes through an AgentUI, so the
same loop can run with a terminal or a web UI.

The model gets a goal and rules (SYSTEM_PROMPT), not a fixed order of steps: it decides
which tool to call, reads the result and decides what to do next. It stops when it answers
without calling a tool (goal reached) or after config.MAX_STEPS model calls.
"""

import json
from datetime import date as calendar_date

import openai

import config
import storage
import tools
from ui.base import AgentUI

SYSTEM_PROMPT = """You are an expense-tracking agent for one person in Israel. Today is {today}.

GOAL: process every new document (receipt, bill, payslip, refund) in her inbox, record each one
correctly exactly once, then show her the monthly summary and report what happened.
You decide the order of the steps yourself, using the tools.

RULES
- Documents marked already_recorded are done - do not read them again.
- Before recording a document, check that it is not already recorded (find_duplicate). The
  candidates may be the same document with a slightly different business name ("שופרסל" /
  "שופרסל דיל") - compare and decide. If you are not sure, ask her.
- read_document "warnings" mean the reading may be wrong: do not record blindly. Ask her, say
  what looks wrong, and record only what she confirms (or skip if she says so).
  "info" is for your knowledge only and does not block recording.
- If read_document returns an error (e.g. a blurry image), do not record anything for that file;
  tell her in the final report which files to upload again.
- If the business name is missing, ask her for it. Never record a placeholder such as "לא ידוע"
  and never guess the name from a slogan or logo.
- Type: receipt and bill -> expense. payslip -> income, category "משכורת". refund -> expense with a
  NEGATIVE amount, in the category of what was returned. "other" (not a financial document) ->
  ask her whether to skip it.
- Category: a known business (list below) gets its category without asking. Otherwise choose
  from the business name and the items only if it is obvious. If not obvious, ask her with
  ask_user and offer 2-4 likely categories as options; after she answers, call remember_business
  so you will not ask again.
- If a tool returns ok=false, read the error: fix the values and try again, or ask her.
- Never calculate totals or budgets yourself - get_monthly_summary does that.
- When all documents are handled, call get_monthly_summary once for EACH month (YYYY-MM of the date)
  in which you recorded a transaction during this run - e.g. records dated 2026-09-10 and 2010-03-13
  need two calls. If nothing new was recorded, call it for the current month: {this_month}.
- Before each tool call, write one short sentence in Hebrew saying what you are doing and why.

FINAL ANSWER (when you are done, answer without calling a tool), in Hebrew, short:
what was recorded, what was skipped and why, files to upload again, and any budget whose status
is near (say "קרוב לתקציב") or over (say "חריגה מהתקציב"). Do not repeat the whole summary table - she already sees it.

Expense categories: {expense_categories}
Income categories: {income_categories}
Monthly budgets ({currency}): {budgets}

Known businesses (business -> category), learned from her earlier answers:
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
                },
                "required": ["type", "date", "amount", "category", "business", "source_file"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_business",
            "description": "Remember a business's category for next time. Call only after the user confirmed the category.",
            "parameters": {
                "type": "object",
                "properties": {"business": {"type": "string"}, "category": {"type": "string"}},
                "required": ["business", "category"],
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
    "remember_business": tools.remember_business,
    "get_monthly_summary": tools.get_monthly_summary,
}


def _system_prompt() -> str:
    """Layer 1 (goal and rules) + layer 1a (what the agent learned, from memory.json)."""
    today = calendar_date.today()
    known = storage.load_memory()["business_categories"]
    return SYSTEM_PROMPT.format(
        today=today.isoformat(),
        this_month=today.strftime("%Y-%m"),
        expense_categories=", ".join(config.EXPENSE_CATEGORIES),
        income_categories=", ".join(config.INCOME_CATEGORIES),
        currency=config.CURRENCY,
        budgets=", ".join(f"{c}: {a}" for c, a in config.BUDGETS.items()),
        known_businesses="\n".join(f"- {b} -> {c}" for b, c in known.items()) or "(none yet)",
    )


def _run_tool(name: str, raw_args: str, ui: AgentUI) -> dict:
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
        answer = ui.ask(question, args.get("options"))
        if not answer:
            return {"ok": True, "answer": "", "note": "המשתמשת לא ענתה"}
        return {"ok": True, "answer": answer}

    function = _TOOL_FUNCTIONS.get(name)
    if function is None:
        return {"ok": False, "error": f"אין כלי בשם '{name}'. הכלים: {', '.join(_TOOL_FUNCTIONS)}, ask_user"}
    try:
        return function(**args)
    except TypeError as e:  # missing or unknown parameter name
        return {"ok": False, "error": f"פרמטרים שגויים ל-{name}: {e}"}


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


def _progress_report(recorded: int) -> str:
    """What was done before a stop - computed by Python, not the model."""
    text = f"נרשמו בריצה הזו: {recorded} תנועות (הן שמורות)."
    documents = tools.list_documents()
    if documents.get("ok"):
        left = [d["file"] for d in documents["documents"] if not d["already_recorded"]]
        if left:
            text += "\nלא טופלו עדיין: " + ", ".join(left) + "\nאפשר להריץ שוב כדי להמשיך."
    return text


def run(ui: AgentUI) -> None:
    """Run the agent until it reaches its goal or the step limit.

    Raises AgentStopped (API problem or Ctrl+C) and storage.StorageError (damaged data file);
    the caller shows them to the user.
    """
    recorded = [0]  # a list so _loop can update it and the count survives an exception
    try:
        _loop(ui, recorded)
    except (KeyboardInterrupt, EOFError):  # Ctrl+C (or Ctrl+Z at a question)
        raise AgentStopped("עצרת את הריצה.\n" + _progress_report(recorded[0])) from None
    except openai.OpenAIError as e:
        raise AgentStopped(_api_error_message(e) + "\n" + _progress_report(recorded[0])) from e


def _loop(ui: AgentUI, recorded: list[int]) -> None:
    client = openai.OpenAI(timeout=config.MODEL_TIMEOUT_SECONDS)
    messages = [{"role": "system", "content": _system_prompt()}]

    for _ in range(config.MAX_STEPS):
        response = client.chat.completions.create(model=config.MODEL, messages=messages, tools=TOOL_DEFINITIONS)
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:  # no tool chosen: the model says it is done
            ui.final(message.content or "סיימתי.")
            return

        ui.thinking(message.content or "")
        for call in message.tool_calls:
            name = call.function.name
            result = _run_tool(name, call.function.arguments, ui)
            ui.tool_result(name, result)
            if result.get("ok") and name == "get_monthly_summary":
                ui.show_summary(result)
            if result.get("ok") and name == "add_transaction":
                recorded[0] += 1
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    ui.final(f"עצרתי: הגעתי למגבלה של {config.MAX_STEPS} צעדים לפני שסיימתי.\n" + _progress_report(recorded[0]))
