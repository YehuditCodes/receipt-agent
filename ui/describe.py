"""One readable Hebrew line per tool result - shared by the terminal and the web UIs."""

import config

DOCUMENT_TYPES = {"receipt": "קבלה", "bill": "חשבון", "payslip": "תלוש שכר", "refund": "זיכוי", "other": "לא מסמך כספי"}
TRANSACTION_TYPES = {"expense": "הוצאה", "income": "הכנסה"}


def money(amount: float) -> str:
    return f"{amount:,.2f} {config.CURRENCY}"


def describe_result(name: str, result: dict) -> str:
    """What a successful tool result means, in one line (errors are shown by each UI)."""
    if name == "list_documents":
        new = "מסמך חדש אחד" if result["new_count"] == 1 else f"{result['new_count']} חדשים"
        text = f"נמצאו {len(result['documents'])} מסמכים, מתוכם {new}"
        if result.get("unsupported_files"):
            text += f" (לא נתמכים: {', '.join(result['unsupported_files'])})"
        # A copy of a recorded file found in this run - otherwise "0 new" alone does not say why
        for copy in result.get("copies_now", []):
            text += f" · {copy['file']} הוא עותק של {copy['copy_of']} (כבר רשום)"
        return text
    if name == "read_document":
        kind = DOCUMENT_TYPES.get(result.get("document_type"), result.get("document_type"))
        business = result.get("business") or "ללא שם עסק"
        date = result.get("date") or result.get("date_printed") or "ללא תאריך"
        total = money(result["total"]) if result.get("total") is not None else "ללא סכום"
        text = f"{kind} | {business} | {date} | {total}"
        if known := result.get("known_business"):
            by = "לפי ח.פ." if known["matched_by"] == "tax_id" else "לפי השם"
            text += f" · עסק מוכר ({by}): {known['category']}"
        return text
    if name == "add_transaction":
        t = result["transaction"]
        text = (f"נרשם {t['id']}: {TRANSACTION_TYPES.get(t['type'], t['type'])} {money(t['amount'])}"
                f" – {t['category']} ({t['business']})")
        if result.get("learned_business"):
            text += " · נשמר בזיכרון לפעם הבאה"
        return text
    if name == "find_duplicate":
        if not result["possible_duplicate"]:
            return "לא נמצאה כפילות"
        found = ", ".join(f"{c.get('id')} {c.get('business', '')}" for c in result["candidates"])
        return f"נמצאו תנועות דומות: {found}"
    if name == "skip_document":
        return f"דולג: {result['reason']}"
    if name == "get_monthly_summary":
        return f"סיכום חודש {result['month']}: {result['transaction_count']} תנועות"
    return name
