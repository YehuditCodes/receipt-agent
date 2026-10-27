"""The tools' own checks (tools.py): validation, duplicates, memory and the monthly summary."""

import tools

TAX_ID = "513461053"  # a valid company number (check digit), from sample 01


def record(file, amount=100.0, date="2026-09-09", category="מזון וקניות", type="expense", **extra):
    return tools.add_transaction(type=type, date=date, amount=amount, category=category,
                                 business="אושר עד", source_file=file, **extra)


# --- Her answers to date / amount questions ---

def test_date_answer_accepts_iso_and_printed_forms():
    assert tools.normalize_answer("2026-09-09", "date") == "2026-09-09"
    assert tools.normalize_answer("09/09/26", "date") == "2026-09-09"
    assert tools.normalize_answer("08-Sep-26", "date") == "2026-09-08"


def test_date_answer_rejects_nonsense_and_future():
    # A test run: she typed "מ" and "222222" for a date
    assert tools.normalize_answer("מ", "date") is None
    assert tools.normalize_answer("222222", "date") is None
    assert tools.normalize_answer("2099-01-01", "date") is None


def test_amount_answer():
    assert tools.normalize_answer("₪ 1,200.50", "amount") == "1200.5"
    assert tools.normalize_answer("80 ש\"ח", "amount") == "80.0"
    assert tools.normalize_answer("0", "amount") is None
    assert tools.normalize_answer("abc", "amount") is None


# --- Business number ---

def test_tax_id_check_digit():
    assert tools._valid_tax_id(TAX_ID) == TAX_ID
    assert tools._valid_tax_id("51-346105-3") == TAX_ID
    assert tools._valid_tax_id("513461054") is None  # one misread digit
    assert tools._valid_tax_id("123") is None


# --- Recording ---

def test_record_rejects_bad_values(add_file):
    file = add_file("a.jpg")
    assert not record(file, category="קטגוריה שלא קיימת")["ok"]
    assert not record(file, amount=0)["ok"]
    assert not record(file, date="2099-01-01")["ok"]
    assert not record(file, type="income", amount=-50, category="משכורת")["ok"]
    assert not record("not-in-inbox.jpg")["ok"]


def test_same_file_is_never_recorded_twice(add_file):
    file = add_file("a.jpg")
    assert record(file)["ok"]
    assert not record(file, amount=200)["ok"]


def test_skipped_file_cannot_be_recorded(add_file):
    file = add_file("a.jpg")
    assert tools.skip_document(file, "הזמנה, לא קבלה")["ok"]
    assert not record(file)["ok"]


def test_recorded_file_cannot_be_skipped(add_file):
    # Check 10: a test run recorded 08.jpg and later "skipped" it as a duplicate
    file = add_file("08.jpg")
    assert record(file)["ok"]
    result = tools.skip_document(file, "כפילות של 01.jpg")
    assert not result["ok"]
    assert "כבר נרשם" in result["error"]


def test_recording_remembers_the_business_by_number(add_file):
    record(add_file("a.jpg"), tax_id=TAX_ID)
    assert tools._known_business(TAX_ID, None) == {
        "business": "אושר עד", "category": "מזון וקניות", "matched_by": "tax_id"}


# --- Duplicates ---

def test_duplicate_by_date_and_amount(add_file):
    record(add_file("a.jpg"), amount=370.6)
    result = tools.find_duplicate("2026-09-09", 370.6, source_file=add_file("b.jpg", b"another photo"))
    assert [c["matched_by"] for c in result["candidates"]] == ["date_and_amount"]


def test_same_business_same_day_other_amount_is_a_candidate(add_file):
    # Another photo of a handwritten receipt came back with another amount (630 / 650)
    record(add_file("a.jpg"), amount=630, tax_id=TAX_ID)
    result = tools.find_duplicate("2026-09-09", 650, source_file=add_file("b.jpg", b"x"), tax_id=TAX_ID)
    assert [c["matched_by"] for c in result["candidates"]] == ["same_business_and_date"]


def test_no_duplicate_for_a_different_purchase(add_file):
    record(add_file("a.jpg"), amount=370.6)
    result = tools.find_duplicate("2026-10-23", 891, source_file=add_file("b.jpg", b"x"))
    assert result["possible_duplicate"] is False


def test_exact_copy_under_another_name_is_marked(add_file):
    record(add_file("01.jpg", b"same bytes"))
    add_file("01-copy.jpg", b"same bytes")
    documents = {d["file"]: d for d in tools.list_documents()["documents"]}
    assert documents["01-copy.jpg"]["copy_of"] == "t001"


# --- Monthly summary (all the math is in Python) ---

def test_monthly_summary_totals_and_budget(add_file):
    record(add_file("a.jpg"), amount=500, category="ביגוד והנעלה")
    record(add_file("b.jpg", b"b"), amount=200, category="ביגוד והנעלה")
    record(add_file("c.jpg", b"c"), amount=100.25)
    record(add_file("d.jpg", b"d"), amount=-50, category="ביגוד והנעלה")  # a refund
    record(add_file("e.jpg", b"e"), amount=5000, type="income", category="משכורת")
    record(add_file("f.jpg", b"f"), amount=999, date="2026-08-01")  # another month

    summary = tools.get_monthly_summary("2026-09")
    assert summary["transaction_count"] == 5
    assert summary["expenses_by_category"] == {"ביגוד והנעלה": 650, "מזון וקניות": 100.25}
    assert summary["expense_total"] == 750.25
    assert summary["income_total"] == 5000
    assert summary["balance"] == 4249.75
    clothing = next(b for b in summary["budgets"] if b["category"] == "ביגוד והנעלה")
    assert clothing["status"] == "over"  # 650 of 600


def test_summary_rejects_a_bad_month():
    assert not tools.get_monthly_summary("ספטמבר")["ok"]
