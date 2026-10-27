"""Rules Python enforces on the model's tool calls (agent._run_tool), with a fake UI and no model."""

import json

import agent
from ui.base import AgentUI


class FakeUI(AgentUI):
    """Answers every question with a fixed answer and shows nothing."""

    def __init__(self, answer=""):
        self.answer = answer
        self.questions = []

    def thinking(self, text): pass
    def tool_call(self, name, args): pass
    def tool_result(self, name, result): pass
    def guard(self, text): pass
    def show_summary(self, summary): pass
    def final(self, text): pass
    def error(self, text): pass

    def ask(self, question, options=None, answer_type="text"):
        self.questions.append(question)
        return self.answer


def run_tool(name, args, state, ui=None):
    return agent._run_tool(name, json.dumps(args, ensure_ascii=False), ui or FakeUI(), state)


def transaction(file, **changes):
    args = {"type": "expense", "date": "2026-09-09", "amount": 370.6, "category": "מזון וקניות",
            "business": "אושר עד", "source_file": file}
    return {**args, **changes}


def test_no_recording_before_the_duplicate_check(add_file):
    # Check 1: the model skipped the check and recorded a receipt twice
    state = agent._RunState()
    file = add_file("01.jpg")
    refused = run_tool("add_transaction", transaction(file), state)
    assert not refused["ok"] and "notice" in refused

    run_tool("find_duplicate", {"date": "2026-09-09", "amount": 370.6, "source_file": file}, state)
    assert run_tool("add_transaction", transaction(file), state)["ok"]


def test_other_category_only_if_she_chose_it(add_file):
    # Check 7: "אחר" was used to avoid asking
    state = agent._RunState()
    file = add_file("01.jpg")
    run_tool("find_duplicate", {"date": "2026-09-09", "amount": 370.6, "source_file": file}, state)
    assert not run_tool("add_transaction", transaction(file, category="אחר"), state)["ok"]

    run_tool("ask_user", {"question": "לאיזו קטגוריה?", "file": file, "options": ["אחר"]}, state, FakeUI("אחר"))
    assert run_tool("add_transaction", transaction(file, category="אחר"), state)["ok"]


def test_a_receipt_is_never_income(add_file):
    state = agent._RunState()
    file = add_file("05.jpg")
    state.document_types[file] = "receipt"
    run_tool("find_duplicate", {"date": "2026-09-09", "amount": 370.6, "source_file": file}, state)
    result = run_tool("add_transaction", transaction(file, type="income", category="מתנות והטבות"), state)
    assert not result["ok"]


def test_invalid_date_answer_is_not_passed_on(add_file):
    state = agent._RunState()
    result = run_tool("ask_user", {"question": "מה התאריך?", "answer_type": "date"}, state, FakeUI("222222"))
    assert result["answer"] == ""


def test_question_shows_which_document_it_is_about(add_file):
    state = agent._RunState()
    state.documents["01.jpg"] = "אושר עד · 2026-09-09 · 370.60 ₪"
    ui = FakeUI("מזון וקניות")
    run_tool("ask_user", {"question": "לאיזו קטגוריה?", "file": "01.jpg"}, state, ui)
    assert ui.questions[0].startswith("מסמך: אושר עד")


def test_unknown_tool_and_broken_arguments_go_back_to_the_model():
    state = agent._RunState()
    assert not run_tool("delete_everything", {}, state)["ok"]
    assert not agent._run_tool("add_transaction", "{not json", FakeUI(), state)["ok"]
