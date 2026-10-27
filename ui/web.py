"""Web UI: the same AgentUI contract, but every message goes to the browser as a JSON event.

The agent runs in a worker thread (it is plain blocking code); the server sends the events over a
WebSocket. ask() blocks the agent's thread until the browser sends an answer back.
"""

import queue
from typing import Callable

from ui.base import AgentUI
from ui.describe import describe_result


class BrowserGone(EOFError):
    """The browser tab was closed during a run. An EOFError, so the agent stops like on Ctrl+Z."""


class WebUI(AgentUI):
    def __init__(self, send: Callable[[dict], None]) -> None:
        self._send = send  # thread-safe: puts the event on the server's queue
        self._answers: queue.Queue[str | None] = queue.Queue()
        self._gone = False

    # --- Called by the server ---

    def answer(self, text: str) -> None:
        self._answers.put(text)

    def disconnect(self) -> None:
        """The browser left: stop at the next message or question (the work so far is saved)."""
        self._gone = True
        self._answers.put(None)

    def _emit(self, event: dict) -> None:
        if self._gone:
            raise BrowserGone("הדפדפן נסגר")
        self._send(event)

    # --- AgentUI ---

    def thinking(self, text: str) -> None:
        if text and text.strip():
            self._emit({"type": "thinking", "text": text.strip()})

    def tool_call(self, name: str, args: dict) -> None:
        if name != "ask_user":  # the question itself is shown by ask()
            self._emit({"type": "tool_call", "name": name, "args": args})

    def guard(self, text: str) -> None:
        self._emit({"type": "guard", "text": text})

    def tool_result(self, name: str, result: dict) -> None:
        if not result.get("ok"):
            self._emit({"type": "tool_result", "name": name, "ok": False,
                        "text": result.get("error", "שגיאה לא ידועה")})
            return
        if name == "ask_user":  # her answer is already shown in the question card
            return
        event = {"type": "tool_result", "name": name, "ok": True, "text": describe_result(name, result),
                 "warnings": result.get("warnings", []), "info": result.get("info", [])}
        if name == "add_transaction":
            event["transaction"] = result["transaction"]
        self._emit(event)

    def ask(self, question: str, options: list[str] | None = None, answer_type: str = "text") -> str:
        options = [o for o in (options or []) if isinstance(o, str) and o.strip()]
        self._emit({"type": "ask", "question": question, "options": options, "answer_type": answer_type})
        answer = self._answers.get()
        if answer is None:
            raise BrowserGone("הדפדפן נסגר")
        return answer.strip()

    def show_summary(self, summary: dict) -> None:
        self._emit({"type": "summary", "summary": summary})

    def final(self, text: str) -> None:
        if not self._gone:
            self._send({"type": "final", "text": text})

    def error(self, text: str) -> None:
        if not self._gone:
            self._send({"type": "error", "text": text})
