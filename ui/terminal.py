"""Terminal UI with rich: colors, panels and a summary table.

Hebrew: most Windows terminals (incl. VS Code's) do not support right-to-left text and show
Hebrew letters reversed. When printing to a real terminal we reorder each line with python-bidi
so it reads correctly. When output goes to a file (e.g. saving a demo run) we keep the normal
order, because editors already display Hebrew correctly.
"""

import re
import sys
import textwrap

from bidi.algorithm import get_display
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import tools
from ui.base import AgentUI
from ui.describe import describe_result, money as _money

_HEBREW_LETTER = re.compile(r"[֐-׿]")
# Hebrew or English only: symbols such as "ℹ" count as letters in Unicode
_ANY_LETTER = re.compile(r"[A-Za-z֐-׿]")
# Output never uses more columns than this, even in a wide window: lines padded to the full
# width of a wide window break apart when the window is later made narrower
_MAX_WIDTH = 100
# Panel border + padding take 4 columns of the line
_PANEL_MARGIN = 4
# Long tool arguments (e.g. an items list) are cut to this length on screen
_MAX_ARG_LENGTH = 60

_BUDGET_STATUS = {"ok": ("תקין", "green"), "near": ("קרוב לתקציב", "yellow"), "over": ("חריגה!", "bold red")}


def _starts_rtl(text: str) -> bool:
    """True if the first letter is Hebrew - then the line is right-aligned."""
    first = _ANY_LETTER.search(text)
    return bool(first and _HEBREW_LETTER.match(first.group()))


class TerminalUI(AgentUI):
    def __init__(self) -> None:
        # Windows may default to a code page without Hebrew (crashes when output is redirected)
        for stream in (sys.stdout, sys.stdin):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")
        self.console = Console()
        self.fix_rtl = self.console.is_terminal

    # --- Layout and Hebrew helpers ---

    @property
    def _width(self) -> int:
        return min(self.console.width, _MAX_WIDTH)

    def _print(self, renderable) -> None:
        """Print a panel / table on the right side, as Hebrew is read."""
        self.console.print(Align.right(renderable), width=self._width)

    def _visual(self, text: str, width: int | None = None) -> str:
        """Text as it should be printed. In a terminal: wrapped first, then each line reordered
        (wrapping after reordering would put the end of a sentence on the first line)."""
        if not self.fix_rtl:
            return text
        width = width or self._width
        lines = []
        for paragraph in str(text).splitlines() or [""]:
            lines.extend(textwrap.wrap(paragraph, width) or [""])
        return "\n".join(get_display(line) for line in lines)

    def _text(self, text: str, style: str = "", width: int | None = None) -> Text:
        """A rich Text, right-aligned when it is Hebrew."""
        justify = "right" if _starts_rtl(str(text)) else "left"
        return Text(self._visual(text, width), style=style, justify=justify)

    def _panel(self, text: str, title: str, style: str) -> Panel:
        """A frame as wide as its text (not the whole window)."""
        body = self._text(text, width=self._width - _PANEL_MARGIN)
        return Panel(body, title=self._visual(title), border_style=style, expand=False)

    def _line(self, text: str, style: str = "") -> None:
        line = self._text(text, style)
        # rich ignores a top-level Text's own justify, so it is passed to print
        self.console.print(line, justify=line.justify, width=self._width)

    # --- AgentUI ---

    def thinking(self, text: str) -> None:
        if text and text.strip():
            self._line(f"💭 {text.strip()}", "dim")

    def tool_call(self, name: str, args: dict) -> None:
        shown = []
        for key, value in args.items():
            value = repr(value)
            if len(value) > _MAX_ARG_LENGTH:
                value = value[:_MAX_ARG_LENGTH] + "…"
            shown.append(f"{key}={value}")
        self._line(f"→ {name}({', '.join(shown)})", "cyan")
        # Reading takes 20-60 s with no output: in a clean-install test she took it for stuck and pressed Ctrl+C
        if name == "read_document":
            self._line("  קורא את המסמך... (עד דקה)", "dim")

    def tool_result(self, name: str, result: dict) -> None:
        if not result.get("ok"):
            self._line(f"✗ {name}: {result.get('error', 'שגיאה לא ידועה')}", "red")
            return
        self._line(f"✓ {describe_result(name, result)}", "green")
        for warning in result.get("warnings", []):
            self._line(f"⚠ {warning}", "yellow")
        for note in result.get("info", []):
            self._line(f"ℹ {note}", "dim")

    def guard(self, text: str) -> None:
        self._line(f"🛡 {text}", "dim")

    def ask(self, question: str, options: list[str] | None = None, answer_type: str = "text") -> str:
        # The terminal shows typed Hebrew reversed, so when the agent offers options the user
        # types only a number. The last number is always "type my own answer".
        options = [o for o in (options or []) if isinstance(o, str) and o.strip()]
        if not options:
            self._print(self._panel(question, "שאלה אלייך", "yellow"))
            return self._free_answer(answer_type)

        own_answer = len(options) + 1
        lines = [question, ""] + [f"{n} – {o}" for n, o in enumerate(options, 1)]
        lines.append(f"{own_answer} – אחר (אקליד בעצמי)")
        self._print(self._panel("\n".join(lines), "שאלה אלייך – הקלידי מספר", "yellow"))
        while True:
            answer = self._input()
            if not answer:
                return ""
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                choice = options[int(answer) - 1]
                self._line(f"✓ בחרת: {choice}", "yellow")
                return choice
            if answer == str(own_answer):
                return self._free_answer(answer_type)
            self._line(f"יש להקליד מספר בין 1 ל-{own_answer}", "red")

    def _input(self) -> str:
        return self.console.input("[bold yellow]> [/bold yellow]").strip()

    def _free_answer(self, answer_type: str = "text") -> str:
        answer = self._input()
        # A date / amount is asked again until it is a valid one (or left empty)
        while answer and answer_type != "text" and tools.normalize_answer(answer, answer_type) is None:
            example = "למשל 14.3.2025" if answer_type == "date" else "מספר גדול מ-0, למשל 200"
            self._line(f"זה לא {'תאריך' if answer_type == 'date' else 'סכום'} תקין ({example}). נסי שוב:", "red")
            answer = self._input()
        # Echo the answer: while typing, the terminal showed the Hebrew letters reversed
        if answer:
            self._line(f"✓ התשובה שלך: {answer}", "yellow")
        return answer

    def show_summary(self, summary: dict) -> None:
        v = self._visual
        table = Table(title=v(f"סיכום חודש {summary['month']}"), title_style="bold")
        table.add_column(v("סכום"), justify="right")
        table.add_column(v("פירוט"), justify="right")
        table.add_row(_money(summary["income_total"]), v("הכנסות"), style="green")
        for category, amount in summary["expenses_by_category"].items():
            table.add_row(_money(amount), v(category))
        table.add_row(_money(summary["expense_total"]), v("סה\"כ הוצאות"), style="bold", end_section=True)
        balance_style = "bold green" if summary["balance"] >= 0 else "bold red"
        table.add_row(_money(summary["balance"]), v("יתרה"), style=balance_style)
        self._print(table)

        budgets = Table(title=v("תקציבים"), title_style="bold")
        for header in ("מצב", "אחוז", "תקציב", "הוצאה", "קטגוריה"):
            budgets.add_column(v(header), justify="right")
        for b in summary["budgets"]:
            label, style = _BUDGET_STATUS.get(b["status"], (b["status"], ""))
            budgets.add_row(Text(v(label), style=style), f"{b['percent']}%", _money(b["budget"]),
                            _money(b["spent"]), v(b["category"]))
        self._print(budgets)
        if summary.get("note"):
            self._line(summary["note"], "dim")

    def final(self, text: str) -> None:
        self._print(self._panel(text, "סיום", "green"))

    def error(self, text: str) -> None:
        self._print(self._panel(text, "שגיאה", "red"))
