"""The contract between the agent and whatever shows it to the user.

The agent never prints or reads input itself: it calls these methods. A terminal UI
(terminal.py) implements them today; a web UI could implement them later without
touching the agent.
"""

from abc import ABC, abstractmethod


class AgentUI(ABC):
    """Everything the agent may show or ask. Every UI must implement all of these."""

    @abstractmethod
    def thinking(self, text: str) -> None:
        """The model explains what it plans to do next."""

    @abstractmethod
    def tool_call(self, name: str, args: dict) -> None:
        """The model chose a tool, with these arguments."""

    @abstractmethod
    def tool_result(self, name: str, result: dict) -> None:
        """A tool returned (the {"ok": ...} dict from tools.py)."""

    @abstractmethod
    def ask(self, question: str, options: list[str] | None = None) -> str:
        """Ask the user a question and return the answer (stripped, may be empty).

        With `options` (chosen by the agent, e.g. likely categories) the user picks one of them
        or types a free answer; the chosen option's text is returned.
        """

    @abstractmethod
    def show_summary(self, summary: dict) -> None:
        """Show a monthly summary (the result of tools.get_monthly_summary)."""

    @abstractmethod
    def final(self, text: str) -> None:
        """The agent finished; this is its last message."""

    @abstractmethod
    def error(self, text: str) -> None:
        """An error that stops the agent (damaged data file, missing API key...)."""
