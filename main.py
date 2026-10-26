"""Entry point: python main.py - processes the images in inbox/ and shows the monthly summary.

The browser version (web_app.py) runs the same agent through run_with().
"""

import os
import sys

import agent
import config
from storage import StorageError
from ui.base import AgentUI
from ui.terminal import TerminalUI


def run_with(ui: AgentUI) -> int:
    """Check the settings, run the agent with this UI and show any stopping error. Returns an exit code."""
    try:
        config.check_config()
    except ValueError as e:
        ui.error(str(e))
        return 1
    if not os.environ.get("OPENAI_API_KEY"):
        ui.error("לא נמצא מפתח API. יש להגדיר את משתנה הסביבה OPENAI_API_KEY (ראו README)")
        return 1

    try:
        agent.run(ui)
    except StorageError as e:  # a damaged data file: stop without touching it
        ui.error(str(e))
        return 1
    except agent.AgentStopped as e:  # API problem or Ctrl+C - already explained, with what was saved
        ui.error(str(e))
        return 1
    return 0


def main() -> int:
    return run_with(TerminalUI())


if __name__ == "__main__":
    sys.exit(main())
