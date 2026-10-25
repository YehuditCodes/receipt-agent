"""Entry point: python main.py - processes the images in inbox/ and shows the monthly summary."""

import os
import sys

import openai

import agent
import config
from storage import StorageError
from ui.terminal import TerminalUI


def main() -> int:
    ui = TerminalUI()
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
    except openai.OpenAIError as e:
        ui.error(f"שגיאה בפנייה למודל: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
