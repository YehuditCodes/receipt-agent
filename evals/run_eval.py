"""Measure read_document on a fixed set of images with known answers.

    python evals/run_eval.py               -> expected.json (the images used while tuning)
    python evals/run_eval.py holdout.json  -> images NOT used while tuning (does it generalize?)

Each field of each document gets one of:
  right    - read correctly
  missing  - not read (the agent will ask her - safe)
  caught   - read wrong, but the reading has a warning, so the agent asks before recording
  WRONG    - read wrong with no warning: it would be recorded silently - the number that matters

Costs about 1 cent per image (it calls the reading model). Images are in evals/images/ (not in git).
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS_DIR.parent))

import config  # noqa: E402 - after the path change above
import tools  # noqa: E402

FIELDS = ["document_type", "date", "total", "business", "tax_id"]


def _digits(value) -> str:
    return re.sub(r"\D", "", str(value or ""))


def _same_name(read: str, expected: str) -> bool:
    # Geresh and quote marks vary between readings ("ניל'ס" / "נילס")
    def clean(s):
        return re.sub(r"[\"'׳״`]", "", s).lower()
    return clean(expected) in clean(read)


def _score(field: str, reading: dict, expected) -> str:
    if field not in reading:  # a field the reading does not return (yet)
        return "n/a"
    value = reading[field]
    allowed = expected if isinstance(expected, list) else [expected]
    if value is None:
        return "right" if None in allowed else "missing"
    names = [a for a in allowed if a is not None]
    if field == "business":
        ok = any(_same_name(value, a) for a in names)
    elif field == "tax_id":
        ok = _digits(value) in {_digits(a) for a in names}
    elif field == "total":
        ok = any(abs(value - a) < 0.01 for a in names)
    else:
        ok = value in names
    if ok:
        return "right"
    return "caught" if reading.get("warnings") else "WRONG"


def main() -> None:
    answers_file = sys.argv[1] if len(sys.argv) > 1 else "expected.json"
    expected = json.loads((EVALS_DIR / answers_file).read_text(encoding="utf-8"))["documents"]
    config.INBOX_DIR = EVALS_DIR / "images"
    config.USE_READ_CACHE = False  # measure a fresh reading every time
    config.USE_ONLINE_LOOKUPS = False  # measure the reading itself (no registry names, no currency conversion)
    totals = {field: Counter() for field in FIELDS}
    questions = 0

    for file, answers in expected.items():
        reading = tools.read_document(file)
        if not reading["ok"]:
            print(f"\n{file}: ERROR {reading['error']}")
            for field in FIELDS:
                totals[field]["missing"] += 1
            questions += 1
            continue
        questions += bool(reading["warnings"])
        print(f"\n{file}  ({answers['tests']})")
        for field in FIELDS:
            result = _score(field, reading, answers[field])
            totals[field][result] += 1
            if result != "right":
                print(f"  {field}: {result:8} read {reading.get(field)!r}, expected {answers[field]!r}")
        for warning in reading["warnings"]:
            print(f"  warning: {warning}")

    print(f"\n=== {len(expected)} documents, {questions} would get a question ===")
    for field in FIELDS:
        c = totals[field]
        if c["n/a"]:
            print(f"{field:14} n/a (not read yet)")
            continue
        print(f"{field:14} right {c['right']:2} | missing {c['missing']:2} | caught {c['caught']:2} | WRONG {c['WRONG']:2}")


if __name__ == "__main__":
    main()
