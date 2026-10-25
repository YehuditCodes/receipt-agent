"""All settings in one place: model, paths, categories, budgets, limits."""

from pathlib import Path

# --- Model ---
# The API key is NOT here: it is read from the OPENAI_API_KEY environment variable.
MODEL = "gpt-4o-mini"
# Model that reads the receipt images (read_document). Chosen by a test on real phone photos:
# gpt-4o-mini misread 4 of 5 totals and invented an amount from a blurry image;
# gpt-5.4-mini read all totals correctly and refused to guess from a blurry image.
READ_MODEL = "gpt-5.4-mini"
# How long the reading model "thinks" before answering: none / low / medium / high.
# medium: in tests it stopped inventing store names and misreading dates; high was
# a little more accurate but took up to ~30 seconds per image.
READ_REASONING_EFFORT = "medium"

# --- Agent limits ---
# Stop condition: the agent never runs more than this many steps.
MAX_STEPS = 25

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
INBOX_DIR = BASE_DIR / "inbox"
DATA_DIR = BASE_DIR / "data"
TRANSACTIONS_FILE = DATA_DIR / "transactions.json"
MEMORY_FILE = DATA_DIR / "memory.json"

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# --- Reading images (read_document) ---
# Larger files are refused before sending (OpenAI's own limit is 20 MB).
MAX_IMAGE_MB = 10
# Seconds to wait for the model to read one image.
READ_TIMEOUT_SECONDS = 90
# Cross-checks (items vs total, payments vs total) allow this much difference (₪) for rounding.
AMOUNT_TOLERANCE = 0.5

CURRENCY = "₪"

# --- Categories ---
# Category is what the money was for; the business (e.g. "חברת החשמל") is stored separately.
EXPENSE_CATEGORIES = [
    "מזון וקניות",
    "מסעדות ובתי קפה",
    "חשבונות בית",
    "דיור",
    "תחבורה",
    "בריאות ופארם",
    "ביגוד והנעלה",
    "לימודים",
    "פנאי ובילויים",
    "אחר",
]

INCOME_CATEGORIES = [
    "משכורת",
    "אחר",
]

# --- Monthly budgets (₪) ---
# Only for flexible spending; fixed costs (rent, bills) get no budget warning.
BUDGETS = {
    "מזון וקניות": 2000,
    "מסעדות ובתי קפה": 500,
    "ביגוד והנעלה": 400,
    "פנאי ובילויים": 400,
}

# Warn "close to budget" from this share of the budget; above 100% it is an overrun.
BUDGET_WARNING_RATIO = 0.8


def check_config() -> None:
    """Catch typos in the settings above before the agent starts. Raises ValueError."""
    problems = []
    if not isinstance(MAX_STEPS, int) or MAX_STEPS < 1:
        problems.append("MAX_STEPS חייב להיות מספר שלם חיובי")
    if not isinstance(BUDGET_WARNING_RATIO, (int, float)) or not 0 < BUDGET_WARNING_RATIO < 1:
        problems.append("BUDGET_WARNING_RATIO חייב להיות בין 0 ל-1")
    if READ_REASONING_EFFORT not in ("none", "low", "medium", "high"):
        problems.append("READ_REASONING_EFFORT חייב להיות none / low / medium / high")
    for name, value in (("MAX_IMAGE_MB", MAX_IMAGE_MB), ("READ_TIMEOUT_SECONDS", READ_TIMEOUT_SECONDS),
                        ("AMOUNT_TOLERANCE", AMOUNT_TOLERANCE)):
        if not isinstance(value, (int, float)) or value <= 0:
            problems.append(f"{name} חייב להיות מספר חיובי")
    for name, categories in (("EXPENSE_CATEGORIES", EXPENSE_CATEGORIES), ("INCOME_CATEGORIES", INCOME_CATEGORIES)):
        if len(set(categories)) != len(categories):
            problems.append(f"יש קטגוריה כפולה ב-{name}")
    for category, amount in BUDGETS.items():
        if category not in EXPENSE_CATEGORIES:
            problems.append(f"לתקציב '{category}' אין קטגוריית הוצאה כזו")
        if not isinstance(amount, (int, float)) or amount <= 0:
            problems.append(f"התקציב של '{category}' חייב להיות מספר חיובי")
    if problems:
        raise ValueError("בעיה בהגדרות (config.py):\n- " + "\n- ".join(problems))
