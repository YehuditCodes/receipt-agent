"""All settings in one place: model, paths, categories, budgets, limits."""

from pathlib import Path

# --- Model ---
# The API key is NOT here: it is read from the OPENAI_API_KEY environment variable.
MODEL = "gpt-4o-mini"

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


def check_config() -> None:
    """Catch typos in the settings above before the agent starts. Raises ValueError."""
    problems = []
    if not isinstance(MAX_STEPS, int) or MAX_STEPS < 1:
        problems.append("MAX_STEPS חייב להיות מספר שלם חיובי")
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
