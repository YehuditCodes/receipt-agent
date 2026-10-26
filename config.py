"""All settings in one place: model, paths, categories, budgets, limits."""

from pathlib import Path

# --- Model ---
# The API key is NOT here: it is read from the OPENAI_API_KEY environment variable.
# Model that runs the agent loop (chooses the tools). Chosen by a test on the same 2 images:
# gpt-4o-mini skipped the duplicate check, guessed a missing store name, summarized the wrong
# month and never explained its steps; gpt-5.4-mini followed the rules.
MODEL = "gpt-5.4-mini"
# Model that reads the receipt images (read_document). Chosen by a test on real phone photos:
# gpt-4o-mini misread 4 of 5 totals and invented an amount from a blurry image;
# gpt-5.4-mini read all totals correctly and refused to guess from a blurry image.
READ_MODEL = "gpt-5.4-mini"
# How long the reading model "thinks" before answering: none / low / medium / high.
# medium: in tests it stopped inventing store names and misreading dates; high was
# a little more accurate but took up to ~30 seconds per image.
READ_REASONING_EFFORT = "medium"

# --- Agent limits ---
# Stop condition: the agent never runs more than this many steps. It handles one document
# at a time (read, duplicate check, question if needed, record): about 3-4 steps per document.
MAX_STEPS = 100
# Seconds to wait for one answer of the loop model (the SDK default is 10 minutes).
MODEL_TIMEOUT_SECONDS = 60

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
INBOX_DIR = BASE_DIR / "inbox"
DATA_DIR = BASE_DIR / "data"
TRANSACTIONS_FILE = DATA_DIR / "transactions.json"
MEMORY_FILE = DATA_DIR / "memory.json"
# Files the agent decided not to record (duplicate, "do not record"), so the next run skips them
SKIPPED_FILE = DATA_DIR / "skipped.json"
# The model's reading of each image, so the same image is not paid for twice (see read_document)
READ_CACHE_FILE = DATA_DIR / "read_cache.json"
USE_READ_CACHE = True

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# --- Reading images (read_document) ---
# Larger files are refused before sending (OpenAI's own limit is 20 MB).
MAX_IMAGE_MB = 10
# Seconds to wait for the model to read one image.
READ_TIMEOUT_SECONDS = 90
# A misread year (in tests: 22/10/26 read as 22/10/24): if the same day and month in the current
# year fall within this many days before today, the agent asks. Other old documents are not
# questioned - in run 2 all three "old date" questions were about dates that were read correctly.
YEAR_MISREAD_DAYS = 90
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
    "תרומות",
    "אחר",
]

INCOME_CATEGORIES = [
    "משכורת",
    "אחר",
]

# What belongs in each expense category - given to the model. With names only, it put an
# electricity bill under "דיור" and a driving school called "הבירה" under restaurants.
CATEGORY_DESCRIPTIONS = {
    "מזון וקניות": "סופרמרקט, מכולת, ירקן, מאפייה, מוצרים לבית",
    "מסעדות ובתי קפה": "מסעדה, בית קפה, משלוח אוכל, פאב",
    "חשבונות בית": "חשמל, מים, גז, ארנונה, טלפון, אינטרנט, טלוויזיה - כל חשבון (bill)",
    "דיור": "שכר דירה, משכנתא, ועד בית, תיקונים בדירה",
    "תחבורה": "דלק, תחבורה ציבורית, מונית, חניה, טיפול ברכב, טסט לרכב",
    "בריאות ופארם": "רופא, תרופות, בית מרקחת, קוסמטיקה וטיפוח, רפואה משלימה",
    "ביגוד והנעלה": "בגדים, נעליים, תיקים",
    "לימודים": "שכר לימוד, קורסים, שיעורים פרטיים, שיעורי נהיגה, ספרי לימוד",
    "פנאי ובילויים": "כלי נגינה, ספרים, תחביבים, הופעות, טיולים, מתנות",
    "תרומות": "קבלה על תרומה, עמותה, צדקה",
    "אחר": "רק אם המשתמשת בחרה בזה",
}

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
    for name, value in (("MAX_IMAGE_MB", MAX_IMAGE_MB), ("YEAR_MISREAD_DAYS", YEAR_MISREAD_DAYS), ("READ_TIMEOUT_SECONDS", READ_TIMEOUT_SECONDS),
                        ("MODEL_TIMEOUT_SECONDS", MODEL_TIMEOUT_SECONDS), ("AMOUNT_TOLERANCE", AMOUNT_TOLERANCE)):
        if not isinstance(value, (int, float)) or value <= 0:
            problems.append(f"{name} חייב להיות מספר חיובי")
    for name, categories in (("EXPENSE_CATEGORIES", EXPENSE_CATEGORIES), ("INCOME_CATEGORIES", INCOME_CATEGORIES)):
        if len(set(categories)) != len(categories):
            problems.append(f"יש קטגוריה כפולה ב-{name}")
    for category in CATEGORY_DESCRIPTIONS:
        if category not in EXPENSE_CATEGORIES:
            problems.append(f"לתיאור '{category}' אין קטגוריית הוצאה כזו")
    for category, amount in BUDGETS.items():
        if category not in EXPENSE_CATEGORIES:
            problems.append(f"לתקציב '{category}' אין קטגוריית הוצאה כזו")
        if not isinstance(amount, (int, float)) or amount <= 0:
            problems.append(f"התקציב של '{category}' חייב להיות מספר חיובי")
    if problems:
        raise ValueError("בעיה בהגדרות (config.py):\n- " + "\n- ".join(problems))
