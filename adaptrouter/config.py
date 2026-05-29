# adaptrouter/config.py
import os
from dotenv import load_dotenv

load_dotenv()

LLM_ROUTER_PATH = os.getenv(
    "LLM_ROUTER_PATH",
    r"C:\Users\Shreya\OneDrive\Desktop\llm-router"
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── SELF-IMPROVING SETTINGS ───────────────────────────────────────────────────
RETRAIN_THRESHOLD           = 20
MIN_RETRAIN_INTERVAL_HOURS  = 1.0
DRIFT_THRESHOLD             = 0.15
ACCURACY_TOLERANCE          = 0.02

# Fix 3: lowered from 0.1 to 0.05 so SHAP direction labels fire correctly
SHAP_DIRECTION_THRESHOLD    = 0.05

# Fix 7: long query word limit — queries above this go directly to smart model
LONG_QUERY_WORD_LIMIT       = 150

# Fix 9: minimum bucket size for calibration curve plotting
CALIBRATION_MIN_BUCKET_SIZE = 3

REPHRASING_SIMILARITY_THRESHOLD    = 0.82
REPHRASING_TIME_WINDOW_SECONDS     = 60

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "adaptrouter.db")

FAST_MODEL  = "llama-3.1-8b-instant"
SMART_MODEL = "llama-3.3-70b-versatile"