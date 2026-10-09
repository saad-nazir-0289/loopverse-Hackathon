"""Settings for the advisory assistant. Everything can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path=ROOT / ".env"):
    """Minimal .env reader (KEY=VALUE lines). Real environment variables take precedence."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()
DOCS_DIR = ROOT / "docs"
PREDICTIONS = ROOT / "predictions.csv"
# Released at the event midpoint; format assumed: sensor_id,target_date,predicted_pm25[,hazardous]
REFERENCE_FORECAST = Path(os.environ.get("REFERENCE_FORECAST_PATH", ROOT / "reference" / "reference_forecast.csv"))
# "team" (default; falls back to reference only if predictions.csv is missing) or "reference"
FORECAST_SOURCE = os.environ.get("ADVISOR_FORECAST_SOURCE", "team")
METADATA = ROOT / "weather" / "sensor_metadata.csv"
HOLDOUT = ROOT / "holdout" / "holdout_inputs.csv"



def _default_today():
    """Scenario clock = the day before the first forecast target (= last observed day).
    Derived from the holdout file, so judges' files with other dates just work."""
    try:
        import pandas as pd
        first = pd.to_datetime(pd.read_csv(HOLDOUT)["target_date"]).min()
        return (first - pd.Timedelta(days=1)).date().isoformat()
    except Exception:
        return "2026-10-28"


# "Tomorrow" means TODAY + 1 = the first forecast day. Override with ADVISOR_TODAY=YYYY-MM-DD.
TODAY = os.environ.get("ADVISOR_TODAY") or _default_today()

HAZARD = 165.0           # SPEC section 3 and DOC-01: what the DOCUMENTS call hazardous (fixed)
# Alarm threshold for the hazardous yes/no flag (forecast tool, UI, predictions download).
# None = use the "hazardous" column of predictions.csv as written by forecast.py.
_alarm = os.environ.get("ADVISOR_ALARM_THRESHOLD")
ALARM_THRESHOLD = float(_alarm) if _alarm else None
UNHEALTHY = 100.0        # DOC-01 band boundary
SCHOOL_FULL_CLOSURE = 250.0  # DOC-03


def _typical_error():
    """Walk-forward MAE of the submitted model, read from forecast.py's output (default 13)."""
    try:
        import pandas as pd
        t = pd.read_csv(ROOT / "outputs" / "walk_forward_scores.csv", index_col=0)
        return int(round(t.drop(index=["persistence", "mean7"], errors="ignore")["MAE"].iloc[0]))
    except Exception:
        return 13


TYPICAL_ERROR = _typical_error()  # ug/m3, stated in every forecast answer

# LLM: "openai" (default when OPENAI_API_KEY is set) or "none" (deterministic answers only).
# Any OpenAI-compatible server works via OPENAI_BASE_URL (e.g. Ollama: http://localhost:11434/v1).
LLM_PROVIDER = os.environ.get("ADVISOR_LLM", "openai" if os.environ.get("OPENAI_API_KEY") else "none")
LLM_MODEL = os.environ.get("OPENAI_MODEL", "gpt-6.1-sol")
LLM_FALLBACK_MODEL = os.environ.get("OPENAI_FALLBACK_MODEL", "gpt-6-luna")
LLM_TIMEOUT = float(os.environ.get("ADVISOR_LLM_TIMEOUT", "60"))
# "auto": Responses API, then Chat Completions (what most free OpenAI-compatible providers offer)
LLM_API = os.environ.get("ADVISOR_LLM_API", "auto")

TOP_K_CHUNKS = 5
