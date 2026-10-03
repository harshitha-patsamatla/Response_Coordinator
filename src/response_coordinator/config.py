"""Runtime configuration with safe local defaults."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "emergency-xlm-roberta"
DB_PATH = Path(os.getenv("COORDINATOR_DB_PATH", PROJECT_ROOT / ".data" / "incidents.sqlite3"))
ALLOW_MODEL_DOWNLOAD = os.getenv("ALLOW_MODEL_DOWNLOAD", "true").lower() == "true"
ENABLE_OSM_LOOKUPS = os.getenv("ENABLE_OSM_LOOKUPS", "true").lower() == "true"
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "30"))
DEDUP_WINDOW_MINUTES = 30
DEDUP_RADIUS_KM = 1.5
DEDUP_SIMILARITY = 0.78
NOMINATIM_USER_AGENT = os.getenv(
    "NOMINATIM_USER_AGENT", "emergency-response-coordinator-prototype/0.1 (local demo)"
)