"""
config.py
---------
Centralised paths and constants so every script agrees on where things live,
instead of each file guessing relative paths independently.
"""
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

PRIMARY_DATASET = os.path.join(DATA_DIR, "hannousse_raw.csv")
UCI_DATASET = os.path.join(DATA_DIR, "uci_phishing_30.csv")
GREGA_DATASET = os.path.join(DATA_DIR, "grega_phishing_111.csv")

MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.joblib")
BEST_MODEL_NAME_PATH = os.path.join(MODELS_DIR, "best_model_name.joblib")

METRICS_PRIMARY_JSON = os.path.join(REPORTS_DIR, "metrics_primary.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_URL_LENGTH = 2048  # RFC-practical upper bound; longer input is rejected by the app
