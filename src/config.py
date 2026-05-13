"""
src/config.py
=============
Central path configuration for AgroTree.
"""
import os

# Root folder of the project (AGROTREE/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_PATH           = os.path.join(BASE_DIR, "data", "raw",       "raw_data.csv")
PROCESSED_DATASET_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")
MODEL_PATH             = os.path.join(BASE_DIR, "models",  "crop_model.pkl")
SCALER_PATH            = os.path.join(BASE_DIR, "models",  "scaler.pkl")
METRICS_PATH           = os.path.join(BASE_DIR, "results", "metrics",    "model_metrics.json")

# Derived artefact paths (saved alongside scaler)
MEDIANS_PATH           = os.path.join(BASE_DIR, "models",  "crop_medians.pkl")
FENCES_PATH            = os.path.join(BASE_DIR, "models",  "iqr_fences.pkl")