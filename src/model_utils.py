import pandas as pd
import joblib
import json
import os
import sys

# Crucial for loading the pickled model which was saved when 'model.py'
# was top-level. We map 'model' to 'src.model' so joblib can find the class.
try:
    import src.model as model_pkg
    sys.modules['model'] = model_pkg
except ImportError:
    pass

from src.config import (
    DATASET_PATH,
    PROCESSED_DATASET_PATH,
    MODEL_PATH,
    SCALER_PATH,
    METRICS_PATH,
    MEDIANS_PATH,
    FENCES_PATH,
)
from src.logger import logger


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_dataset():
    """
    Loads the dataset for training.

    Priority:
        1. Processed dataset (Processed_data.csv)
           — used if it already exists from a previous preprocessing run.
        2. Raw dataset (data.csv)
           — used on first run; preprocess.py will clean and save it.

    Returns:
        df (DataFrame): Loaded dataset
    """
    if os.path.exists(PROCESSED_DATASET_PATH):
        logger.info(f"Processed dataset found. Loading from: {PROCESSED_DATASET_PATH}")
        df = pd.read_csv(PROCESSED_DATASET_PATH)
        logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

    logger.info(f"Loading raw dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ── Model ─────────────────────────────────────────────────────────────────────

def save_model(model):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Model saved to: {MODEL_PATH}")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded from: {MODEL_PATH}")
    return model


# ── Scaler ────────────────────────────────────────────────────────────────────

def save_scaler(scaler):
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"Scaler saved to: {SCALER_PATH}")


# ── Medians ───────────────────────────────────────────────────────────────────

def save_medians(medians):
    os.makedirs(os.path.dirname(MEDIANS_PATH), exist_ok=True)
    joblib.dump(medians, MEDIANS_PATH)
    logger.info(f"Medians saved to: {MEDIANS_PATH}")


def load_medians():
    if not os.path.exists(MEDIANS_PATH):
        raise FileNotFoundError(f"Medians not found at: {MEDIANS_PATH}")
    medians = joblib.load(MEDIANS_PATH)
    logger.info(f"Medians loaded from: {MEDIANS_PATH}")
    return medians


# ── IQR Fences ────────────────────────────────────────────────────────────────

def save_fences(fences):
    os.makedirs(os.path.dirname(FENCES_PATH), exist_ok=True)
    joblib.dump(fences, FENCES_PATH)
    logger.info(f"Fences saved to: {FENCES_PATH}")


def load_fences():
    if not os.path.exists(FENCES_PATH):
        raise FileNotFoundError(f"Fences not found at: {FENCES_PATH}")
    fences = joblib.load(FENCES_PATH)
    logger.info(f"Fences loaded from: {FENCES_PATH}")
    return fences


# ── Metrics ───────────────────────────────────────────────────────────────────

def save_metrics(metrics):
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to: {METRICS_PATH}")