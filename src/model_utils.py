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

from src.config import DATASET_PATH, MODEL_PATH, METRICS_PATH


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def save_model(model):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from: {MODEL_PATH}")
    return model


def save_metrics(metrics):
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to: {METRICS_PATH}")