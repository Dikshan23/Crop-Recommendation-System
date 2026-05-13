"""
src/predict.py
==============
Inference module with Platt scaling calibration.
"""

import numpy as np
import joblib
import os

from src.model_utils import load_model, load_scaler
from src.validations import validate_inputs, warn_inputs, VALID_RANGES, BOUNDARY_MARGIN
from src.config import MODEL_PATH
from src.calibration import apply_platt_scaling, load_platt_params

# Feature column order – must match training
FEATURE_COLUMNS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
LOG_COLUMNS     = {'K': 2, 'rainfall': 6}

_model  = None
_scaler = None

def _get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model

def _get_scaler():
    global _scaler
    if _scaler is None:
        _scaler = load_scaler()
    return _scaler

def _compute_penalty(N, P, K, temperature, humidity, ph, rainfall):
    """Heuristic penalty (still useful for extreme values)."""
    NEAR_BOUNDARY_PENALTY = 0.05
    UNUSUAL_RANGE_PENALTY = 0.03
    MAX_PENALTY = 0.40

    values = {
        "nitrogen":    N,
        "phosphorus":  P,
        "potassium":   K,
        "temperature": temperature,
        "humidity":    humidity,
        "ph":          ph,
        "rainfall":    rainfall,
    }

    total_penalty = 0.0
    for key, value in values.items():
        meta    = VALID_RANGES[key]
        min_val = meta["min"]
        max_val = meta["max"]
        margin  = (max_val - min_val) * BOUNDARY_MARGIN
        if value <= min_val + margin or value >= max_val - margin:
            total_penalty += NEAR_BOUNDARY_PENALTY

    warnings = warn_inputs(N, P, K, temperature, humidity, ph, rainfall)
    unusual_count = sum(1 for w in warnings if "unusually low" in w or "unusually high" in w)
    total_penalty += unusual_count * UNUSUAL_RANGE_PENALTY

    return min(total_penalty, MAX_PENALTY)

def _preprocess_inputs(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]], dtype=float)
    for col, idx in LOG_COLUMNS.items():
        features[0][idx] = np.log1p(features[0][idx])
    scaler = _get_scaler()
    features = scaler.transform(features)
    return features

def predict_crop(N, P, K, temperature, humidity, ph, rainfall, use_calibration=True):
    """
    Predict the most suitable crop.

    Parameters
    ----------
    use_calibration : bool, default True
        If True, returns Platt‑scaled confidence (recommended).
        If False, returns raw tree confidence.

    Returns
    -------
    crop       : str
    confidence : float (calibrated or raw, then multiplied by penalty)
    proba      : dict of class probabilities (raw)
    """
    # Hard validation
    errors = validate_inputs(N, P, K, temperature, humidity, ph, rainfall)
    if errors:
        raise ValueError("Invalid input values:\n" + "\n".join(f"• {e}" for e in errors))

    # Preprocess
    features = _preprocess_inputs(N, P, K, temperature, humidity, ph, rainfall)

    # Predict
    model = _get_model()
    crop = model.predict(features)[0]
    proba = model.predict_proba(features)[0]   # list of dicts

    # Raw confidence of predicted class
    raw_confidence = proba.get(str(crop), proba.get(crop, 0.0))

    # Apply Platt scaling if requested
    if use_calibration:
        confidence = apply_platt_scaling(raw_confidence)
    else:
        confidence = raw_confidence

    # Heuristic penalty (still applied to avoid overconfidence on extreme inputs)
    penalty = _compute_penalty(N, P, K, temperature, humidity, ph, rainfall)
    confidence = confidence * (1 - penalty)

    return crop, confidence, proba