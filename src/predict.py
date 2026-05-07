import numpy as np

from src.model_utils import load_model
from src.validations import validate_inputs, warn_inputs, VALID_RANGES, BOUNDARY_MARGIN


_model = None


def _get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def _compute_penalty(N, P, K, temperature, humidity, ph, rainfall):
    NEAR_BOUNDARY_PENALTY = 0.05
    UNUSUAL_RANGE_PENALTY = 0.03
    MAX_PENALTY           = 0.40

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
    unusual_count = sum(
        1 for w in warnings
        if "unusually low" in w or "unusually high" in w
    )
    total_penalty += unusual_count * UNUSUAL_RANGE_PENALTY

    return min(total_penalty, MAX_PENALTY)


def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    errors = validate_inputs(N, P, K, temperature, humidity, ph, rainfall)
    if errors:
        raise ValueError("Invalid input values:\n" + "\n".join(f"• {e}" for e in errors))

    model    = _get_model()
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    crop  = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    raw_confidence      = proba.get(str(crop), proba.get(crop, 0.0))
    penalty             = _compute_penalty(N, P, K, temperature, humidity, ph, rainfall)
    adjusted_confidence = raw_confidence * (1 - penalty)

    return crop, adjusted_confidence, proba
