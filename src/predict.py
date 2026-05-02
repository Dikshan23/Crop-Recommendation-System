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
    """
    Returns a penalty value between 0.0 and 0.40.
    Applied to raw confidence to account for suspicious inputs.

    Penalty sources:
        - Value near a valid boundary  → -5% per parameter
        - Value in unusual agro range  → -3% per parameter
        - Maximum total penalty capped → -40%
    """
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
    """
    Predict the most suitable crop and return confidence score.

    Parameters:
        N           (float): Nitrogen content in soil
        P           (float): Phosphorus content in soil
        K           (float): Potassium content in soil
        temperature (float): Temperature in Celsius
        humidity    (float): Relative humidity (%)
        ph          (float): Soil pH value
        rainfall    (float): Rainfall in mm

    Returns:
        tuple:
            crop        (str):   Predicted crop name e.g. 'rice'
            confidence  (float): Penalty-adjusted confidence score 0.0 – 1.0
            proba       (dict):  Full class probability distribution (raw)

    Raises:
        ValueError: If any input is outside the valid range
    """
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