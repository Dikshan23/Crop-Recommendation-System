import numpy as np

from src.model_utils import load_model
from src.validations import validate_inputs, warn_inputs


_model = None

# Confidence penalty factors for warnings
BOUNDARY_WARNING_FACTOR = 0.80      # 80% of original confidence if near boundary
UNUSUAL_WARNING_FACTOR = 0.88       # 88% of original confidence if unusual
MIN_CONFIDENCE_WHEN_WARNED = 0.05   # Minimum confidence floor when warnings exist


def _get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def _apply_warning_penalty(confidence, warnings):
    """
    Reduce confidence when inputs are near boundaries or agronomically unusual.
    
    Parameters:
        confidence (float): Raw model confidence (0.0 - 1.0)
        warnings (list): List of warning messages from warn_inputs()
    
    Returns:
        float: Adjusted confidence score with penalties applied
    """
    if not warnings:
        return max(0.0, min(1.0, float(confidence)))

    adjusted = float(confidence)
    for warning in warnings:
        if "near the" in warning:
            adjusted *= BOUNDARY_WARNING_FACTOR
        else:
            adjusted *= UNUSUAL_WARNING_FACTOR

    adjusted = max(MIN_CONFIDENCE_WHEN_WARNED, adjusted)
    return max(0.0, min(1.0, adjusted))


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
            confidence  (float): Confidence score 0.0 – 1.0 (penalized if warnings exist)
            proba       (dict):  Full class probability distribution

    Raises:
        ValueError: If any input is outside the valid range
    """
    errors = validate_inputs(N, P, K, temperature, humidity, ph, rainfall)
    if errors:
        raise ValueError("Invalid input values:\n" + "\n".join(f"• {e}" for e in errors))

    model    = _get_model()
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    crop  = model.predict(features)[0]
    proba = model.predict_proba(features)[0]  # dict of {class: probability}

    raw_confidence = proba.get(str(crop), proba.get(crop, 0.0))
    
    # Apply penalty for boundary/unusual values
    warnings = warn_inputs(N, P, K, temperature, humidity, ph, rainfall)
    confidence = _apply_warning_penalty(raw_confidence, warnings)

    return crop, confidence, proba