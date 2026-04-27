import numpy as np

from src.model_utils import load_model
from src.validations import validate_inputs


_model = None


def _get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


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
            confidence  (float): Confidence score 0.0 – 1.0
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

    confidence = proba.get(str(crop), proba.get(crop, 0.0))

    return crop, confidence, proba