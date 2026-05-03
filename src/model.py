import numpy as np

from src.model_utils import load_model
from src.validations import validate_inputs


_model = None


def _get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def adjusted_confidence(confidence, warnings_list, proba):
    """
    Adjust model confidence based on:
    1. Warning penalty  — each warning reduces confidence by 10%
    2. Leaf size penalty — small leaf = less reliable prediction

    Args:
        confidence   (float): Raw model confidence from leaf node
        warnings_list (list): Warnings from warn_inputs()
        proba         (dict): Full class probability distribution

    Returns:
        float: Adjusted confidence score (floored at 0.50)
    """
    # 1. Warning penalty — 10% per warning, max 40% total
    warning_penalty = min(len(warnings_list) * 0.10, 0.40)

    # 2. Leaf size penalty — based on total samples in the leaf
    total_samples = sum(proba.values()) if all(
        isinstance(v, (int, float)) for v in proba.values()
    ) else None

    # proba values are probabilities (0-1), not raw counts
    # use spread of distribution as proxy for leaf reliability
    probs = list(proba.values())
    max_prob = max(probs)
    second_prob = sorted(probs, reverse=True)[1] if len(probs) > 1 else 0

    leaf_size_penalty = 0.0
    if max_prob - second_prob < 0.2:
        leaf_size_penalty = 0.20  # very mixed leaf
    elif max_prob - second_prob < 0.4:
        leaf_size_penalty = 0.10  # somewhat mixed leaf

    # combine penalties, floor at 0.50
    adjusted = confidence - warning_penalty - leaf_size_penalty
    return round(max(adjusted, 0.50), 4)


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
