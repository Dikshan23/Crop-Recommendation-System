"""
src/inference_pipeline.py
=========================
ONLY safe entry point for prediction (security gate)
"""

from src.validations import validate_inputs, warn_inputs
from src.predict import predict_crop


class PredictionBlocked(Exception):
    pass


def run_inference(N, P, K, temperature, humidity, ph, rainfall):

    # 🔴 HARD VALIDATION FIRST (NO EXCEPTIONS BEFORE THIS)
    errors = validate_inputs(N, P, K, temperature, humidity, ph, rainfall)

    if errors:
        raise PredictionBlocked(errors)

    # 🧠 MODEL INFERENCE (ONLY IF VALID)
    crop, confidence, proba = predict_crop(
        N, P, K, temperature, humidity, ph, rainfall
    )

    # ⚠️ NON-BLOCKING WARNINGS
    warnings = warn_inputs(N, P, K, temperature, humidity, ph, rainfall)

    return {
        "crop": crop,
        "confidence": confidence,
        "proba": proba,
        "warnings": warnings
    }