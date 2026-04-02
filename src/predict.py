import numpy as np

from src.model_utils import load_model


_model = None


def _get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """
    Predict the most suitable crop.

    Parameters:
        N           (float): Nitrogen content in soil
        P           (float): Phosphorus content in soil
        K           (float): Potassium content in soil
        temperature (float): Temperature in Celsius
        humidity    (float): Relative humidity (%)
        ph          (float): Soil pH value
        rainfall    (float): Rainfall in mm

    Returns:
        str: Predicted crop name e.g. 'rice', 'maize'
    """
    model    = _get_model()
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    return model.predict(features)[0]