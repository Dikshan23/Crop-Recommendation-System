"""
src/calibration.py
==================
Platt scaling implementation from scratch.
"""

import numpy as np
import joblib
import os
from scipy.optimize import minimize
from src.config import MODEL_PATH

# ----------------------------------------------------------------------
# Core functions
# ----------------------------------------------------------------------
def logit(p):
    """Probability → log-odds, clipped for stability."""
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))

def sigmoid(x):
    """Logistic function."""
    return 1 / (1 + np.exp(-x))

def _platt_loss(params, z, targets):
    """Negative log-likelihood for Platt scaling."""
    A, B = params
    cal = sigmoid(A * z + B)
    cal = np.clip(cal, 1e-15, 1 - 1e-15)
    loss = -np.mean(targets * np.log(cal) + (1 - targets) * np.log(1 - cal))
    return loss

def fit_platt_scaling(raw_confs, correctness):
    """Learn A, B such that sigmoid(A*logit(p) + B) is calibrated."""
    raw_confs = np.asarray(raw_confs)
    correctness = np.asarray(correctness)
    z = logit(raw_confs)
    res = minimize(_platt_loss, [1.0, 0.0], args=(z, correctness), method='L-BFGS-B')
    return res.x[0], res.x[1]

# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
def get_default_params_path():
    return os.path.join(os.path.dirname(MODEL_PATH), "platt_params.pkl")

def save_platt_params(A, B, path=None):
    if path is None:
        path = get_default_params_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"A": A, "B": B}, path)
    print(f"Platt parameters saved to {path}")

def load_platt_params(path=None):
    if path is None:
        path = get_default_params_path()
    if not os.path.exists(path):
        print("Warning: platt_params.pkl not found. Using identity (no calibration).")
        return 1.0, 0.0
    params = joblib.load(path)
    return params["A"], params["B"]

def apply_platt_scaling(raw_conf, A=None, B=None):
    if A is None or B is None:
        A, B = load_platt_params()
    z = logit(np.atleast_1d(raw_conf))
    return sigmoid(A * z + B)[0]

# ----------------------------------------------------------------------
# Evaluation metric (ECE)
# ----------------------------------------------------------------------
def expected_calibration_error(confidences, accuracies, n_bins=10):
    """Expected Calibration Error (lower is better)."""
    confidences = np.asarray(confidences)
    accuracies = np.asarray(accuracies)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
        if np.sum(in_bin) > 0:
            avg_conf = np.mean(confidences[in_bin])
            avg_acc = np.mean(accuracies[in_bin])
            ece += np.sum(in_bin) * np.abs(avg_acc - avg_conf) / len(confidences)
    return ece