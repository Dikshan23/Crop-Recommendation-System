"""
src/train.py
============
Training script with integrated Platt scaling calibration.
"""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from src.config import DATASET_PATH, SCALER_PATH
from src.preprocess import full_pipeline
from src.model_utils import save_model, save_metrics
from src.model import DecisionTreeCART
from src.calibration import fit_platt_scaling, save_platt_params, apply_platt_scaling

def train():
    # ── 1. Preprocessing ──────────────────────────────────────────────────
    X_train, X_test, X_val, y_train, y_test, y_val, scaler, crop_medians, fences = full_pipeline(
        path=DATASET_PATH,
        scaler_save_path=SCALER_PATH,
    )

    # ── 2. Train custom CART model ────────────────────────────────────────
    model = DecisionTreeCART()
    model.fit(X_train, y_train)
    print("Model training complete.\n")

    # ── 3. Basic test evaluation (optional) ───────────────────────────────
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy (uncalibrated): {accuracy:.4f}")

    # ── 4. Collect raw confidences and correctness on validation set ──────
    print("\nCollecting validation set predictions for calibration...")
    raw_confs_val = []
    correctness_val = []
    for i in range(len(X_val)):
        proba_dict = model.predict_proba(X_val[i:i+1])[0]
        pred_class = max(proba_dict.items(), key=lambda x: x[1])[0]
        raw_conf = proba_dict[pred_class]
        raw_confs_val.append(raw_conf)
        correctness_val.append(1 if pred_class == y_val[i] else 0)

    # ── 5. Fit Platt scaling ──────────────────────────────────────────────
    print("Fitting Platt scaling...")
    A, B = fit_platt_scaling(raw_confs_val, correctness_val)
    save_platt_params(A, B)
    print(f"Platt parameters: A={A:.4f}, B={B:.4f}")

    # ── 6. Check calibration improvement on test set ──────────────────────
    test_raw_confs = []
    test_correctness = []
    for i in range(len(X_test)):
        proba_dict = model.predict_proba(X_test[i:i+1])[0]
        pred_class = max(proba_dict.items(), key=lambda x: x[1])[0]
        raw_conf = proba_dict[pred_class]
        test_raw_confs.append(raw_conf)
        test_correctness.append(1 if pred_class == y_test[i] else 0)
    test_raw_confs = np.array(test_raw_confs)
    test_correctness = np.array(test_correctness)

    test_cal_confs = np.array([apply_platt_scaling(c, A, B) for c in test_raw_confs])

    from src.calibration import expected_calibration_error
    ece_raw = expected_calibration_error(test_raw_confs, test_correctness)
    ece_cal = expected_calibration_error(test_cal_confs, test_correctness)

    print(f"\nCalibration results on test set:")
    print(f"  Uncalibrated – avg confidence: {test_raw_confs.mean():.4f}, accuracy: {test_correctness.mean():.4f}, ECE: {ece_raw:.4f}")
    print(f"  Calibrated   – avg confidence: {test_cal_confs.mean():.4f}, accuracy: {test_correctness.mean():.4f}, ECE: {ece_cal:.4f}")

    # ── 7. Save final model and metrics ───────────────────────────────────
    save_model(model)
    save_metrics({
        "test_accuracy": round(float(accuracy), 4),
        "platt_scaling": {"A": float(A), "B": float(B)},
        "ece_uncalibrated": round(float(ece_raw), 4),
        "ece_calibrated": round(float(ece_cal), 4),
        "preprocessing_steps": [
            "fix_impossible_values",
            "per_crop_iqr_winsorisation",
            "per_crop_median_imputation",
            "log1p(K, rainfall)",
            "standard_scaling",
            "platt_scaling_calibration",
        ]
    })

    print("\nTraining complete. Model and calibration saved.")

if __name__ == "__main__":
    train()