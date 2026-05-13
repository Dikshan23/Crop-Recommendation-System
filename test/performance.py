#!/usr/bin/env python3
"""
performance.py
==============
Evaluate trained model and calibration (loads pre‑saved parameters).
"""

import time
import numpy as np
import pandas as pd
from collections import defaultdict

from src.config import DATASET_PATH, SCALER_PATH
from src.preprocess import full_pipeline
from src.model_utils import load_model
from src.evaluation import evaluate_model
from src.calibration import load_platt_params, apply_platt_scaling, expected_calibration_error

# Optional matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

# ----------------------------------------------------------------------
# Feature importance (split frequency)
# ----------------------------------------------------------------------
def compute_feature_importance(tree_model, feature_names):
    importance = defaultdict(int)
    def count_splits(node):
        if node.value is not None:
            return
        if node.feature is not None:
            importance[node.feature] += 1
        if node.left:
            count_splits(node.left)
        if node.right:
            count_splits(node.right)
    count_splits(tree_model.root)
    total = sum(importance.values())
    if total == 0:
        return {f: 1/len(feature_names) for f in feature_names}
    return {feature_names[i]: importance[i]/total for i in range(len(feature_names))}

# ----------------------------------------------------------------------
# Learning curves (using best parameters from CV)
# ----------------------------------------------------------------------
def learning_curves(X_train, y_train, X_val, y_val, best_params,
                    train_sizes=np.linspace(0.1, 1.0, 8), random_state=42):
    from src.model import DecisionTreeCART
    n_total = len(X_train)
    train_scores = []
    val_scores = []
    actual_sizes = []
    rng = np.random.RandomState(random_state)

    for frac in train_sizes:
        n_samples = int(np.ceil(n_total * frac))
        actual_sizes.append(n_samples)

        # stratified subsampling
        indices = []
        for cls in np.unique(y_train):
            cls_idx = np.where(y_train == cls)[0]
            n_cls = max(1, int(np.ceil(len(cls_idx) * frac)))
            n_cls = min(n_cls, len(cls_idx))
            chosen = rng.choice(cls_idx, size=n_cls, replace=False)
            indices.extend(chosen)
        indices = np.array(indices)
        rng.shuffle(indices)

        X_sub = X_train[indices]
        y_sub = y_train[indices]

        model = DecisionTreeCART(**best_params, random_state=random_state)
        model.fit(X_sub, y_sub)

        train_acc = np.mean(model.predict(X_sub) == y_sub)
        val_acc = np.mean(model.predict(X_val) == y_val)

        train_scores.append(train_acc)
        val_scores.append(val_acc)

    return actual_sizes, train_scores, val_scores

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("AgroTree Model Performance Evaluation (with pre‑saved calibration)")
    print("=" * 70 + "\n")

    # 1. Load data
    print("[1] Loading data...")
    start = time.time()
    X_train, X_test, X_val, y_train, y_test, y_val, scaler, _, _ = full_pipeline(
        path=DATASET_PATH,
        scaler_save_path=SCALER_PATH
    )
    print(f"    Data ready in {time.time()-start:.2f}s")
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}\n")

    # 2. Load model
    print("[2] Loading trained model...")
    model = load_model()
    print("    Model loaded.\n")

    # 3. Load Platt parameters
    print("[3] Loading calibration parameters...")
    A, B = load_platt_params()
    print(f"    A = {A:.4f}, B = {B:.4f}\n")

    # 4. Collect test predictions and confidences
    print("[4] Evaluating on test set...")
    test_raw_confs = []
    test_preds = []
    for i in range(len(X_test)):
        proba_dict = model.predict_proba(X_test[i:i+1])[0]
        pred_class = max(proba_dict.items(), key=lambda x: x[1])[0]
        test_preds.append(pred_class)
        test_raw_confs.append(proba_dict[pred_class])
    test_raw_confs = np.array(test_raw_confs)
    test_preds = np.array(test_preds)
    test_correct = (test_preds == y_test).astype(int)
    test_acc = np.mean(test_correct)

    # Calibrated confidences
    test_cal_confs = np.array([apply_platt_scaling(c, A, B) for c in test_raw_confs])

    # Calibration metrics
    ece_raw = expected_calibration_error(test_raw_confs, test_correct)
    ece_cal = expected_calibration_error(test_cal_confs, test_correct)

    print(f"    Accuracy: {test_acc:.4f}")
    print(f"    Uncalibrated avg confidence: {test_raw_confs.mean():.4f}")
    print(f"    Calibrated avg confidence  : {test_cal_confs.mean():.4f}")
    print(f"    ECE (uncalibrated): {ece_raw:.4f}")
    print(f"    ECE (calibrated)  : {ece_cal:.4f}\n")

    # 5. Detailed classification report
    print("[5] Detailed metrics (uncalibrated tree):")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    Macro F1: {metrics['macro_avg']['f1_score']:.4f}")
    print(f"    Weighted F1: {metrics['weighted_avg']['f1_score']:.4f}")
    if metrics.get('low_performance_classes'):
        print(f"    Low F1 classes: {metrics['low_performance_classes']}\n")

    # 6. Feature importance
    print("[6] Feature importance (split frequency):")
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    imp = compute_feature_importance(model, feature_names)
    for feat, val in sorted(imp.items(), key=lambda x: x[1], reverse=True):
        print(f"        {feat:12s} : {val:.4f}")
    print()

    # 7. Learning curves (use best parameters from earlier CV)
    best_params = {'max_depth': 100, 'min_samples_split': 5, 'min_samples_leaf': 5}
    print("[7] Learning curves...")
    sizes, train_accs, val_accs = learning_curves(X_train, y_train, X_val, y_val, best_params)
    print("\n    Training size | Train acc | Validation acc")
    print("    " + "-" * 45)
    for sz, tr, vl in zip(sizes, train_accs, val_accs):
        print(f"    {sz:>12} | {tr:.4f}    | {vl:.4f}")
    if HAS_PLT:
        plt.figure(figsize=(8,5))
        plt.plot(sizes, train_accs, 'o-', label='Train')
        plt.plot(sizes, val_accs, 's-', label='Validation')
        plt.xlabel('Training samples')
        plt.ylabel('Accuracy')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig('learning_curves.png')
        print("\n    Learning curve plot saved as 'learning_curves.png'")
    print()

    # 8. Overfitting check
    train_acc = np.mean(model.predict(X_train) == y_train)
    val_acc = np.mean(model.predict(X_val) == y_val)
    gap = train_acc - val_acc
    print("[8] Overfitting assessment:")
    print(f"    Train accuracy  : {train_acc:.4f}")
    print(f"    Validation acc  : {val_acc:.4f}")
    print(f"    Gap             : {gap:.4f}")
    if gap > 0.05:
        print("    ⚠️  Possible overfitting – consider simplifying the tree.")
    else:
        print("    ✅ Good generalisation.\n")

    print("=" * 70)
    print("Evaluation complete.")
    print("=" * 70)

if __name__ == "__main__":
    main()