#!/usr/bin/env python3
"""
test/roc_auc_analysis.py
=========================
Computes multiclass ROC curves and AUC scores for the AgroTree CART model
using one-vs-rest decomposition (standard approach for multiclass ROC/AUC).

Since DecisionTreeCART is a 22-class classifier, a single ROC curve doesn't
apply directly — each crop is scored as "this class vs. all others" using
the leaf class-distribution probabilities from predict_proba().

Outputs:
    - results/roc_auc_curve.png   (all 22 per-class curves + micro/macro avg)
    - results/roc_auc_summary.csv (AUC per class, sorted lowest to highest)

Usage:
    python test/roc_auc_analysis.py
"""

import os
import sys
from itertools import cycle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DATASET_PATH
from src.model import DecisionTreeCART
from src.preprocess import full_pipeline
from src.logger import logger


OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
PLOT_PATH = os.path.join(OUTPUT_DIR, "roc_auc_curve.png")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "roc_auc_summary.csv")


def build_full_proba_matrix(model, X, classes):
    """
    model.predict_proba() only returns nonzero entries for classes present
    in a leaf's class_distribution. This expands that into a dense
    (n_samples, n_classes) matrix with 0.0 for absent classes, which is
    what sklearn's roc_curve / label_binarize expect.
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    proba_dicts = model.predict_proba(X)

    y_score = np.zeros((len(X), n_classes))
    for i, dist in enumerate(proba_dicts):
        for cls_str, p in dist.items():
            if cls_str in class_to_idx:
                y_score[i, class_to_idx[cls_str]] = p

    return y_score


def compute_roc_auc(y_test, y_score, classes):
    """Compute per-class, micro-average, and macro-average ROC/AUC."""
    n_classes = len(classes)
    y_test_bin = label_binarize(y_test, classes=classes)

    fpr, tpr, roc_auc = {}, {}, {}

    # Per-class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average (pools all classes together)
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average (unweighted mean over classes — treats each class equally)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc


def plot_roc_curves(fpr, tpr, roc_auc, classes, save_path):
    n_classes = len(classes)
    plt.figure(figsize=(10, 9))

    plt.plot(
        fpr["micro"], tpr["micro"],
        label=f"micro-average (AUC = {roc_auc['micro']:.3f})",
        color="deeppink", linestyle=":", linewidth=3,
    )
    plt.plot(
        fpr["macro"], tpr["macro"],
        label=f"macro-average (AUC = {roc_auc['macro']:.3f})",
        color="navy", linestyle=":", linewidth=3,
    )

    colors = cycle(plt.cm.tab20.colors)
    for i, cls, color in zip(range(n_classes), classes, colors):
        plt.plot(
            fpr[i], tpr[i], color=color, lw=1,
            label=f"{cls} (AUC = {roc_auc[i]:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curve — AgroTree CART (One-vs-Rest)")
    plt.legend(loc="lower right", fontsize=7, ncol=2)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    print("=" * 60)
    print("AgroTree — ROC / AUC Analysis")
    print("=" * 60)

    # -------------------------------------------------
    # 1. Preprocess + split (same pipeline as train.py)
    # -------------------------------------------------
    (
        X_train, X_test, X_val,
        y_train, y_test, y_val,
        scaler, crop_medians, fences,
    ) = full_pipeline(
        path=DATASET_PATH,
        scaler_save_path=None,
    )

    # -------------------------------------------------
    # 2. Train CART (same default params as model.py)
    # -------------------------------------------------
    print("\nTraining CART model...")
    model = DecisionTreeCART(
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("Training complete.")

    # -------------------------------------------------
    # 3. Build probability matrix + compute ROC/AUC
    # -------------------------------------------------
    classes = sorted(np.unique(y_train))
    y_score = build_full_proba_matrix(model, X_test, classes)

    fpr, tpr, roc_auc = compute_roc_auc(y_test, y_score, classes)

    print(f"\nMicro-average AUC: {roc_auc['micro']:.4f}")
    print(f"Macro-average AUC: {roc_auc['macro']:.4f}")
    print("\nPer-class AUC:")
    for i, cls in enumerate(classes):
        print(f"  {cls:15s}: {roc_auc[i]:.4f}")

    # -------------------------------------------------
    # 4. Save plot + summary CSV
    # -------------------------------------------------
    plot_roc_curves(fpr, tpr, roc_auc, classes, PLOT_PATH)
    print(f"\nROC plot saved to: {PLOT_PATH}")

    summary = pd.DataFrame({
        "class": list(classes) + ["micro_avg", "macro_avg"],
        "auc": [roc_auc[i] for i in range(len(classes))]
              + [roc_auc["micro"], roc_auc["macro"]],
    }).sort_values("auc")

    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"AUC summary saved to: {SUMMARY_PATH}")

    logger.info(f"ROC/AUC analysis complete. Macro AUC: {roc_auc['macro']:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()