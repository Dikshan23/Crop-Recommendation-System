#!/usr/bin/env python3
"""
check_log_transform.py
======================
Analyze feature distributions and recommend log1p transformation.
"""

import numpy as np
import pandas as pd
import os
from src.config import DATASET_PATH

def main():
    print("=" * 60)
    print("Log‑transformation analysis for AgroTree features")
    print("=" * 60)

    # Load raw dataset
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH)
    print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")

    # Identify numeric columns (excluding target 'label')
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'label' in df.columns:
        # label is usually categorical, but if it's numeric (encoded), skip it
        numeric_cols = [c for c in numeric_cols if c != 'label']
    
    print("Numeric features to analyze:")
    for col in numeric_cols:
        print(f"  • {col}")
    print()

    # Compute skewness
    skewness = {}
    for col in numeric_cols:
        # Drop NaN (shouldn't be any in raw, but just in case)
        data = df[col].dropna()
        skew = data.skew()
        skewness[col] = skew

    # Determine recommended columns for log1p
    recommended = []
    borderline = []
    for col, skew in skewness.items():
        if skew > 1.0:
            recommended.append((col, skew))
        elif skew > 0.5:
            borderline.append((col, skew))

    # Print results
    print("Skewness values (higher = more right‑skewed):")
    for col, skew in sorted(skewness.items(), key=lambda x: x[1], reverse=True):
        print(f"  {col:12s} : {skew:.3f}")

    print("\n" + "-" * 60)
    if recommended:
        print(f"\n✅ Features with skew > 1.0 (strongly recommend log1p):")
        for col, skew in recommended:
            print(f"   • {col} (skew = {skew:.3f})")
    else:
        print("\n✅ No feature has skew > 1.0 – log1p may not be necessary.")

    if borderline:
        print(f"\n⚠️ Features with skew between 0.5 and 1.0 (consider log1p):")
        for col, skew in borderline:
            print(f"   • {col} (skew = {skew:.3f})")

    # Compare with current LOG_COLUMNS in preprocess.py
    print("\n" + "-" * 60)
    try:
        from src.preprocess import LOG_COLUMNS
        print(f"\nCurrent LOG_COLUMNS in preprocess.py: {LOG_COLUMNS}")
        current_set = set(LOG_COLUMNS)
        recommended_set = set([col for col, _ in recommended])

        if current_set == recommended_set:
            print("✅ Current LOG_COLUMNS matches the recommendation.")
        else:
            print("⚠️ Current LOG_COLUMNS differs from recommendation.")
            if not recommended_set.issubset(current_set):
                missing = recommended_set - current_set
                print(f"   Missing from LOG_COLUMNS: {missing}")
            if not current_set.issubset(recommended_set):
                extra = current_set - recommended_set
                print(f"   Extra in LOG_COLUMNS (maybe not needed): {extra}")
    except ImportError:
        print("\nCould not import src.preprocess.LOG_COLUMNS to compare.")
    except Exception as e:
        print(f"\nError comparing with preprocess.py: {e}")

    # Optional plotting
    try:
        import matplotlib.pyplot as plt
        n_cols = len(numeric_cols)
        if n_cols > 0:
            fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 4))
            if n_cols == 1:
                axes = [axes]
            for ax, col in zip(axes, numeric_cols):
                ax.hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                ax.set_title(f"{col}\nskew = {skewness[col]:.2f}")
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
            plt.tight_layout()
            plt.savefig("feature_histograms.png", dpi=150)
            print("\n📊 Histograms saved as 'feature_histograms.png'")
            plt.show()
    except ImportError:
        print("\n📊 Matplotlib not installed – skipping plots.")
    except Exception as e:
        print(f"\nCould not create plots: {e}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()