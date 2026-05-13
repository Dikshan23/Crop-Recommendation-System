"""
src/preprocess.py
=================
Full preprocessing pipeline for AgroTree crop recommendation model.

Pipeline steps (in order):
    1. Fix impossible values  — humidity > 100 clamped, N == 0 set to NaN
    2. IQR winsorisation      — per-crop outlier clipping using 1.5×IQR fences
    3. Median imputation      — per-crop median for missing values
                                (0 nulls → keep, 1 null → impute, 2+ → drop)
    4. Log1p transform        — applied to K and rainfall (right-skewed)
    5. Standard scaling       — fit on train set only, applied to both splits
    6. Train/test split       — 80/20, stratified by label

Saves:
    - data/processed/Processed_data.csv  — cleaned dataset (reused on next run)
    - models/scaler.pkl                  — StandardScaler fitted on train set
    - models/crop_medians.pkl            — per-crop medians (used in inference)
    - models/iqr_fences.pkl              — per-crop IQR fences (used in inference)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DATASET_PATH, MEDIANS_PATH, FENCES_PATH

FEATURE_COLUMNS  = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
LOG_COLUMNS      = ['K', 'rainfall']   # right-skewed — benefit from log1p
TARGET_COLUMN    = 'label'
TEST_SIZE        = 0.20   # 20% of total → test set
VAL_SIZE         = 0.125  # 12.5% of remaining 80% → 10% of total → validation
RANDOM_STATE     = 42


# ── Step 1: Fix impossible values ─────────────────────────────────────────────

def fix_impossible_values(df):
    """
    Corrects values that are physically impossible before any other step.

    Rules:
        - humidity > 100  → clamp to 100.0  (relative humidity cannot exceed 100%)
        - N == 0          → set to NaN       (zero nitrogen is unrealistic; treat as missing)

    Args:
        df (DataFrame): Raw dataset

    Returns:
        df (DataFrame): Dataset with impossible values fixed
    """
    df = df.copy()

    over_100 = (df['humidity'] > 100).sum()
    if over_100:
        df['humidity'] = df['humidity'].clip(upper=100.0)
        print(f"  fix_impossible_values: clamped {over_100} humidity > 100 rows")

    zero_n = (df['N'] == 0).sum()
    if zero_n:
        df.loc[df['N'] == 0, 'N'] = np.nan
        print(f"  fix_impossible_values: set {zero_n} N==0 rows to NaN")

    return df


# ── Step 2: Per-crop IQR winsorisation ────────────────────────────────────────

def compute_iqr_fences(df):
    """
    Computes per-crop IQR fences (Q1 - 1.5×IQR, Q3 + 1.5×IQR) for each
    feature column. Fences are computed on non-null values only.

    Args:
        df (DataFrame): Dataset with label column present

    Returns:
        fences (dict): {crop: {col: (lower_fence, upper_fence)}}
    """
    fences = {}

    for crop, group in df.groupby(TARGET_COLUMN):
        fences[crop] = {}
        for col in FEATURE_COLUMNS:
            vals = group[col].dropna()
            q1   = vals.quantile(0.25)
            q3   = vals.quantile(0.75)
            iqr  = q3 - q1
            fences[crop][col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    return fences


def apply_iqr_winsorisation(df, fences):
    """
    Clips each feature value to the per-crop IQR fence.
    Values outside the fence are clipped to the fence boundary,
    not removed — this preserves row count while reducing outlier influence.

    Args:
        df     (DataFrame): Dataset
        fences (dict):      Output of compute_iqr_fences()

    Returns:
        df (DataFrame): Winsorised dataset
    """
    df = df.copy()
    total_clipped = 0

    for crop, group_idx in df.groupby(TARGET_COLUMN).groups.items():
        for col in FEATURE_COLUMNS:
            lower, upper = fences[crop][col]
            col_vals     = df.loc[group_idx, col]
            clipped      = ((col_vals < lower) | (col_vals > upper)).sum()
            total_clipped += clipped
            df.loc[group_idx, col] = col_vals.clip(lower=lower, upper=upper)

    print(f"  iqr_winsorisation: clipped {total_clipped} outlier values across all crops/features")
    return df


# ── Step 3: Per-crop median imputation ────────────────────────────────────────

def impute_missing_values(df):
    """
    Handles missing values based on null count per row.

    Strategy:
        - 0 nulls  → keep as-is
        - 1 null   → impute with per-crop median of that feature
        - 2+ nulls → drop (too many missing values to impute reliably)

    Per-crop median is used instead of global median because crops within
    the same class share similar soil and climate profiles. A missing N
    value for rice should be filled with rice's median N, not the median
    of all 22 crops combined.

    Args:
        df (DataFrame): Dataset after fixing impossible values and winsorising

    Returns:
        df_clean    (DataFrame): Cleaned dataset
        crop_medians (dict):     {crop: {col: median}} — saved for inference
    """
    # Compute per-crop medians before any imputation
    crop_medians = {}
    for crop, group in df.groupby(TARGET_COLUMN):
        crop_medians[crop] = {}
        for col in FEATURE_COLUMNS:
            crop_medians[crop][col] = group[col].median()

    original_count = len(df)
    null_counts     = df.isnull().sum(axis=1)
    complete        = df[null_counts == 0].copy()
    single_null     = df[null_counts == 1].copy()
    multi_null      = df[null_counts >= 2]

    # Impute single-null rows using per-crop median
    for col in FEATURE_COLUMNS:
        null_mask = single_null[col].isnull()
        if null_mask.any():
            single_null.loc[null_mask, col] = single_null.groupby(TARGET_COLUMN)[col].transform(
                lambda x: x.fillna(x.median())
            )[null_mask]

    df_clean = pd.concat([complete, single_null], ignore_index=True)

    print(f"  imputation: {len(complete)} complete, "
          f"{len(single_null)} imputed (1 null), "
          f"{len(multi_null)} dropped (2+ nulls) → "
          f"{len(df_clean)} usable rows")

    return df_clean, crop_medians


# ── Step 4: Log1p transform ───────────────────────────────────────────────────

def apply_log_transform(df):
    """
    Applies log1p transform to right-skewed columns (K and rainfall).
    log1p(x) = log(1 + x) — safe for zero values, reduces right skew.

    Args:
        df (DataFrame): Dataset after imputation

    Returns:
        df (DataFrame): Dataset with log-transformed K and rainfall
    """
    df = df.copy()
    for col in LOG_COLUMNS:
        df[col] = np.log1p(df[col])
    print(f"  log1p transform applied to: {LOG_COLUMNS}")
    return df


# ── Step 5 & 6: Scale and split ───────────────────────────────────────────────

def scale_and_split(df, scaler_save_path):
    """
    Performs a 70/20/10 train/test/validation split then fits StandardScaler
    on the train set only. Scaler is applied to all three sets after fitting.

    Split logic:
        Step 1 — split full dataset into 80% temp and 20% test
        Step 2 — split 80% temp into 87.5% train and 12.5% val
                 → 87.5% of 80% = 70% of total  (train)
                 → 12.5% of 80% = 10% of total  (val)
                 → 20% of total                  (test)

    Fitting on train only prevents data leakage — test and val set
    statistics must not influence the scaler parameters.

    Args:
        df               (DataFrame): Fully cleaned and transformed dataset
        scaler_save_path (str):       Path to save the fitted scaler

    Returns:
        X_train, X_test, X_val (np.ndarray): Scaled feature arrays
        y_train, y_test, y_val (np.ndarray): Label arrays
        scaler                 (StandardScaler): Fitted scaler
    """
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    # Step 1 — split off test set (20% of total)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Step 2 — split remaining 80% into train (87.5%) and val (12.5%)
    #           → 70% train and 10% val of total
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_temp
    )

    # Fit scaler on train only, apply to all three sets
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    X_val   = scaler.transform(X_val)

    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    print(f"  scaler fitted and saved to: {scaler_save_path}")
    print(f"  train : {len(X_train)} rows (70%)")
    print(f"  test  : {len(X_test)} rows (20%)")
    print(f"  val   : {len(X_val)} rows (10%)")

    return X_train, X_test, X_val, y_train, y_test, y_val, scaler


# ── Processed dataset cache ───────────────────────────────────────────────────

def _save_processed_dataset(df):
    os.makedirs(os.path.dirname(PROCESSED_DATASET_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATASET_PATH, index=False)
    print(f"  processed dataset saved to: {PROCESSED_DATASET_PATH}")


def _load_processed_dataset():
    if os.path.exists(PROCESSED_DATASET_PATH):
        print(f"Processed dataset found. Loading from: {PROCESSED_DATASET_PATH}")
        df = pd.read_csv(PROCESSED_DATASET_PATH)
        print(f"  loaded rows: {len(df)}")
        return df
    return None


# ── Full pipeline (entry point for train.py) ──────────────────────────────────

def full_pipeline(path, scaler_save_path):
    """
    Runs the complete preprocessing pipeline and returns train/test splits.

    On first run:
        - Loads raw dataset from `path`
        - Runs all cleaning steps (fix → winsorise → impute → log)
        - Saves cleaned dataset to PROCESSED_DATASET_PATH for future runs
        - Saves fences and medians to models/

    On subsequent runs:
        - Loads cleaned dataset directly from PROCESSED_DATASET_PATH
        - Skips all cleaning steps
        - Still recomputes scaler (fit on current train split)

    Args:
        path             (str): Path to raw dataset CSV
        scaler_save_path (str): Path to save the fitted StandardScaler

    Returns:
        X_train      (np.ndarray): 70% of data
        X_test       (np.ndarray): 20% of data
        X_val        (np.ndarray): 10% of data
        y_train      (np.ndarray)
        y_test       (np.ndarray)
        y_val        (np.ndarray)
        scaler       (StandardScaler)
        crop_medians (dict): {crop: {col: median}}
        fences       (dict): {crop: {col: (lower, upper)}}
    """
    crop_medians = None
    fences       = None

    # Try loading already-processed dataset
    df_clean = _load_processed_dataset()

    if df_clean is None:
        print("No processed dataset found. Running full preprocessing pipeline...")

        # Load raw
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at: {path}")
        df = pd.read_csv(path)
        print(f"Raw dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Step 1 — fix impossible values
        print("\n[Step 1] Fixing impossible values...")
        df = fix_impossible_values(df)

        # Step 2 — IQR winsorisation (compute fences first)
        print("\n[Step 2] IQR winsorisation...")
        fences = compute_iqr_fences(df)
        df     = apply_iqr_winsorisation(df, fences)

        # Step 3 — median imputation
        print("\n[Step 3] Median imputation...")
        df_clean, crop_medians = impute_missing_values(df)

        # Step 4 — log transform
        print("\n[Step 4] Log1p transform...")
        df_clean = apply_log_transform(df_clean)

        # Save processed dataset and artefacts
        _save_processed_dataset(df_clean)

        os.makedirs(os.path.dirname(MEDIANS_PATH), exist_ok=True)
        joblib.dump(crop_medians, MEDIANS_PATH)
        joblib.dump(fences,       FENCES_PATH)
        print(f"  crop_medians saved to: {MEDIANS_PATH}")
        print(f"  iqr_fences saved to:   {FENCES_PATH}")

    else:
        # Load artefacts saved from previous run
        if os.path.exists(MEDIANS_PATH):
            crop_medians = joblib.load(MEDIANS_PATH)
        if os.path.exists(FENCES_PATH):
            fences = joblib.load(FENCES_PATH)

    # Step 5 & 6 — scale and split (always runs fresh)
    print("\n[Step 5] Scaling and splitting...")
    X_train, X_test, X_val, y_train, y_test, y_val, scaler = scale_and_split(df_clean, scaler_save_path)

    return X_train, X_test, X_val, y_train, y_test, y_val, scaler, crop_medians, fences