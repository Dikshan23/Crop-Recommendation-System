"""
src/preprocess.py
=================
AgroTree preprocessing pipeline.

Final pipeline:
    1. Fix impossible values
    2. IQR winsorisation
    3. Median imputation
    4. Log1p transform (K, rainfall)
    5. Train/test/validation split

StandardScaler removed because CART is scale invariant.

Saves:
    - processed dataset
    - crop medians
    - IQR fences
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_DATASET_PATH,
    MEDIANS_PATH,
    FENCES_PATH
)

from src.logger import logger


FEATURE_COLUMNS = [
    'N',
    'P',
    'K',
    'temperature',
    'humidity',
    'ph',
    'rainfall'
]

LOG_COLUMNS = [
    'K',
    'rainfall'
]

TARGET_COLUMN = 'label'


TEST_SIZE = 0.20
VAL_SIZE = 0.125

RANDOM_STATE = 42



# =====================================================
# Step 1
# =====================================================

def fix_impossible_values(df):

    df = df.copy()


    humidity_count = (
        df['humidity'] > 100
    ).sum()


    if humidity_count:

        df['humidity'] = df['humidity'].clip(
            upper=100
        )


        logger.info(
            f"  humidity fixed: {humidity_count}"
        )


    nitrogen_count = (
        df['N'] == 0
    ).sum()


    if nitrogen_count:

        df.loc[
            df['N'] == 0,
            'N'
        ] = np.nan


        logger.info(
            f"  N==0 converted: {nitrogen_count}"
        )


    return df



# =====================================================
# Step 2
# =====================================================

def compute_iqr_fences(df):

    fences = {}


    for col in FEATURE_COLUMNS:

        values = df[col].dropna()


        q1 = values.quantile(0.25)

        q3 = values.quantile(0.75)

        iqr = q3 - q1


        fences[col] = (
            q1 - 1.5 * iqr,
            q3 + 1.5 * iqr
        )


    return fences



def apply_iqr_winsorisation(df, fences):

    df = df.copy()

    total = 0


    for col in FEATURE_COLUMNS:

        lower, upper = fences[col]


        count = (
            (df[col] < lower) |
            (df[col] > upper)
        ).sum()


        total += count


        df[col] = df[col].clip(
            lower,
            upper
        )


    logger.info(
        f"  IQR clipped values: {total}"
    )


    return df



# =====================================================
# Step 3
# =====================================================

def impute_missing_values(df, medians=None):

    df = df.copy()


    if medians is None:

        medians = {
            col: df[col].median()
            for col in FEATURE_COLUMNS
        }



    nulls = df[FEATURE_COLUMNS].isnull().sum(axis=1)


    complete = df[nulls == 0]

    single = df[nulls == 1]

    dropped = df[nulls >= 2]


    for col in FEATURE_COLUMNS:

        mask = single[col].isnull()


        single.loc[
            mask,
            col
        ] = medians[col]



    cleaned = pd.concat(
        [
            complete,
            single
        ]
    )


    logger.info(
        f"  complete: {len(complete)}, "
        f"imputed: {len(single)}, "
        f"dropped: {len(dropped)}"
    )


    return cleaned, medians



# =====================================================
# Step 4
# =====================================================

def apply_log_transform(df):

    df = df.copy()


    for col in LOG_COLUMNS:

        df[col] = np.log1p(
            df[col]
        )


    logger.info(
        "  log1p applied to K and rainfall"
    )


    return df



# =====================================================
# Split without scaling
# =====================================================

def scale_and_split(
        df,
        scaler_save_path,
        train_idx=None,
        val_idx=None,
        test_idx=None
):

    if (
        train_idx is not None
        and val_idx is not None
        and test_idx is not None
    ):

        train_df = df.loc[train_idx]

        val_df = df.loc[val_idx]

        test_df = df.loc[test_idx]


        X_train = train_df[FEATURE_COLUMNS].values

        y_train = train_df[TARGET_COLUMN].values


        X_val = val_df[FEATURE_COLUMNS].values

        y_val = val_df[TARGET_COLUMN].values


        X_test = test_df[FEATURE_COLUMNS].values

        y_test = test_df[TARGET_COLUMN].values


    else:

        X = df[FEATURE_COLUMNS].values

        y = df[TARGET_COLUMN].values


        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )


        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=VAL_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_temp
        )


    logger.info(
        "  StandardScaler skipped: CART does not require scaling"
    )


    logger.info(
        f"  train : {len(X_train)}"
    )

    logger.info(
        f"  test  : {len(X_test)}"
    )

    logger.info(
        f"  val   : {len(X_val)}"
    )


    scaler = None


    return (
        X_train,
        X_test,
        X_val,
        y_train,
        y_test,
        y_val,
        scaler
    )



# =====================================================
# Full pipeline
# =====================================================

def full_pipeline(path, scaler_save_path):


    logger.info(
        "Running preprocessing pipeline..."
    )


    df = pd.read_csv(path)


    logger.info(
        f"Raw dataset: {df.shape}"
    )


    df = fix_impossible_values(df)



    indices = np.arange(
        len(df)
    )


    idx_temp, idx_test = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN]
    )


    idx_train, idx_val = train_test_split(
        idx_temp,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df.loc[idx_temp, TARGET_COLUMN]
    )



    fences = compute_iqr_fences(
        df.loc[idx_train]
    )


    df = apply_iqr_winsorisation(
        df,
        fences
    )



    _, medians = impute_missing_values(
        df.loc[idx_train]
    )


    df, _ = impute_missing_values(
        df,
        medians
    )



    df = apply_log_transform(
        df
    )



    valid_train = [
        i for i in idx_train
        if i in df.index
    ]

    valid_val = [
        i for i in idx_val
        if i in df.index
    ]

    valid_test = [
        i for i in idx_test
        if i in df.index
    ]



    os.makedirs(
        os.path.dirname(PROCESSED_DATASET_PATH),
        exist_ok=True
    )


    df.to_csv(
        PROCESSED_DATASET_PATH,
        index=False
    )


    joblib.dump(
        medians,
        MEDIANS_PATH
    )


    joblib.dump(
        fences,
        FENCES_PATH
    )



    return scale_and_split(
        df,
        scaler_save_path,
        train_idx=valid_train,
        val_idx=valid_val,
        test_idx=valid_test
    ) + (
        medians,
        fences
    )