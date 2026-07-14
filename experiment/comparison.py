"""
AgroTree Preprocessing Comparison Experiment

Experiment A:
Cleaning
+ Winsorisation
+ log1p(K, rainfall)
+ StandardScaler
+ Custom CART

Experiment B:
Cleaning
+ Winsorisation
+ log1p(K, rainfall)
+ Custom CART

Experiment C:
Cleaning only
+ Custom CART
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# Add project root
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.model import DecisionTreeCART


# =========================
# Configuration
# =========================

DATA_PATH = "data/raw/raw_data.csv"

TARGET = "label"

FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]


RANDOM_STATE = 42


MODEL_PARAMS = {
    "max_depth": 10,
    "min_samples_split": 10,
    "min_samples_leaf": 5
}


# =========================
# Cleaning
# =========================

def clean_data(df):

    df = df.copy()

    print("Cleaning data...")

    humidity_count = (
        df["humidity"] > 100
    ).sum()

    df.loc[
        df["humidity"] > 100,
        "humidity"
    ] = 100


    nitrogen_count = (
        df["N"] == 0
    ).sum()

    df.loc[
        df["N"] == 0,
        "N"
    ] = np.nan


    print(
        f"Humidity fixed: {humidity_count}"
    )

    print(
        f"N==0 converted: {nitrogen_count}"
    )


    return df



# =========================
# Winsorisation
# =========================

def winsorise(df):

    df = df.copy()

    print(
        "Applying winsorisation..."
    )

    for col in FEATURES:

        q1 = df[col].quantile(0.25)

        q3 = df[col].quantile(0.75)

        iqr = q3 - q1


        lower = q1 - (1.5 * iqr)

        upper = q3 + (1.5 * iqr)


        df[col] = np.clip(
            df[col],
            lower,
            upper
        )


    return df



# =========================
# Imputation
# =========================

def impute(df):

    df = df.copy()

    print(
        "Applying median imputation..."
    )

    for col in FEATURES:

        df[col] = df[col].fillna(
            df[col].median()
        )


    return df



# =========================
# Log Transformation
# =========================

def log_transform(df):

    df = df.copy()

    print(
        "Applying log1p transform..."
    )

    for col in [
        "K",
        "rainfall"
    ]:

        df[col] = np.log1p(
            df[col]
        )


    return df



# =========================
# Dataset preparation
# =========================

def prepare_dataset(
        use_winsor,
        use_log,
        use_scaler
):

    df = pd.read_csv(
        DATA_PATH
    )


    print(
        f"Loaded dataset: {df.shape}"
    )


    df = clean_data(df)


    if use_winsor:

        df = winsorise(df)


    df = impute(df)


    if use_log:

        df = log_transform(df)



    X = df[FEATURES]

    y = df[TARGET]



    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )


    # IMPORTANT:
    # Custom CART requires numpy arrays

    X_train = X_train.values

    X_test = X_test.values

    y_train = y_train.values

    y_test = y_test.values



    if use_scaler:

        print(
            "Applying StandardScaler..."
        )


        scaler = StandardScaler()


        X_train = scaler.fit_transform(
            X_train
        )


        X_test = scaler.transform(
            X_test
        )


    return (
        X_train,
        X_test,
        y_train,
        y_test
    )



# =========================
# Run experiment
# =========================

def run_experiment(
        name,
        use_winsor,
        use_log,
        use_scaler
):

    print("\n")
    print("=" * 60)
    print(name)
    print("=" * 60)


    X_train, X_test, y_train, y_test = prepare_dataset(
        use_winsor,
        use_log,
        use_scaler
    )


    print(
        "Training CART..."
    )


    model = DecisionTreeCART(
        **MODEL_PARAMS
    )


    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    print(
        f"Accuracy: {accuracy:.4f}"
    )


    return accuracy



# =========================
# Main
# =========================

if __name__ == "__main__":


    results = {}


    results["Experiment A"] = run_experiment(
        name="Experiment A: Full Pipeline",
        use_winsor=True,
        use_log=True,
        use_scaler=True
    )


    results["Experiment B"] = run_experiment(
        name="Experiment B: No Scaling",
        use_winsor=True,
        use_log=True,
        use_scaler=False
    )


    results["Experiment C"] = run_experiment(
        name="Experiment C: Cleaning Only",
        use_winsor=False,
        use_log=False,
        use_scaler=False
    )



    print("\n")
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)


    for name, score in results.items():

        print(
            f"{name}: {score:.4f}"
        )