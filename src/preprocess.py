import numpy as np
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET_COLUMN   = 'label'


def split_features_labels(df):
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    print(f"Feature shape: {X.shape} | Target shape: {y.shape}")
    return X, y


def train_test_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test