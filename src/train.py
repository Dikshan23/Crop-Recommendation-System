"""
src/train.py
============
Training script for AgroTree CART model.

Pipeline:
    Raw data
        ↓
    Preprocessing
        ↓
    Custom CART
        ↓
    Evaluation
        ↓
    Save model
"""

from sklearn.metrics import accuracy_score

from src.config import DATASET_PATH, SCALER_PATH
from src.logger import logger

from src.preprocess import full_pipeline
from src.model_utils import save_model, save_metrics
from src.model import DecisionTreeCART



def train():

    # -------------------------------------------------
    # 1. Preprocessing
    # -------------------------------------------------

    (
        X_train,
        X_test,
        X_val,
        y_train,
        y_test,
        y_val,
        scaler,
        crop_medians,
        fences
    ) = full_pipeline(
        path=DATASET_PATH,
        scaler_save_path=SCALER_PATH
    )


    # -------------------------------------------------
    # 2. Train CART
    # -------------------------------------------------

    model = DecisionTreeCART()

    model.fit(
        X_train,
        y_train
    )

    logger.info(
        "Model training complete.\n"
    )


    # -------------------------------------------------
    # 3. Evaluate model
    # -------------------------------------------------

    y_pred = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    logger.info(
        f"Test Accuracy: {accuracy:.4f}"
    )


    # -------------------------------------------------
    # 4. Save model and metrics
    # -------------------------------------------------

    save_model(
        model
    )


    save_metrics(
        {
            "test_accuracy": round(
                float(accuracy),
                4
            ),

            "preprocessing_steps": [

                "fix_impossible_values",

                "IQR_winsorisation",

                "median_imputation",

                "log1p(K,rainfall)",

                "CART"

            ]
        }
    )


    logger.info(
        "\nTraining complete. Model saved."
    )



if __name__ == "__main__":

    train()