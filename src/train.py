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

from src.config import DATASET_PATH, SCALER_PATH
from src.logger import logger

from src.preprocess import full_pipeline
from src.model_utils import save_model, save_metrics
from src.model import DecisionTreeCART
from src.evaluation import evaluate_model
from src.cross_validation import cross_validate



def _to_serializable(value):
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return _to_serializable(value.tolist())
        except TypeError:
            pass
    if hasattr(value, "item"):
        try:
            return _to_serializable(value.item())
        except Exception:
            pass
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)



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
    # 2. Cross-validation before final training
    # -------------------------------------------------

    param_grid = {
    "max_depth": [10, 15],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 3],
    "random_state": [42],
}

    best_params, best_score, cv_results = cross_validate(
        DecisionTreeCART,
        X_train,
        y_train,
        param_grid=param_grid,
        n_splits=5,
        random_state=42
    )

    best_cv_result = max(
        cv_results,
        key=lambda result: result["mean_score"]
    )

    print("Cross-validation results:")
    for fold_idx, fold_score in enumerate(best_cv_result["fold_scores"], start=1):
        print(f"Fold {fold_idx} accuracy: {fold_score:.4f}")
    print(f"Mean accuracy: {best_cv_result['mean_score']:.4f}")
    print(f"Standard deviation: {best_cv_result['std_score']:.4f}")

    logger.info(
        f"Cross-validation best params: {best_params}"
    )
    logger.info(
        f"Cross-validation mean accuracy: {best_cv_result['mean_score']:.4f}"
    )
    logger.info(
        f"Cross-validation std: {best_cv_result['std_score']:.4f}"
    )


    # -------------------------------------------------
    # 3. Train CART
    # -------------------------------------------------

    model = DecisionTreeCART(**best_params)

    model.fit(
        X_train,
        y_train
    )

    logger.info(
        "Model training complete.\n"
    )


    # -------------------------------------------------
    # 4. Evaluate model
    # -------------------------------------------------

    evaluation_results = evaluate_model(
        model,
        X_test,
        y_test,
        label_names=None
    )

    serializable_results = _to_serializable(evaluation_results)
    serializable_results["confusion_matrix"] = evaluation_results["confusion_matrix"].tolist()
    serializable_results["labels"] = [str(label) for label in evaluation_results["labels"]]

    accuracy = serializable_results["accuracy"]
    macro_avg = serializable_results["macro_avg"]
    weighted_avg = serializable_results["weighted_avg"]
    low_performance_classes = serializable_results.get("low_performance_classes", [])

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {macro_avg['precision']:.4f}")
    print(f"Macro Recall: {macro_avg['recall']:.4f}")
    print(f"Macro F1-score: {macro_avg['f1_score']:.4f}")
    print(f"Weighted Precision: {weighted_avg['precision']:.4f}")
    print(f"Weighted Recall: {weighted_avg['recall']:.4f}")
    print(f"Weighted F1-score: {weighted_avg['f1_score']:.4f}")

    if low_performance_classes:
        print("Low-performance classes:", ", ".join(low_performance_classes))
    else:
        print("Low-performance classes: None")

    logger.info(
        f"Test Accuracy: {accuracy:.4f}"
    )
    logger.info(
        f"Macro Precision: {macro_avg['precision']:.4f}"
    )
    logger.info(
        f"Macro Recall: {macro_avg['recall']:.4f}"
    )
    logger.info(
        f"Macro F1-score: {macro_avg['f1_score']:.4f}"
    )
    logger.info(
        f"Weighted Precision: {weighted_avg['precision']:.4f}"
    )
    logger.info(
        f"Weighted Recall: {weighted_avg['recall']:.4f}"
    )
    logger.info(
        f"Weighted F1-score: {weighted_avg['f1_score']:.4f}"
    )
    logger.info(
        f"Low-performance classes: {', '.join(low_performance_classes) if low_performance_classes else 'None'}"
    )


    # -------------------------------------------------
    # 5. Save model and metrics
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

            ],
            **serializable_results
        }
    )


    logger.info(
        "\nTraining complete. Model saved."
    )



if __name__ == "__main__":

    train()