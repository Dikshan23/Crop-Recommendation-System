from sklearn.metrics import accuracy_score, classification_report

from src.model_utils import load_dataset, save_model, save_metrics
from src.preprocess import split_features_labels, train_test_data
from src.model import DecisionTreeCART


def train():
    # 1. Load dataset
    df = load_dataset()

    # 2. Split features and labels
    X, y = split_features_labels(df)

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_data(X, y)

    # 4. Train custom CART model
    model = DecisionTreeCART()
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 5. Evaluate
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"\nClassification Report:\n{report}")

    # 6. Save model and metrics
    save_model(model)
    save_metrics({
        "accuracy": round(float(accuracy), 4),
        "classification_report": report
    })

    print("\nTraining pipeline complete.")


if __name__ == '__main__':
    train()