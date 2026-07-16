import json
import joblib
import pandas as pd

from src.config import (
    DATASET_PATH,
    FENCES_PATH,
    MEDIANS_PATH
)

from src.model_utils import load_model
from src.preprocess import (
    FEATURE_COLUMNS,
    fix_impossible_values,
    apply_iqr_winsorisation,
    impute_missing_values,
    apply_log_transform
)

OUTPUT = "results/demo_inputs.json"
SAMPLES_PER_CROP = 5


def preprocess(sample, fences, medians):
    """
    Apply exactly the same preprocessing
    used during training.
    """

    sample = fix_impossible_values(sample)

    sample = apply_iqr_winsorisation(
        sample,
        fences
    )

    sample, _ = impute_missing_values(
        sample,
        medians
    )

    sample = apply_log_transform(sample)

    return sample


def main():

    print("Loading model...")

    model = load_model()

    fences = joblib.load(FENCES_PATH)

    medians = joblib.load(MEDIANS_PATH)

    df = pd.read_csv(DATASET_PATH)

    demo = {}

    total_correct = 0

    for _, row in df.iterrows():

        crop = row["label"]

        if crop not in demo:
            demo[crop] = []

        if len(demo[crop]) >= SAMPLES_PER_CROP:
            continue

        raw = row.copy()

        sample = pd.DataFrame([raw])

        processed = preprocess(
            sample,
            fences,
            medians
        )

        if processed.empty:
            continue

        X = processed[FEATURE_COLUMNS].values

        prediction = model.predict(X)[0]

        if prediction == crop:

            total_correct += 1

            demo[crop].append({

                "N": round(float(raw["N"]),2),

                "P": round(float(raw["P"]),2),

                "K": round(float(raw["K"]),2),

                "temperature": round(float(raw["temperature"]),2),

                "humidity": round(float(raw["humidity"]),2),

                "ph": round(float(raw["ph"]),2),

                "rainfall": round(float(raw["rainfall"]),2)

            })

    with open(OUTPUT,"w") as f:

        json.dump(
            demo,
            f,
            indent=4
        )

    print("\nSaved:",OUTPUT)

    print("\nSummary\n")

    for crop in sorted(demo):

        print(
            f"{crop:<15} {len(demo[crop])}"
        )


if __name__ == "__main__":

    main()