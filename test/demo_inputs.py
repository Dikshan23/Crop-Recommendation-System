#!/usr/bin/env python3
"""
test/generate_demo_predictions.py
==================================
Builds one demo input set per crop (using per-crop mean values from the
raw training dataset), runs each through the real, gated inference path
(src.inference_pipeline.run_inference — same entry point the dashboard
uses), and writes everything to results/demo_predictions.json.

Usage:
    python test/generate_demo_predictions.py

Place this file in the `test/` folder so the sys.path append below
resolves the project root correctly.
"""

import json
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import DATASET_PATH
from src.inference_pipeline import run_inference, PredictionBlocked

# Raw dataset columns -> run_inference() kwarg names
FEATURE_MAP = {
    "N": "nitrogen",
    "P": "phosphorus",
    "K": "potassium",
    "temperature": "temperature",
    "humidity": "humidity",
    "ph": "ph",
    "rainfall": "rainfall",
}

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "demo_predictions.json"
)


def build_demo_inputs():
    """One representative (mean) input row per crop label, from raw data."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the raw dataset.")

    missing = [c for c in FEATURE_MAP if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    demo_inputs = {}
    grouped = df.groupby("label")[list(FEATURE_MAP.keys())].mean()

    for crop, row in grouped.iterrows():
        demo_inputs[str(crop)] = {
            api_name: round(float(row[col]), 2)
            for col, api_name in FEATURE_MAP.items()
        }

    return demo_inputs


def run_demo(demo_inputs):
    """Feed every demo input through the real inference gate."""
    results = {}

    for crop, inputs in demo_inputs.items():
        try:
            output = run_inference(
                N=inputs["nitrogen"],
                P=inputs["phosphorus"],
                K=inputs["potassium"],
                temperature=inputs["temperature"],
                humidity=inputs["humidity"],
                ph=inputs["ph"],
                rainfall=inputs["rainfall"],
            )
            predicted = output["crop"]
            match = predicted.strip().lower() == crop.strip().lower()

            results[crop] = {
                "input": inputs,
                "predicted_crop": predicted,
                "confidence": round(output["confidence"], 4),
                "match_expected": match,
                "warnings": output["warnings"],
            }

        except PredictionBlocked as e:
            results[crop] = {
                "input": inputs,
                "error": "blocked_by_hard_validation",
                "details": e.args[0],
            }
        except Exception as e:
            results[crop] = {
                "input": inputs,
                "error": str(e),
            }

    return results


def main():
    print("Building demo inputs from per-crop dataset means...")
    demo_inputs = build_demo_inputs()
    print(f"Found {len(demo_inputs)} crops.\n")

    print("Running each demo input through run_inference()...")
    results = run_demo(demo_inputs)

    matched = sum(1 for r in results.values() if r.get("match_expected"))
    errored = sum(1 for r in results.values() if "error" in r)
    total = len(demo_inputs)

    report = {
        "summary": {
            "total_crops": total,
            "correct_predictions": matched,
            "errors": errored,
            "accuracy": round(matched / total, 4) if total else 0.0,
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{matched}/{total} crops matched their expected label "
          f"({report['summary']['accuracy'] * 100:.1f}% accuracy).")
    if errored:
        print(f"{errored} crop(s) raised an error — see JSON for details.")
    print(f"Results written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()