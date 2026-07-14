"""
src/predict.py
==============
Inference module for AgroTree.

Pipeline:

Input
 ↓
Validation
 ↓
Preprocessing
 ↓
CART prediction
 ↓
Confidence output
"""

import pandas as pd


from src.model_utils import (
    load_model,
    load_medians,
    load_fences
)


from src.preprocess import (
    fix_impossible_values,
    apply_iqr_winsorisation,
    impute_missing_values,
    apply_log_transform
)


from src.validations import (
    validate_inputs,
    warn_inputs,
    VALID_RANGES,
    BOUNDARY_MARGIN
)



FEATURE_COLUMNS = [

    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"

]



_model = None
_medians = None
_fences = None



def _get_model():

    global _model

    if _model is None:
        _model = load_model()

    return _model



def _get_medians():

    global _medians

    if _medians is None:
        _medians = load_medians()

    return _medians



def _get_fences():

    global _fences

    if _fences is None:
        _fences = load_fences()

    return _fences



def _compute_penalty(
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
):

    NEAR_BOUNDARY_PENALTY = 0.05
    UNUSUAL_RANGE_PENALTY = 0.03
    MAX_PENALTY = 0.40


    values = {

        "nitrogen": N,
        "phosphorus": P,
        "potassium": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall

    }


    penalty = 0


    for key,value in values.items():

        meta = VALID_RANGES[key]


        margin = (
            meta["max"] -
            meta["min"]
        ) * BOUNDARY_MARGIN


        if (
            value <= meta["min"] + margin
            or
            value >= meta["max"] - margin
        ):

            penalty += NEAR_BOUNDARY_PENALTY



    warnings = warn_inputs(
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    )


    unusual = sum(
        1
        for w in warnings
        if (
            "unusually low" in w
            or
            "unusually high" in w
        )
    )


    penalty += (
        unusual *
        UNUSUAL_RANGE_PENALTY
    )


    return min(
        penalty,
        MAX_PENALTY
    )



def _preprocess_inputs(
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
):


    df = pd.DataFrame(
        [
            {

                "N":N,
                "P":P,
                "K":K,
                "temperature":temperature,
                "humidity":humidity,
                "ph":ph,
                "rainfall":rainfall

            }
        ]
    )


    df = fix_impossible_values(
        df
    )


    df = apply_iqr_winsorisation(
        df,
        _get_fences()
    )


    df,_ = impute_missing_values(
        df,
        medians=_get_medians()
    )


    df = apply_log_transform(
        df
    )


    return df[
        FEATURE_COLUMNS
    ].values



def predict_crop(
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
):


    # -------------------------
    # Validation
    # -------------------------

    errors = validate_inputs(
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    )


    if errors:

        raise ValueError(
            "Invalid input values:\n"
            +
            "\n".join(
                f"• {e}"
                for e in errors
            )
        )



    # -------------------------
    # Preprocessing
    # -------------------------

    features = _preprocess_inputs(
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    )



    # -------------------------
    # Prediction
    # -------------------------

    model = _get_model()


    crop = model.predict(
        features
    )[0]


    probabilities = model.predict_proba(
        features
    )[0]



    confidence = probabilities.get(
        crop,
        0.0
    )



    # confidence reduction for extreme inputs

    penalty = _compute_penalty(
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    )


    confidence *= (
        1 - penalty
    )



    return (

        str(crop),

        float(confidence),

        {
            str(k): float(v)
            for k,v in probabilities.items()
        }

    )