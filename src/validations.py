"""
src/validations.py
==================
Input validation and borderline warnings — calibrated from EDA crop means.

Per-crop mean ranges (from dataset):
    N           : 18.8 – 117.8  (mean 50.6)
    P           : 16.6 – 134.2  (mean 53.4)   ← apple/grapes need ~134
    K           : 10.0 – 200.1  (mean 48.1)   ← apple/grapes need ~200
    temperature : 18.9 – 33.7   (mean 25.6)
    humidity    : 16.9 – 94.8   (mean 71.5)
    ph          : 5.8  – 7.3    (mean 6.5)
    rainfall    : 24.7 – 236.2  (mean 103.5)  ← rice needs ~236
"""

# ── Valid hard ranges ─────────────────────────────────────────────────────────
VALID_RANGES = {
    "nitrogen":    {"min":  1.0,  "max": 155.0, "unit": "mg/kg", "label": "Nitrogen (N)"},
    "phosphorus":  {"min": 10.0,  "max": 145.0, "unit": "mg/kg", "label": "Phosphorus (P)"},
    "potassium":   {"min": 10.0,  "max": 240.0, "unit": "mg/kg", "label": "Potassium (K)"},
    "temperature": {"min":  8.0,  "max":  45.0, "unit": "°C",    "label": "Temperature"},
    "humidity":    {"min": 14.0,  "max": 100.0, "unit": "%",     "label": "Humidity"},
    "ph":          {"min":  3.5,  "max":   9.5, "unit": "",      "label": "Soil pH"},
    "rainfall":    {"min": 20.0,  "max": 850.0, "unit": "mm",    "label": "Rainfall"},
}

# 8% boundary margin
BOUNDARY_MARGIN = 0.08

# ── Agronomically unusual ranges ──────────────────────────────────────────────
# Calibrated from actual per-crop means + a small buffer so crops that
# genuinely need extreme values (apple→P=134, grapes→K=200, rice→rain=236)
# do NOT trigger false warnings.
UNUSUAL_RANGES = {
    "nitrogen":    {"warn_below":  5.0,  "warn_above": 125.0},  # cotton mean=117 → buffer to 125
    "phosphorus":  {"warn_below": 12.0,  "warn_above": 138.0},  # apple mean=134  → buffer to 138
    "potassium":   {"warn_below":  8.0,  "warn_above": 205.0},  # apple mean=200  → buffer to 205
    "temperature": {"warn_below": 17.0,  "warn_above":  36.0},  # papaya mean=33.7→ buffer to 36
    "humidity":    {"warn_below": 14.0,  "warn_above":  97.0},  # coconut mean=94 → buffer to 97
    "ph":          {"warn_below":  5.5,  "warn_above":   7.5},  # chickpea mean=7.34→buffer to 7.5
    "rainfall":    {"warn_below": 22.0,  "warn_above": 250.0},  # rice mean=236   → buffer to 250
}


def validate_inputs(N, P, K, temperature, humidity, ph, rainfall):
    """
    Hard validation — blocks prediction if any value is outside VALID_RANGES.
    Returns list of error strings (empty = all valid).
    """
    values = {
        "nitrogen":    N,
        "phosphorus":  P,
        "potassium":   K,
        "temperature": temperature,
        "humidity":    humidity,
        "ph":          ph,
        "rainfall":    rainfall,
    }

    errors = []

    for key, value in values.items():
        meta    = VALID_RANGES[key]
        label   = meta["label"]
        min_val = meta["min"]
        max_val = meta["max"]
        unit    = f" {meta['unit']}" if meta["unit"] else ""

        if value is None:
            errors.append(f"{label}: Value is required.")
        elif not isinstance(value, (int, float)):
            errors.append(f"{label}: Must be a number.")
        elif value < min_val or value > max_val:
            errors.append(
                f"{label}: {value} is out of range. "
                f"Expected between {min_val} and {max_val}{unit}."
            )

    return errors


def warn_inputs(N, P, K, temperature, humidity, ph, rainfall):
    """
    Soft warnings — value is valid but near a boundary or agronomically unusual.
    Does not block prediction. Used to inform user and reduce displayed confidence.
    Returns list of warning strings.
    """
    values = {
        "nitrogen":    N,
        "phosphorus":  P,
        "potassium":   K,
        "temperature": temperature,
        "humidity":    humidity,
        "ph":          ph,
        "rainfall":    rainfall,
    }

    warnings = []

    for key, value in values.items():
        meta    = VALID_RANGES[key]
        label   = meta["label"]
        min_val = meta["min"]
        max_val = meta["max"]
        unit    = f" {meta['unit']}" if meta["unit"] else ""
        margin  = (max_val - min_val) * BOUNDARY_MARGIN
        unusual = UNUSUAL_RANGES[key]

        # Near lower hard boundary
        if value <= min_val + margin:
            warnings.append(
                f"{label}: {value}{unit} is near the lower boundary ({min_val}{unit}). "
                f"Prediction confidence may be reduced."
            )
        # Near upper hard boundary
        elif value >= max_val - margin:
            warnings.append(
                f"{label}: {value}{unit} is near the upper boundary ({max_val}{unit}). "
                f"Prediction confidence may be reduced."
            )
        # Agronomically unusual — too low
        elif value < unusual["warn_below"]:
            warnings.append(
                f"{label}: {value}{unit} is unusually low for typical crop conditions."
            )
        # Agronomically unusual — too high
        elif value > unusual["warn_above"]:
            warnings.append(
                f"{label}: {value}{unit} is unusually high for typical crop conditions."
            )

    return warnings


def is_valid(N, P, K, temperature, humidity, ph, rainfall):
    """Quick boolean — True if all inputs pass hard validation."""
    return len(validate_inputs(N, P, K, temperature, humidity, ph, rainfall)) == 0