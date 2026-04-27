"""
Input validation and borderline warning for crop prediction parameters.
Single source of truth for valid ranges — used by both backend and UI.
"""

# Valid ranges for each input feature
VALID_RANGES = {
    "nitrogen":    {"min": 0,    "max": 140,  "unit": "mg/kg", "label": "Nitrogen (N)"},
    "phosphorus":  {"min": 5,    "max": 145,  "unit": "mg/kg", "label": "Phosphorus (P)"},
    "potassium":   {"min": 5,    "max": 205,  "unit": "mg/kg", "label": "Potassium (K)"},
    "temperature": {"min": 8.0,  "max": 44.0, "unit": "°C",    "label": "Temperature"},
    "humidity":    {"min": 14.0, "max": 100.0,"unit": "%",     "label": "Humidity"},
    "ph":          {"min": 3.5,  "max": 9.5,  "unit": "",      "label": "Soil pH"},
    "rainfall":    {"min": 20.0, "max": 300.0,"unit": "mm",    "label": "Rainfall"},
}

# Borderline threshold — values within this % of the boundary are flagged as warnings
BOUNDARY_MARGIN = 0.08  # 8% of total range

# Agronomically unusual ranges — values technically valid but rarely seen in good predictions
UNUSUAL_RANGES = {
    "nitrogen":    {"warn_below": 10,   "warn_above": 120},
    "phosphorus":  {"warn_below": 10,   "warn_above": 130},
    "potassium":   {"warn_below": 10,   "warn_above": 190},
    "temperature": {"warn_below": 10.0, "warn_above": 42.0},
    "humidity":    {"warn_below": 20.0, "warn_above": 98.0},
    "ph":          {"warn_below": 4.5,  "warn_above": 8.5},
    "rainfall":    {"warn_below": 30.0, "warn_above": 270.0},
}


def validate_inputs(N, P, K, temperature, humidity, ph, rainfall):
    """
    Hard validation — values outside allowed ranges.

    Returns:
        errors (list[str]): Empty means all inputs are valid.
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
        meta = VALID_RANGES[key]
        label = meta["label"]
        min_val = meta["min"]
        max_val = meta["max"]
        unit = f" {meta['unit']}" if meta["unit"] else ""

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
    Soft warnings — values are valid but near boundaries or agronomically unusual.
    These don't block prediction but may reduce confidence.

    Returns:
        warnings (list[str]): Informational messages about suspicious values.
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
        meta = VALID_RANGES[key]
        label = meta["label"]
        min_val = meta["min"]
        max_val = meta["max"]
        unit = f" {meta['unit']}" if meta["unit"] else ""
        margin = (max_val - min_val) * BOUNDARY_MARGIN

        unusual = UNUSUAL_RANGES[key]

        # Near lower boundary
        if value <= min_val + margin:
            warnings.append(
                f"{label}: {value}{unit} is near the lower boundary ({min_val}{unit}). "
                f"Prediction confidence may be reduced."
            )
        # Near upper boundary
        elif value >= max_val - margin:
            warnings.append(
                f"{label}: {value}{unit} is near the upper boundary ({max_val}{unit}). "
                f"Prediction confidence may be reduced."
            )
        # Agronomically unusual — low
        elif value < unusual["warn_below"]:
            warnings.append(
                f"{label}: {value}{unit} is unusually low for typical crop conditions."
            )
        # Agronomically unusual — high
        elif value > unusual["warn_above"]:
            warnings.append(
                f"{label}: {value}{unit} is unusually high for typical crop conditions."
            )

    return warnings


def is_valid(N, P, K, temperature, humidity, ph, rainfall):
    """Quick boolean check — True if all inputs pass hard validation."""
    return len(validate_inputs(N, P, K, temperature, humidity, ph, rainfall)) == 0