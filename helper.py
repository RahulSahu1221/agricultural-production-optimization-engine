"""
helper.py
----------------
Small utility functions used by the Flask application: input
validation/parsing for the crop prediction form, and a friendly
welcome message.
"""

# Real min/max values pulled directly from the 2,200-sample training
# dataset (N, P, K in kg/ha soil ratio; temperature in Celsius; humidity
# in %; rainfall in mm). Values outside these ranges are outside anything
# the model has ever seen, so predictions there are not reliable.
VALID_RANGES = {
    "nitrogen": (0, 140),
    "phosphorus": (5, 145),
    "potassium": (5, 205),
    "temperature": (8, 44),
    "humidity": (14, 100),
    "ph": (3.5, 10),
    "rainfall": (20, 300),
}

FORM_TO_FEATURE = {
    "nitrogen": "N",
    "phosphorus": "P",
    "potassium": "K",
    "temperature": "temperature",
    "humidity": "humidity",
    "ph": "ph",
    "rainfall": "rainfall",
}


def welcome():
    return "Welcome to OptiCrop"


def parse_and_validate_form(form):
    """
    Parse the raw Flask request.form data into a features dict ready
    for CropPredictor.predict(), validating each field along the way.

    Returns (features_dict, errors_list, raw_values). If errors_list
    is non-empty, features_dict should not be used for prediction.
    """
    errors = []
    raw_values = {}
    features = {}

    for field, low_high in VALID_RANGES.items():
        raw = (form.get(field) or "").strip()
        raw_values[field] = raw

        if raw == "":
            errors.append(f"{field.capitalize()} is required.")
            continue

        try:
            value = float(raw)
        except ValueError:
            errors.append(f"{field.capitalize()} must be a number.")
            continue

        low, high = low_high
        if not (low <= value <= high):
            errors.append(
                f"{field.capitalize()} should be between {low} and {high}."
            )
            continue

        features[FORM_TO_FEATURE[field]] = value

    return features, errors, raw_values
