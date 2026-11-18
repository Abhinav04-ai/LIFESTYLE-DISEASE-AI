import numpy as np
import pandas as pd

from .config import DISEASES, MODEL_NAMES


def validate_input(disease: str, values: dict) -> pd.DataFrame:
    if disease not in DISEASES:
        raise ValueError(f"Unknown disease key: {disease}")
    features = DISEASES[disease]["features"]
    missing = [feature for feature in features if feature not in values]
    if missing:
        raise ValueError(f"Missing input values: {', '.join(missing)}.")
    try:
        row = pd.DataFrame([[float(values[name]) for name in features]], columns=features)
    except (TypeError, ValueError) as exc:
        raise ValueError("Every feature must have a valid numeric value.") from exc
    if not np.isfinite(row.to_numpy(dtype=float)).all():
        raise ValueError("Feature values must be finite numbers.")
    if disease == "diabetes":
        zero_missing = DISEASES[disease]["zero_is_missing"]
        if (row[zero_missing] < 0).any(axis=None):
            raise ValueError("Glucose, blood pressure, skin thickness, insulin, and BMI cannot be negative.")
        if not 0 <= row.loc[0, "Pregnancies"] <= 30 or not 1 <= row.loc[0, "Age"] <= 120:
            raise ValueError("Pregnancies must be 0-30 and age must be 1-120.")
        if not 0 <= row.loc[0, "DiabetesPedigreeFunction"] <= 10:
            raise ValueError("Diabetes pedigree function must be between 0 and 10.")
    else:
        allowed = {
            "sex": {0, 1},
            "cp": {1, 2, 3, 4},
            "fbs": {0, 1},
            "restecg": {0, 1, 2},
            "exang": {0, 1},
            "slope": {1, 2, 3},
            "ca": {0, 1, 2, 3},
            "thal": {3, 6, 7},
        }
        for name, options in allowed.items():
            if row.loc[0, name] not in options:
                raise ValueError(f"{name} must be one of {sorted(options)}.")
        if not 1 <= row.loc[0, "age"] <= 120:
            raise ValueError("Age must be between 1 and 120.")
        if row.loc[0, "trestbps"] <= 0 or row.loc[0, "chol"] <= 0 or row.loc[0, "thalach"] <= 0:
            raise ValueError("Blood pressure, cholesterol, and maximum heart rate must be positive.")
    return row


def predict(model, disease: str, values: dict, model_key: str) -> dict:
    if model_key not in MODEL_NAMES:
        raise ValueError(f"Unknown model key: {model_key}")
    row = validate_input(disease, values)
    if model is None:
        raise RuntimeError("The selected model is not available. Train it after adding the dataset.")
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])
    return {
        "class": prediction,
        "label": DISEASES[disease]["classes"][prediction],
        "probability": probability,
        "model_name": MODEL_NAMES[model_key],
        "inputs": row.iloc[0].to_dict(),
    }
