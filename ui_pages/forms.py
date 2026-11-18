import streamlit as st

from src.config import DISEASES, MODEL_NAMES

DIABETES_DEFAULTS = {
    "Pregnancies": (6, 0, 30, 1),
    "Glucose": (120, 0, 300, 1),
    "BloodPressure": (70, 0, 200, 1),
    "SkinThickness": (20, 0, 100, 1),
    "Insulin": (79, 0, 1000, 1),
    "BMI": (32.0, 0.0, 100.0, 0.1),
    "DiabetesPedigreeFunction": (0.5, 0.0, 10.0, 0.01),
    "Age": (33, 1, 120, 1),
}

HEART_DEFAULTS = {
    "age": (54, 1, 120, 1),
    "trestbps": (130, 50, 300, 1),
    "chol": (240, 50, 800, 1),
    "thalach": (150, 40, 250, 1),
    "oldpeak": (1.0, -5.0, 15.0, 0.1),
}

CATEGORICAL = {
    "sex": ([0, 1], {0: "0 · Female", 1: "1 · Male"}),
    "cp": ([1, 2, 3, 4], {1: "1 · Typical angina", 2: "2 · Atypical angina", 3: "3 · Non-anginal pain", 4: "4 · Asymptomatic"}),
    "fbs": ([0, 1], {0: "0 · No", 1: "1 · Yes"}),
    "restecg": ([0, 1, 2], {0: "0 · Normal", 1: "1 · ST-T abnormality", 2: "2 · Left ventricular hypertrophy"}),
    "exang": ([0, 1], {0: "0 · No", 1: "1 · Yes"}),
    "slope": ([1, 2, 3], {1: "1 · Upsloping", 2: "2 · Flat", 3: "3 · Downsloping"}),
    "ca": ([0, 1, 2, 3], {value: str(value) for value in range(4)}),
    "thal": ([3, 6, 7], {3: "3 · Normal", 6: "6 · Fixed defect", 7: "7 · Reversible defect"}),
}


def render_feature_form(disease: str, key_prefix: str):
    features = DISEASES[disease]["features"]
    defaults = DIABETES_DEFAULTS if disease == "diabetes" else HEART_DEFAULTS
    values = {}
    columns = st.columns(2)
    for index, feature in enumerate(features):
        with columns[index % 2]:
            if feature in CATEGORICAL:
                options, labels = CATEGORICAL[feature]
                values[feature] = st.selectbox(
                    feature,
                    options,
                    format_func=lambda value, labels=labels: labels[value],
                    key=f"{key_prefix}_{feature}",
                    help="Dataset category code; the model receives the numeric code.",
                )
            else:
                default, minimum, maximum, step = defaults[feature]
                values[feature] = st.number_input(
                    feature,
                    min_value=float(minimum),
                    max_value=float(maximum),
                    value=float(default),
                    step=float(step),
                    key=f"{key_prefix}_{feature}",
                )
    return values


def render_model_selector(state, disease: str):
    available = state["models"].get(disease, {})
    options = [key for key in MODEL_NAMES if key in available]
    if not options:
        st.info("No trained model is available for this dataset yet.")
        return None
    return st.selectbox(
        "Prediction model",
        options,
        format_func=lambda key: MODEL_NAMES[key],
        index=options.index("rf") if "rf" in options else 0,
        key=f"{disease}_model_selection",
        help="Random Forest is the model selected for analysis in the submitted paper. Other trained models remain available.",
    )
