from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "datasets"
MODEL_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"

DISEASES = {
    "diabetes": {
        "title": "Diabetes",
        "target": "Outcome",
        "dataset": "pima-indians-diabetes.data.txt",
        "classes": ["No Diabetes", "Diabetes"],
        "source_url": (
            "https://www.kaggle.com/datasets/uciml/"
            "pima-indians-diabetes-database"
        ),
        "features": [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ],
        "zero_is_missing": [
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
        ],
    },
    "heart": {
        "title": "Cardiovascular disease",
        "target": "num",
        "dataset": "processed.cleveland.txt",
        "classes": ["No Heart Disease", "Heart Disease"],
        "source_url": (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "heart-disease/processed.cleveland.data"
        ),
        "features": [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ],
        "zero_is_missing": [],
    },
}

MODEL_NAMES = {
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
}

MODEL_FILES = {
    disease: {
        key: MODEL_DIR / f"{disease}_{key}.pkl"
        for key in MODEL_NAMES
    }
    for disease in DISEASES
}

EVALUATION_FILES = {
    disease: MODEL_DIR / f"{disease}_evaluation.joblib"
    for disease in DISEASES
}
