"""Published values and figure paths from the user's supplied research paper."""

from .config import ASSETS_DIR

PAPER_TITLE = (
    "Early Prediction of Lifestyle Diseases Using Explainable Machine Learning"
)

PAPER_TEST_METRICS = {
    "diabetes": {
        "lr": {
            "Accuracy": 0.7078,
            "Precision": 0.6000,
            "Recall": 0.5000,
            "F1-score": 0.5455,
            "ROC-AUC": 0.8130,
        },
        "rf": {
            "Accuracy": 0.7468,
            "Precision": 0.6316,
            "Recall": 0.6667,
            "F1-score": 0.6486,
            "ROC-AUC": 0.8222,
        },
        "xgb": {
            "Accuracy": 0.7727,
            "Precision": 0.6939,
            "Recall": 0.6296,
            "F1-score": 0.6602,
            "ROC-AUC": 0.8289,
        },
    },
    "heart": {
        "lr": {
            "Accuracy": 0.8689,
            "Precision": 0.8125,
            "Recall": 0.9286,
            "F1-score": 0.8667,
            "ROC-AUC": 0.9513,
        },
        "rf": {
            "Accuracy": 0.9016,
            "Precision": 0.8667,
            "Recall": 0.9286,
            "F1-score": 0.8966,
            "ROC-AUC": 0.9545,
        },
        "xgb": {
            "Accuracy": 0.9016,
            "Precision": 0.8438,
            "Recall": 0.9643,
            "F1-score": 0.9000,
            "ROC-AUC": 0.9578,
        },
    },
}

PAPER_CV_METRICS = {
    "diabetes": {
        "lr": {"mean_f1": 0.6358, "std_f1": 0.0210},
        "rf": {"mean_f1": 0.6705, "std_f1": 0.0467},
        "xgb": {"mean_f1": 0.6545, "std_f1": 0.0265},
    },
    "heart": {
        "lr": {"mean_f1": 0.8126, "std_f1": 0.0548},
        "rf": {"mean_f1": 0.8185, "std_f1": 0.0461},
        "xgb": {"mean_f1": 0.8088, "std_f1": 0.0365},
    },
}

PAPER_SHAP_VALUES = {
    "diabetes": [
        ("Glucose", 0.145231),
        ("BMI", 0.080461),
        ("Age", 0.066377),
        ("DiabetesPedigreeFunction", 0.042996),
        ("Insulin", 0.039886),
    ],
    "heart": [
        ("thal", 0.106963),
        ("ca", 0.090695),
        ("cp", 0.083848),
        ("oldpeak", 0.057976),
        ("thalach", 0.044919),
    ],
}

PAPER_FIGURES = {
    "cross_validation": ASSETS_DIR / "paper" / "figure_4_cross_validation.png",
    "diabetes_shap": ASSETS_DIR / "paper" / "figure_5_diabetes_shap.png",
    "heart_shap": ASSETS_DIR / "paper" / "figure_6_heart_shap.png",
}
