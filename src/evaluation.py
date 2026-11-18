import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, x_test, y_test, class_names):
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "metrics": {
            "Accuracy": float(accuracy_score(y_test, predictions)),
            "Precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "Recall": float(recall_score(y_test, predictions, zero_division=0)),
            "F1-score": float(f1_score(y_test, predictions, zero_division=0)),
            "ROC-AUC": float(roc_auc_score(y_test, probabilities)),
        },
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "predictions": np.asarray(predictions, dtype=int),
        "probabilities": np.asarray(probabilities, dtype=float),
    }


def metrics_frame(evaluation: dict, model_names: dict) -> pd.DataFrame:
    rows = []
    for key, name in model_names.items():
        result = evaluation["models"][key]["metrics"]
        rows.append({"Model": name, **result})
    return pd.DataFrame(rows)
