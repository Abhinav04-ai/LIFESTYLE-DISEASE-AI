import numpy as np
import pandas as pd
import shap


def _positive_class_values(values, n_rows: int, n_features: int):
    if isinstance(values, list):
        if len(values) < 2:
            values = values[0]
        else:
            values = values[1]
    array = np.asarray(values)
    if array.ndim == 3:
        if array.shape == (n_rows, n_features, 2):
            array = array[:, :, 1]
        elif array.shape == (2, n_rows, n_features):
            array = array[1]
        else:
            raise ValueError(f"Unsupported SHAP output shape: {array.shape}.")
    if array.shape != (n_rows, n_features):
        raise ValueError(f"Unsupported SHAP output shape: {array.shape}.")
    return array


def explain_random_forest(pipeline, x_values: pd.DataFrame):
    """Return actual positive-class SHAP values for the trained RF pipeline."""
    if "imputer" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        raise ValueError("The saved Random Forest pipeline is missing required steps.")
    imputed = pipeline.named_steps["imputer"].transform(x_values)
    model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(imputed)
    values = _positive_class_values(raw_values, len(x_values), x_values.shape[1])

    base_values = np.asarray(explainer.expected_value)
    if base_values.ndim == 0:
        base = np.full(len(x_values), float(base_values))
    else:
        base = base_values.reshape(-1)
        base = np.full(len(x_values), float(base[1] if len(base) > 1 else base[0]))

    explanation = shap.Explanation(
        values=values,
        base_values=base,
        data=imputed,
        feature_names=list(x_values.columns),
    )
    importance = pd.DataFrame(
        {
            "Feature": list(x_values.columns),
            "Mean |SHAP|": np.abs(values).mean(axis=0),
        }
    ).sort_values("Mean |SHAP|", ascending=False, ignore_index=True)
    return explanation, importance


def explain_one(pipeline, values: pd.DataFrame):
    return explain_random_forest(pipeline, values)
