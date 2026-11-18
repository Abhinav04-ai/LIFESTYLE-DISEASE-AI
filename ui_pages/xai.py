import hashlib

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import DISEASES, MODEL_FILES
from src.explainability import explain_random_forest
from src.paper_reference import PAPER_FIGURES, PAPER_SHAP_VALUES
from src.prediction import validate_input


@st.cache_resource(show_spinner="Calculating SHAP explanations from the trained model...")
def _cached_explanations(disease, model_stamp, data_digest, _pipeline, _x_values):
    del model_stamp, data_digest
    return explain_random_forest(_pipeline, _x_values)


def _explanation_for(state, disease):
    evaluation = state["evaluations"][disease]
    x_test = evaluation["x_test"]
    if len(x_test) > 256:
        x_values = x_test.sample(n=256, random_state=42).sort_index()
    else:
        x_values = x_test
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(x_values, index=True).values.tobytes()
    ).hexdigest()
    model_path = MODEL_FILES[disease]["rf"]
    explanation, importance = _cached_explanations(
        disease, model_path.stat().st_mtime_ns, digest,
        state["models"][disease]["rf"], x_values
    )
    return explanation, importance, x_values


def render(state):
    st.title("Explainable AI")
    st.write(
        "SHAP (SHapley Additive exPlanations) estimates how each input changes "
        "the Random Forest model's output relative to its baseline. These are "
        "model attributions, not evidence of cause or clinical importance."
    )
    available = [
        disease
        for disease in DISEASES
        if disease in state["models"] and disease in state["evaluations"]
    ]
    if not available:
        st.info("SHAP explanations will be available after training with the original datasets.")
        return

    disease = st.selectbox(
        "Dataset", available, format_func=lambda key: DISEASES[key]["title"]
    )
    pipeline = state["models"][disease]["rf"]
    evaluation = state["evaluations"][disease]
    try:
        explanation, importance, explained_rows = _explanation_for(state, disease)
    except Exception as exc:
        st.error(
            "SHAP could not explain this saved Random Forest model. Check SHAP and "
            f"model compatibility, then retrain if needed. Details: {exc}"
        )
        return

    st.markdown("### Global feature importance")
    st.caption(
        "Mean absolute SHAP value across held-out test records. Larger values mean "
        "a feature moved model outputs more in this experiment; they do not imply causation."
    )
    global_fig = px.bar(
        importance.sort_values("Mean |SHAP|"),
        x="Mean |SHAP|",
        y="Feature",
        orientation="h",
        color_discrete_sequence=["#2f7fb0"],
    )
    global_fig.update_layout(
        height=max(350, 28 * len(importance)),
        margin=dict(l=12, r=12, t=12, b=12),
        yaxis_title="",
    )
    st.plotly_chart(global_fig, width="stretch")

    try:
        import shap

        figure = plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(explanation, max_display=min(12, len(importance)), show=False)
        st.pyplot(figure, clear_figure=True, width="stretch")
        plt.close(figure)
    except Exception as exc:
        st.warning(f"The SHAP summary plot could not be rendered: {exc}")

    st.markdown("### Individual prediction explanation")
    x_test = evaluation["x_test"]
    test_choices = [
        f"Held-out test record {index}" for index in explained_rows.index
    ]
    has_latest = f"latest_input_{disease}" in st.session_state
    sources = test_choices + (["My latest prediction input"] if has_latest else [])
    chosen = st.selectbox("Record", sources)
    if chosen == "My latest prediction input":
        try:
            row = validate_input(
                disease, st.session_state[f"latest_input_{disease}"]
            )
        except ValueError as exc:
            st.error(f"Could not use the latest input: {exc}")
            return
        try:
            one_explanation, _ = explain_random_forest(pipeline, row)
        except Exception as exc:
            st.error(f"Could not explain this input with SHAP: {exc}")
            return
        attribution = one_explanation.values[0]
        display_values = row.iloc[0]
    else:
        row_index = int(chosen.rsplit(" ", 1)[-1])
        position = explained_rows.index.get_loc(row_index)
        attribution = explanation.values[position]
        display_values = explained_rows.loc[row_index]

    impacts = pd.DataFrame(
        {
            "Feature": list(x_test.columns),
            "SHAP value": attribution,
            "Input value": display_values.to_numpy(),
        }
    )
    impacts["Direction"] = impacts["SHAP value"].map(
        lambda value: "Pushes model output higher"
        if value > 0
        else "Pushes model output lower"
        if value < 0
        else "No change"
    )
    impacts = impacts.reindex(
        impacts["SHAP value"].abs().sort_values().index
    )
    impact_fig = px.bar(
        impacts,
        x="SHAP value",
        y="Feature",
        orientation="h",
        color="SHAP value",
        color_continuous_scale=["#2f7fb0", "#e7eef5", "#c65c58"],
        hover_data=["Input value", "Direction"],
    )
    impact_fig.update_layout(
        height=max(350, 28 * len(impacts)),
        margin=dict(l=12, r=12, t=12, b=12),
        yaxis_title="",
        coloraxis_showscale=False,
    )
    st.plotly_chart(impact_fig, width="stretch")
    st.dataframe(impacts.iloc[::-1], hide_index=True, width="stretch")
    st.info(
        "Positive and negative describe movement in the Random Forest's output "
        "for the positive class. They are not positive/negative health effects."
    )
    with st.expander("Figures and SHAP values from the uploaded paper"):
        st.caption(
            "These are the figures and mean absolute SHAP values reported in "
            "your paper. The live explanations above are recalculated from the "
            "models trained on this computer and may differ."
        )
        paper_columns = st.columns(2)
        paper_key = "diabetes_shap" if disease == "diabetes" else "heart_shap"
        with paper_columns[0]:
            st.image(
                str(PAPER_FIGURES[paper_key]),
                caption=(
                    "Figure 5 from the paper: diabetes SHAP importance."
                    if disease == "diabetes"
                    else "Figure 6 from the paper: cardiovascular SHAP importance."
                ),
                width="stretch",
            )
        with paper_columns[1]:
            st.dataframe(
                pd.DataFrame(
                    PAPER_SHAP_VALUES[disease],
                    columns=["Feature", "Paper mean |SHAP|"],
                ),
                hide_index=True,
                width="stretch",
            )
