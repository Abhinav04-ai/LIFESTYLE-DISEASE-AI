import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from src.config import DISEASES, MODEL_NAMES
from src.evaluation import metrics_frame
from src.paper_reference import PAPER_CV_METRICS, PAPER_FIGURES, PAPER_TEST_METRICS


def render(state):
    st.title("Model performance")
    st.write(
        "All numbers below are computed from the notebook's stratified 80/20 "
        "holdout and stratified five-fold F1 cross-validation. They are not "
        "clinical performance guarantees."
    )
    available = [
        disease for disease in DISEASES if disease in state["evaluations"]
    ]
    if not available:
        st.info("No evaluation results are available until original datasets are supplied.")
        return
    disease = st.selectbox(
        "Dataset", available, format_func=lambda key: DISEASES[key]["title"]
    )
    evaluation = state["evaluations"][disease]

    st.markdown("### Results reported in the uploaded paper")
    st.caption(
        "Reference values transcribed from Tables II and III. They are shown "
        "separately from local retraining because model results vary with the "
        "installed library versions and experiment environment."
    )
    published_rows = []
    for key, name in MODEL_NAMES.items():
        published_rows.append(
            {
                "Model": name,
                **PAPER_TEST_METRICS[disease][key],
                "CV mean F1": PAPER_CV_METRICS[disease][key]["mean_f1"],
                "CV std F1": PAPER_CV_METRICS[disease][key]["std_f1"],
            }
        )
    published = pd.DataFrame(published_rows)
    st.dataframe(
        published.style.format(
            {column: "{:.4f}" for column in published.columns if column != "Model"}
        ),
        hide_index=True,
        width="stretch",
    )
    with st.expander("Five-fold cross-validation figure from the paper"):
        st.image(
            str(PAPER_FIGURES["cross_validation"]),
            caption="Figure 4 from the uploaded paper.",
            width="stretch",
        )

    st.markdown("### Locally retrained held-out test metrics")
    st.caption(
        "These metrics are computed from the six models saved on this computer; "
        "they are not copied from the paper."
    )
    metrics = metrics_frame(evaluation, MODEL_NAMES)
    st.dataframe(
        metrics.style.format(
            {name: "{:.3f}" for name in metrics.columns if name != "Model"}
        ),
        hide_index=True,
        width="stretch",
    )
    melted = metrics.melt(
        id_vars="Model", var_name="Metric", value_name="Score"
    )
    comparison = px.bar(
        melted,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        range_y=[0, 1],
        color_discrete_sequence=["#163b65", "#2f7fb0", "#6db3d1"],
    )
    comparison.update_layout(
        height=360,
        margin=dict(l=12, r=12, t=12, b=12),
        yaxis_title="Score",
    )
    st.plotly_chart(comparison, width="stretch")

    model_key = st.selectbox(
        "Detailed evaluation",
        list(MODEL_NAMES),
        format_func=lambda key: MODEL_NAMES[key],
    )
    details = evaluation["models"][model_key]
    left, right = st.columns(2)
    with left:
        st.markdown("#### Confusion matrix")
        matrix = details["confusion_matrix"]
        labels = DISEASES[disease]["classes"]
        figure = ff.create_annotated_heatmap(
            z=matrix.tolist(),
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=True,
            annotation_text=[[str(value) for value in row] for row in matrix],
        )
        figure.update_layout(
            xaxis_title="Predicted label",
            yaxis_title="Actual label",
            height=360,
            margin=dict(l=12, r=12, t=12, b=12),
        )
        st.plotly_chart(figure, width="stretch")
    with right:
        st.markdown("#### Classification report")
        report_data = details["classification_report"]
        report_rows = {
            label: values
            if isinstance(values, dict)
            else {"accuracy": values}
            for label, values in report_data.items()
        }
        report = pd.DataFrame.from_dict(report_rows, orient="index")
        report.index.name = "Class / average"
        report = report.reset_index()
        st.dataframe(
            report.style.format(
                {
                    column: "{:.3f}"
                    for column in report.select_dtypes(include="number").columns
                }
            ),
            hide_index=True,
            width="stretch",
        )

    st.markdown("### Locally retrained five-fold stratified cross-validation")
    cv_result = evaluation["cv"][model_key]
    cv_frame = pd.DataFrame(
        {
            "Fold": [f"Fold {index}" for index in range(1, 6)],
            "F1-score": cv_result["fold_scores"],
        }
    )
    cv_fig = px.line(
        cv_frame,
        x="Fold",
        y="F1-score",
        markers=True,
        range_y=[0, 1],
        color_discrete_sequence=["#2f7fb0"],
    )
    cv_fig.update_layout(height=280, margin=dict(l=12, r=12, t=12, b=12))
    st.plotly_chart(cv_fig, width="stretch")
    st.metric(
        "Mean cross-validation F1",
        f"{cv_result['mean_f1']:.3f}",
        delta=f"SD {cv_result['std_f1']:.3f}",
    )
    st.caption(
        f"Evaluation used {evaluation['dataset_rows']} rows and a 42-seeded "
        "stratified train/test split. Cross-validation is run on the full "
        "preprocessed feature set, matching the notebook."
    )
