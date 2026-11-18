import plotly.express as px
import streamlit as st

from src.config import DISEASES, MODEL_NAMES
from src.evaluation import metrics_frame
from src.paper_reference import PAPER_CV_METRICS, PAPER_FIGURES


def _navigate(destination):
    st.session_state["navigation"] = destination
    st.rerun()


def render(state):
    st.markdown(
        '<div class="hero"><div class="eyebrow">RESEARCH PROJECT · EXPLAINABLE MACHINE LEARNING</div>'
        "<h1>Early Prediction of Lifestyle Diseases Using Explainable Machine Learning</h1>"
        "<p>A local application based on the submitted research paper and notebook. "
        "Explore the same two datasets, model families, evaluation workflow and "
        "Random Forest SHAP explanations.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "This educational prototype retrains the research pipelines locally. "
        "Published paper results and current local experiment results are shown "
        "separately where they differ. No medical records are sent to an external service."
    )

    cols = st.columns(4)
    cards = [
        ("Conditions", "2", "Diabetes and cardiovascular disease"),
        ("Algorithms", "3", "Logistic Regression, Random Forest, XGBoost"),
        ("Evaluation", "5-fold", "Stratified cross-validation"),
        ("Explanations", "SHAP", "Random Forest feature attributions"),
    ]
    for col, (title, value, caption) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="eyebrow">{title}</div>'
                f'<div class="card-value">{value}</div><div class="muted">{caption}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Explore the project")
    left, right = st.columns(2)
    with left:
        st.markdown(
            '<div class="panel"><h3>Diabetes research model</h3>'
            "<p>Eight clinical and demographic inputs evaluated using three "
            "classifiers and the Pima Indians Diabetes dataset.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button("Open diabetes prediction", width="stretch"):
            _navigate("Diabetes prediction")
    with right:
        st.markdown(
            '<div class="panel"><h3>Cardiovascular research model</h3>'
            "<p>Thirteen inputs from the Cleveland dataset, with disease severity "
            "grouped into a binary research target.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button("Open heart disease prediction", width="stretch"):
            _navigate("Heart disease prediction")

    st.markdown("### Model results")
    with st.expander("Results reported in the research paper"):
        st.markdown(
            "The paper selected Random Forest using five-fold mean F1: "
            f"**{PAPER_CV_METRICS['diabetes']['rf']['mean_f1']:.4f}** for diabetes "
            f"and **{PAPER_CV_METRICS['heart']['rf']['mean_f1']:.4f}** for "
            "cardiovascular disease. These are the paper's reported results, not "
            "substitutes for the local retraining results below."
        )
        st.image(
            str(PAPER_FIGURES["cross_validation"]),
            caption="Figure 4 from the uploaded paper: five-fold cross-validation mean F1-scores.",
            width="stretch",
        )
    available = [
        disease for disease in DISEASES if disease in state["evaluations"]
    ]
    if available:
        chosen = st.selectbox(
            "Dataset", available, format_func=lambda key: DISEASES[key]["title"]
        )
        table = metrics_frame(state["evaluations"][chosen], MODEL_NAMES)
        melted = table.melt(
            id_vars="Model", var_name="Metric", value_name="Score"
        )
        fig = px.bar(
            melted,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            range_y=[0, 1],
            color_discrete_sequence=["#163b65", "#2f7fb0", "#6db3d1"],
        )
        fig.update_layout(
            height=360,
            margin=dict(l=12, r=12, t=16, b=12),
            legend_title_text="Algorithm",
            yaxis_title="Held-out score",
            xaxis_title="",
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info(
            "Results will appear here after the dataset files are added and real "
            "models have been trained. No metrics are estimated or hard-coded."
        )

    st.markdown(
        '<div class="disclaimer"><strong>Research use only.</strong> This tool '
        "does not diagnose disease. A model probability is not a confirmed medical "
        "risk. Consult a qualified healthcare professional for health decisions.</div>",
        unsafe_allow_html=True,
    )
