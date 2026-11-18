import streamlit as st

from src.config import DISEASES
from src.paper_reference import PAPER_TITLE


def render(state):
    st.title("About the project")
    st.markdown(
        f"**Research paper:** *{PAPER_TITLE}*\n\n"
        "### Project objectives\n"
        "Convert the source research notebook into a local, reproducible "
        "application for comparing classical machine-learning models and "
        "inspecting their feature attributions."
    )
    st.markdown(
        "### Problem statement\n"
        "Lifestyle-related conditions may benefit from earlier awareness. This "
        "project explores binary classification on historical benchmark datasets; "
        "it is not a screening service or a clinical decision system."
    )

    st.markdown("### Datasets")
    st.markdown(
        f"- **Pima Indians Diabetes:** eight numeric features and a binary "
        f"outcome. [Dataset source]({DISEASES['diabetes']['source_url']}). "
        "The source file is expected as `pima-indians-diabetes.data.txt`."
    )
    st.markdown(
        f"- **Cleveland Heart Disease:** thirteen features and target values "
        f"0-4, grouped as 0 = no disease and 1-4 = disease. "
        f"[Original UCI file]({DISEASES['heart']['source_url']}). "
        "Save as `processed.cleveland.txt`."
    )
    for disease, message in state["errors"].items():
        if disease not in state["models"]:
            st.warning(message)

    st.markdown("### Algorithms and experiment method")
    st.markdown(
        "- Logistic Regression with median imputation, standard scaling, "
        "`max_iter=1000`, and `random_state=42`.\n"
        "- Random Forest with 150 trees, maximum depth 10, balanced class weights, "
        "and `random_state=42`.\n"
        "- XGBoost with 100 estimators, depth 3, learning rate 0.1, row/column "
        "subsampling 0.8, and log-loss evaluation.\n"
        "- The notebook's stratified 80/20 split (seed 42), test metrics, and "
        "stratified five-fold F1 cross-validation are retained. Diabetes zeros "
        "for glucose, blood pressure, skin thickness, insulin, and BMI are treated "
        "as missing and median-imputed."
    )
    st.markdown(
        "### Explainable AI\n"
        "TreeSHAP attributions are calculated from each trained Random Forest "
        "using held-out data or a submitted record. Global mean absolute SHAP "
        "values summarize model behavior; individual values indicate how inputs "
        "move that model's positive-class output relative to its baseline."
    )
    st.markdown(
        "### Technology stack\n"
        "Python, Pandas, NumPy, scikit-learn, XGBoost, SHAP, Joblib, Streamlit, "
        "Plotly, and Matplotlib. The application is designed to run on the local "
        "computer and does not call a prediction API."
    )
    st.markdown(
        "### Limitations and future scope\n"
        "These small historical datasets may be unrepresentative, have missing or "
        "outdated measurements, and contain population-specific bias. Probability "
        "outputs are not calibrated medical risks. Future work could include "
        "external validation, calibration, fairness analysis, contemporary "
        "representative data, and review by clinical professionals."
    )
    st.caption(
        "Figures 4-6 and the paper-reported tables in this app are taken from the "
        "research PDF you supplied. Current app metrics are recalculated from the "
        "local training run and are labeled separately."
    )
    st.error(
        "**Medical disclaimer:** This is an educational and research prototype. "
        "Predictions are based on historical datasets, are not confirmed diagnoses "
        "or medical advice, and must not guide treatment decisions. Consult a "
        "qualified healthcare professional."
    )
