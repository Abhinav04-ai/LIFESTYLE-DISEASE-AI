from pathlib import Path

import joblib
import streamlit as st

from src.config import (
    ASSETS_DIR,
    DISEASES,
    EVALUATION_FILES,
    MODEL_FILES,
    MODEL_NAMES,
)
from src.data_loader import DatasetError, load_dataset
from src.paper_reference import PAPER_TITLE
from src.train_models import train_disease

st.set_page_config(
    page_title=PAPER_TITLE,
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Checking saved models and datasets...")
def load_runtime():
    state = {"models": {}, "evaluations": {}, "errors": {}, "datasets": {}}
    for disease, spec in DISEASES.items():
        try:
            frame = load_dataset(disease)
            state["datasets"][disease] = frame
        except DatasetError as exc:
            state["errors"][disease] = str(exc)
            frame = None

        model_paths = MODEL_FILES[disease]
        evaluation_path = EVALUATION_FILES[disease]
        complete = all(path.is_file() for path in model_paths.values()) and evaluation_path.is_file()
        if not complete and frame is not None:
            try:
                with st.spinner(f"Training {spec['title']} models for the first time..."):
                    train_disease(disease)
                complete = all(path.is_file() for path in model_paths.values()) and evaluation_path.is_file()
            except Exception as exc:
                state["errors"][disease] = (
                    f"Automatic training failed for {spec['title']}: {exc}. "
                    "Check the dataset and installed dependencies, then run "
                    "`python -m src.train_models` for details."
                )
        if complete:
            try:
                state["models"][disease] = {
                    key: joblib.load(path) for key, path in model_paths.items()
                }
                state["evaluations"][disease] = joblib.load(evaluation_path)
                state["errors"].pop(disease, None)
            except Exception as exc:
                state["errors"][disease] = (
                    f"Could not load saved {spec['title']} models: {exc}. "
                    "Remove the incompatible model files and train again."
                )
    return state


def render_styles():
    css_path = ASSETS_DIR / "style.css"
    if css_path.is_file():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


def render_data_status(state):
    missing = [
        (disease, message)
        for disease, message in state["errors"].items()
        if disease not in state["models"]
    ]
    if missing:
        with st.expander("Dataset and model setup", expanded=True):
            st.warning(
                "Predictions and experiment metrics are only available after the "
                "original dataset files have been added and the models trained."
            )
            for disease, message in missing:
                st.markdown(f"**{DISEASES[disease]['title']}**")
                st.code(message, language="text")
                st.markdown(
                    f"[Dataset source]({DISEASES[disease]['source_url']})"
                )
            st.caption(
                "Place each downloaded file in this project's datasets folder, "
                "keeping the exact filenames shown above. No sample or synthetic "
                "medical records are used."
            )


def main():
    render_styles()
    state = load_runtime()

    st.sidebar.markdown("## 🩺 Lifestyle Disease XAI")
    st.sidebar.caption("Research prototype · Runs locally")
    pages = {
        "Home dashboard": "home",
        "Diabetes prediction": "diabetes",
        "Heart disease prediction": "heart",
        "Explainable AI": "xai",
        "Model performance": "performance",
        "About project": "about",
    }
    selected = st.sidebar.radio("Navigate", list(pages), key="navigation")
    st.sidebar.divider()
    st.sidebar.markdown(
        "**Educational use only.** Predictions and probabilities are not medical "
        "diagnoses. Consult a qualified healthcare professional."
    )

    render_data_status(state)
    module_name = pages[selected]
    page_module = __import__(f"ui_pages.{module_name}", fromlist=["render"])
    page_module.render(state)


if __name__ == "__main__":
    main()
