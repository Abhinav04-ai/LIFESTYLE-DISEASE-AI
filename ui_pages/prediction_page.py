import plotly.graph_objects as go
import streamlit as st

from src.config import DISEASES
from src.prediction import predict

from .forms import render_feature_form, render_model_selector


def render_prediction_page(state, disease: str):
    spec = DISEASES[disease]
    st.title(f"{spec['title']} prediction")
    st.write(
        "Enter the requested values to explore how a trained research model "
        "classifies this record. The result is not a clinical diagnosis."
    )

    model_key = render_model_selector(state, disease)
    with st.form(f"{disease}_prediction_form"):
        values = render_feature_form(disease, f"{disease}_form")
        submitted = st.form_submit_button(
            "Run research prediction",
            type="primary",
            width="stretch",
            disabled=model_key is None,
        )

    if submitted and model_key:
        st.session_state[f"latest_input_{disease}"] = values
        st.session_state[f"latest_model_{disease}"] = model_key
        try:
            result = predict(
                state["models"][disease][model_key], disease, values, model_key
            )
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
        else:
            st.session_state[f"latest_result_{disease}"] = result

    result = st.session_state.get(f"latest_result_{disease}")
    if result:
        st.markdown("### Model output")
        left, right = st.columns([1, 1.5])
        with left:
            st.markdown(
                f'<div class="result-card"><div class="eyebrow">'
                f'{result["model_name"]}</div><h2>{result["label"]}</h2>'
                f'<p>Predicted class: {result["class"]}</p></div>',
                unsafe_allow_html=True,
            )
            st.metric(
                "Model probability for the positive class",
                f"{result['probability']:.1%}",
            )
            st.caption(
                "This is the selected model's output probability, not a confirmed "
                "personal medical risk."
            )
        with right:
            fig = go.Figure(
                go.Bar(
                    x=[result["probability"]],
                    y=["Positive class"],
                    orientation="h",
                    marker_color="#2f7fb0",
                    text=[f"{result['probability']:.1%}"],
                    textposition="inside",
                )
            )
            fig.update_layout(
                xaxis=dict(range=[0, 1], tickformat=".0%", title="Model output"),
                yaxis_title="",
                height=170,
                margin=dict(l=10, r=10, t=8, b=8),
            )
            st.plotly_chart(fig, width="stretch")
        st.markdown("#### Values used")
        st.dataframe(
            [{"Feature": feature, "Input value": value} for feature, value in result["inputs"].items()],
            hide_index=True,
            width="stretch",
        )
        if disease == "diabetes":
            zero_features = [name for name in spec["zero_is_missing"] if values[name] == 0]
            if zero_features:
                st.warning(
                    "The notebook treats zero values for "
                    f"{', '.join(zero_features)} as missing and imputes them with "
                    "the training-set median."
                )
        st.warning(
            "Educational research output only — do not use this result to start, "
            "stop, or change treatment. Consult a qualified healthcare professional."
        )
