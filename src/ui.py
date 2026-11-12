import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st

from common import FeatureMetadata, get_feature_metadata, is_binary_column

API_URL = os.getenv("CHURNGUARD_API_URL", "http://localhost:8000")
INTERNET_ADDONS = [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]
PRESETS = {"High risk": 1107, "Loyal": 5716}

st.set_page_config(page_title="ChurnGuard", layout="centered")


@st.cache_resource
def load_metadata() -> FeatureMetadata:
    return get_feature_metadata()


def call_api(path: str, values: dict[str, Any]) -> dict[str, Any]:
    try:
        response = requests.post(f"{API_URL}{path}", json=values, timeout=10)
        response.raise_for_status()
        return response.json()
    except (requests.ConnectionError, requests.Timeout):
        st.error(
            f"Could not reach the API at {API_URL}. Start it with "
            "`uvicorn serve:app --app-dir src`."
        )
    except requests.HTTPError as exc:
        err_response = exc.response
        if err_response is None:
            st.error(f"API error: {exc}")
        else:
            st.error(f"API error {err_response.status_code}: {err_response.text}")
    return {}


def as_widget_value(col: str, value: Any, meta: FeatureMetadata) -> Any:
    if is_binary_column(meta.raw[col]):
        return "Yes" if int(value) == 1 else "No"
    if col in meta.categorical_columns:
        return str(value)
    return value


def binary_index(value: Any) -> int:
    return 1 if value == "Yes" else 0


def set_values(row: pd.Series, meta: FeatureMetadata) -> None:
    internet_no = str(row["Internet Service"]) == "No"
    for col in meta.raw.columns:
        value = as_widget_value(col, row[col], meta)
        if col in INTERNET_ADDONS and internet_no:
            value = "No internet service"
        st.session_state[col] = value


def random_row(meta: FeatureMetadata) -> pd.Series:
    no_missing = meta.raw[meta.raw.notna().all(axis=1)]
    index = int(no_missing.index.to_series().sample(1).iloc[0])
    return meta.raw.iloc[int(index)]


meta = load_metadata()

st.title("ChurnGuard")
st.caption("Live churn prediction from the API, explained with SHAP.")

if "values_initialized" not in st.session_state:
    set_values(meta.raw.iloc[0], meta)
    st.session_state.values_initialized = True

def apply_preset(preset: str) -> None:
    if preset == "Default":
        set_values(meta.raw.iloc[0], meta)
    elif preset == "Random customer":
        set_values(random_row(meta), meta)
    else:
        set_values(meta.raw.iloc[int(PRESETS[preset])], meta)


with st.sidebar:
    st.subheader("Customer presets")
    preset = st.selectbox("Preset", ["Default"] + list(PRESETS) + ["Random customer"])
    if preset != st.session_state.get("selected_preset", "Default"):
        apply_preset(preset)
        st.session_state.selected_preset = preset

values: dict[str, Any] = {}
columns = st.columns(3)
for i, col in enumerate(meta.raw.columns):
    with columns[i % 3]:
        if col in meta.categorical_columns:
            if col in INTERNET_ADDONS and st.session_state["Internet Service"] == "No":
                st.session_state[col] = "No internet service"
                st.selectbox(col, meta.categories[col], index=0, disabled=True)
                values[col] = "No internet service"
            else:
                values[col] = st.selectbox(
                    col,
                    meta.categories[col],
                    index=meta.categories[col].index(st.session_state[col]),
                    key=col,
                )
        elif is_binary_column(meta.raw[col]):
            answer = st.selectbox(
                col,
                ["No", "Yes"],
                index=binary_index(st.session_state[col]),
                key=col,
            )
            values[col] = 1 if answer == "Yes" else 0
        else:
            series = meta.raw[col]
            low = float(series.min())
            high = float(series.max())
            if pd.api.types.is_integer_dtype(series):
                values[col] = st.number_input(
                    col,
                    min_value=int(low),
                    max_value=int(high),
                    step=1,
                    value=int(st.session_state[col]),
                    key=col,
                )
            else:
                values[col] = st.number_input(
                    col,
                    min_value=low,
                    max_value=high,
                    value=float(st.session_state[col]),
                    format="%.2f",
                    key=col,
                )

submitted = st.button("Predict")

if submitted:
    prediction = call_api("/predict", values)
    if not prediction:
        st.stop()

    left, right = st.columns(2)
    left.metric("Churn probability", f"{prediction['churn_probability']:.1%}")
    right.metric("Prediction", "Churn" if prediction["churn"] else "No churn")

    explanation = call_api("/explain", values)
    if not explanation:
        st.stop()

    shap_explanation = shap.Explanation(
        values=np.array(explanation["shap_values"]),
        base_values=explanation["base_value"],
        data=np.array(explanation["data"]),
        feature_names=explanation["feature_names"],
    )
    st.subheader("SHAP explanation")
    fig = plt.figure(figsize=(11, 5))
    shap.plots.waterfall(shap_explanation, max_display=10, show=False)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Feature contributions in log-odds space: red pushes toward churn, blue pushes away.")
