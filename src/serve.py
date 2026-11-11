import os
import time
from typing import Any, Literal

import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, create_model

from common import DROP_COLS, INPUT_NAME, ONNX_PATH, TARGET
from train import load_data, prepare_features

df = load_data()
raw = df.drop(columns=DROP_COLS + [TARGET])
X_full, _ = prepare_features(df)
FEATURE_COLUMNS = list(X_full.columns)
CATEGORICAL_COLUMNS = [c for c in raw.columns if pd.api.types.is_string_dtype(raw[c])]
CATEGORIES = {c: sorted(raw[c].dropna().unique().tolist()) for c in CATEGORICAL_COLUMNS}


def _field_name(col: str) -> str:
    return col.replace(" ", "_").lower()


def _field_type(col: str) -> Any:
    if col in CATEGORIES:
        return Literal[tuple(CATEGORIES[col])]
    if pd.api.types.is_integer_dtype(raw[col]):
        return int
    return float


fields: dict[str, Any] = {
    _field_name(c): (_field_type(c), Field(alias=c)) for c in raw.columns
}

CustomerInput: type[BaseModel] = create_model(
    "CustomerInput",
    __config__=ConfigDict(validate_by_name=True, validate_by_alias=True),
    **fields,
)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn: bool


REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
)

session = ort.InferenceSession(
    str(os.getenv("CHURNGUARD_ONNX_PATH", ONNX_PATH)),
    providers=["CPUExecutionProvider"],
)
app = FastAPI(title="ChurnGuard")


def to_feature_vector(customer: BaseModel) -> np.ndarray:
    frame = pd.DataFrame([customer.model_dump(by_alias=True)])
    for col in CATEGORICAL_COLUMNS:
        frame[col] = pd.Categorical(frame[col], categories=CATEGORIES[col])
    encoded = pd.get_dummies(frame, drop_first=True)
    encoded = encoded.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)
    return encoded.astype("float32").to_numpy()


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    endpoint = request.url.path
    REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, endpoint).observe(elapsed)
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):  # type: ignore[valid-type]  # FastAPI needs the runtime model for the body schema
    features = to_feature_vector(customer)
    probabilities, label = session.run(
        ["probabilities", "label"], {INPUT_NAME: features}
    )
    return PredictionResponse(
        churn_probability=float(probabilities[0, 1]),
        churn=bool(int(label[0])),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
