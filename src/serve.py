import os
import time
from typing import Any

import numpy as np
import onnxruntime as ort
import xgboost as xgb
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, create_model

from common import INPUT_NAME, MODEL_PATH, ONNX_PATH, field_types, get_feature_metadata

meta = get_feature_metadata()
raw = meta.raw
CATEGORIES = meta.categories

fields: dict[str, Any] = {
    col.replace(" ", "_").lower(): (typ, Field(alias=col))
    for col, typ in field_types(raw, meta.categorical_columns, CATEGORIES).items()
}

CustomerInput: type[BaseModel] = create_model(
    "CustomerInput",
    __config__=ConfigDict(validate_by_name=True, validate_by_alias=True),
    **fields,
)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn: bool


class ExplainResponse(BaseModel):
    shap_values: list[float]
    base_value: float
    data: list[float]
    feature_names: list[str]


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

booster = xgb.Booster()
booster.load_model(str(os.getenv("CHURNGUARD_XGB_PATH", MODEL_PATH)))

app = FastAPI(title="ChurnGuard")


def to_feature_vector(customer: BaseModel) -> np.ndarray:
    return meta.to_vector(customer.model_dump(by_alias=True))


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


@app.post("/explain", response_model=ExplainResponse)
def explain(customer: CustomerInput):  # type: ignore[valid-type]  # FastAPI needs the runtime model for the body schema
    features = to_feature_vector(customer)
    # shap.TreeExplainer (0.54.1) doesn't reconstruct this model's margin; XGBoost's own contribs do.
    contribs = booster.predict(
        xgb.DMatrix(features, feature_names=booster.feature_names),
        pred_contribs=True,
    )
    return ExplainResponse(
        shap_values=[float(v) for v in contribs[0, :-1]],
        base_value=float(contribs[0, -1]),
        data=[float(v) for v in features[0]],
        feature_names=list(meta.feature_columns),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
