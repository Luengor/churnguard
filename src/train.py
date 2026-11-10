from pathlib import Path

import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import xgboost as xgb
from mlflow.xgboost import log_model as log_xgboost_model
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
DATA_PATH = REPO / "data" / "Telco_customer_churn.xlsx"
MODEL_PATH = REPO / "models" / "churn_xgb.json"
ONNX_PATH = REPO / "models" / "churn_xgb.onnx"

DROP_COLS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Label",
    "Churn Score",
    "CLTV",
    "Churn Reason",
]

TARGET = "Churn Value"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MLFLOW_TRACKING_URI = f"sqlite:///{REPO / 'mlflow.db'}"


def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH)
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df["Senior Citizen"] = df["Senior Citizen"].map({"Yes": 1, "No": 0})
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET]
    X = df.drop(columns=DROP_COLS + [TARGET])
    X = pd.get_dummies(X, drop_first=True).astype("float32")
    return X, y


def export_onnx(model: xgb.Booster, X: pd.DataFrame) -> None:
    feature_names = [f"f{i}" for i in range(X.shape[1])]

    # onnxmltools expects old-style f-indexed feature names in the tree dump.
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    booster.feature_names = feature_names

    onnx_model = convert_xgboost(
        booster,
        initial_types=[("float_input", FloatTensorType([None, X.shape[1]]))],
    )
    onnx.save_model(onnx_model, str(ONNX_PATH))

    sample = X.iloc[:5].to_numpy(dtype=np.float32)
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    onnx_proba = session.run(None, {"float_input": sample})[1][:, 1]
    native_proba = model.predict(xgb.DMatrix(sample, feature_names=model.feature_names))
    max_diff = float(np.max(np.abs(onnx_proba - native_proba)))
    print(f"ONNX max probability diff vs native: {max_diff:.2e}")
    print(f"ONNX saved: {ONNX_PATH}")


def main() -> None:
    df = load_data()
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_test = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 4,
        "eta": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "seed": RANDOM_STATE,
    }

    model = xgb.train(
        params,
        d_train,
        num_boost_round=300,
        evals=[(d_test, "test")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    y_pred_proba = model.predict(d_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, y_pred_proba))
    f1 = float(f1_score(y_test, y_pred))

    print(f"ROC-AUC: {auc:.4f}")
    print(f"F1:      {f1:.4f}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("churnguard")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    export_onnx(model, X)

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics({"roc_auc": auc, "f1": f1})
        log_xgboost_model(model, "xgboost")
        mlflow.log_artifact(str(ONNX_PATH))
        print(f"MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    main()
