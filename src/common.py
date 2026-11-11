from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO / "models" / "churn_xgb.json"
ONNX_PATH = REPO / "models" / "churn_xgb.onnx"
REFERENCE_PATH = REPO / "data" / "reference.csv"
CURRENT_PATH = REPO / "data" / "current.csv"
TARGET = "Churn Value"
RANDOM_STATE = 42
INPUT_NAME = "float_input"

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
