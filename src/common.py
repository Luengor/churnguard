from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO / "models" / "churn_xgb.json"
ONNX_PATH = REPO / "models" / "churn_xgb.onnx"
REFERENCE_PATH = REPO / "data" / "reference.csv"
CURRENT_PATH = REPO / "data" / "current.csv"
DATA_PATH = REPO / "data" / "Telco_customer_churn.xlsx"
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


@dataclass(frozen=True)
class FeatureMetadata:
    raw: pd.DataFrame
    feature_columns: list[str]
    categorical_columns: list[str]
    categories: dict[str, list[str]]

    def to_vector(self, row: dict[str, Any]) -> np.ndarray:
        frame = pd.DataFrame([row])
        for col in self.categorical_columns:
            frame[col] = pd.Categorical(frame[col], categories=self.categories[col])
        encoded = pd.get_dummies(frame, drop_first=True)
        encoded = encoded.reindex(columns=self.feature_columns, fill_value=0.0)
        return encoded.astype("float32").to_numpy()


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


def is_binary_column(series: pd.Series) -> bool:
    return pd.api.types.is_integer_dtype(series) and set(series.unique().tolist()) <= {0, 1}


def field_types(
    raw: pd.DataFrame, categorical_columns: list[str], categories: dict[str, list[str]]
) -> dict[str, Any]:
    return {
        col: (
            Literal[tuple(categories[col])]
            if col in categorical_columns
            else int
            if pd.api.types.is_integer_dtype(raw[col])
            else float
        )
        for col in raw.columns
    }


def get_feature_metadata() -> FeatureMetadata:
    df = load_data()
    raw = df.drop(columns=DROP_COLS + [TARGET])
    X_full, _ = prepare_features(df)
    feature_columns = list(X_full.columns)
    categorical_columns = [
        c for c in raw.columns if pd.api.types.is_string_dtype(raw[c])
    ]
    categories = {
        c: sorted(raw[c].dropna().unique().tolist()) for c in categorical_columns
    }
    return FeatureMetadata(
        raw=raw,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        categories=categories,
    )
