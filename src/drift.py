import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from common import CURRENT_PATH, RANDOM_STATE, REFERENCE_PATH, REPO, TARGET

REPORT_DIR = REPO / "monitoring" / "evidently"


def fabricate_drifted(current: pd.DataFrame) -> pd.DataFrame:
    drifted = current.copy()
    rng = np.random.default_rng(RANDOM_STATE)
    drifted["Tenure Months"] = np.floor(
        drifted["Tenure Months"] * rng.uniform(0.4, 0.6, size=len(drifted))
    ).astype(int)
    drifted["Monthly Charges"] = drifted["Monthly Charges"] + rng.normal(
        25.0, 8.0, size=len(drifted)
    )
    drifted.loc[drifted["Contract"] != "Month-to-month", "Contract"] = "Month-to-month"
    return drifted


def run_report(reference: pd.DataFrame, current: pd.DataFrame, name: str) -> None:
    snapshot = Report([DataDriftPreset()]).run(current, reference)
    output = REPORT_DIR / f"{name}.html"
    snapshot.save_html(str(output))
    drift = next(
        m
        for m in snapshot.dict()["metrics"]
        if m["metric_name"].startswith("DriftedColumnsCount")
    )
    print(
        f"{name}: drifted_columns={drift['value']['count']}, "
        f"share={drift['value']['share']:.2%}, saved={output}"
    )


def main() -> None:
    reference = pd.read_csv(REFERENCE_PATH).drop(columns=[TARGET])
    current = pd.read_csv(CURRENT_PATH).drop(columns=[TARGET])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    run_report(reference, current, "baseline")
    run_report(reference, fabricate_drifted(current), "drifted")


if __name__ == "__main__":
    main()
