from __future__ import annotations

import pandas as pd

from analysis_utils import censor_at, ensure_dirs, fit_cox_model
from config import CANCER_ORDER, CONTINUOUS_COVARIATES, COVARIATE_CATEGORIES, SYNTHETIC_DATA, TABLE_DIR


def run_model(df: pd.DataFrame, label: str, days: float) -> pd.DataFrame:
    censored = censor_at(df, "time_to_os_days", "os_event", days)
    categorical = list(COVARIATE_CATEGORIES)
    summary, model = fit_cox_model(censored, "time_to_os_days", "os_event", categorical, CONTINUOUS_COVARIATES)
    summary.insert(0, "events", int(model["os_event"].sum()))
    summary.insert(0, "n", len(model))
    summary.insert(0, "population", label)
    summary.insert(0, "follow_up", f"{int(days / 365.25)} years")
    return summary


def main() -> None:
    ensure_dirs(TABLE_DIR)
    df = pd.read_csv(SYNTHETIC_DATA)
    frames: list[pd.DataFrame] = []
    for days in [5 * 365.25, 10 * 365.25]:
        frames.append(run_model(df, "Overall", days))
        for cancer in CANCER_ORDER:
            subset = df[df["cancer_type"] == cancer].copy()
            if len(subset) >= 200:
                frames.append(run_model(subset, cancer, days))
    out = pd.concat(frames, ignore_index=True)
    out_path = TABLE_DIR / "main_multivariable_cox.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved main Cox models: {out_path}")


if __name__ == "__main__":
    main()
