from __future__ import annotations

import pandas as pd

from analysis_utils import censor_at, ensure_dirs, fit_cox_model
from config import CONTINUOUS_COVARIATES, COVARIATE_CATEGORIES, SYNTHETIC_DATA, TABLE_DIR


PFS_CANCERS = ["Nasopharyngeal Carcinoma", "Esophageal Cancer", "Lung Cancer"]


def main() -> None:
    ensure_dirs(TABLE_DIR)
    df = pd.read_csv(SYNTHETIC_DATA)
    frames = []
    for cancer in PFS_CANCERS:
        subset = df[df["cancer_type"] == cancer].copy()
        subset = censor_at(subset, "time_to_pfs_days", "pfs_event", 5 * 365.25)
        summary, model = fit_cox_model(
            subset,
            "time_to_pfs_days",
            "pfs_event",
            list(COVARIATE_CATEGORIES),
            CONTINUOUS_COVARIATES,
        )
        summary.insert(0, "events", int(model["pfs_event"].sum()))
        summary.insert(0, "n", len(model))
        summary.insert(0, "cancer_type", cancer)
        frames.append(summary)
    out = pd.concat(frames, ignore_index=True)
    out_path = TABLE_DIR / "pfs_multivariable_cox.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved PFS Cox models: {out_path}")


if __name__ == "__main__":
    main()
