from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import chi2

from analysis_utils import censor_at, ensure_dirs, e_value
from config import (
    CANCER_ORDER,
    CONTINUOUS_COVARIATES,
    COVARIATE_CATEGORIES,
    SYNTHETIC_DATA,
    TABLE_DIR,
    TIMING_ORDER,
)


def encode_for_interaction(df: pd.DataFrame, include_interactions: bool) -> pd.DataFrame:
    model = df[
        [
            "time_to_os_days",
            "os_event",
            "cancer_type",
            *CONTINUOUS_COVARIATES,
            *COVARIATE_CATEGORIES.keys(),
        ]
    ].dropna().copy()
    model["cancer_type"] = pd.Categorical(model["cancer_type"], categories=["Lung Cancer", *[x for x in CANCER_ORDER if x != "Lung Cancer"]])
    model["timing_group"] = pd.Categorical(model["timing_group"], categories=TIMING_ORDER)
    encoded = pd.get_dummies(model, columns=["cancer_type", *COVARIATE_CATEGORIES.keys()], drop_first=True)
    if include_interactions:
        cancer_cols = [c for c in encoded.columns if c.startswith("cancer_type_")]
        timing_cols = [c for c in encoded.columns if c.startswith("timing_group_")]
        for cancer_col in cancer_cols:
            for timing_col in timing_cols:
                encoded[f"{cancer_col}: {timing_col}"] = encoded[cancer_col] * encoded[timing_col]
    for col in encoded.columns:
        encoded[col] = pd.to_numeric(encoded[col], errors="coerce")
    encoded = encoded.dropna()
    zero_var = [c for c in encoded.columns if c not in {"time_to_os_days", "os_event"} and encoded[c].nunique() <= 1]
    return encoded.drop(columns=zero_var)


def fit(encoded: pd.DataFrame) -> CoxPHFitter:
    cph = CoxPHFitter()
    try:
        cph.fit(encoded, "time_to_os_days", "os_event")
    except Exception:
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(encoded, "time_to_os_days", "os_event")
    return cph


def main() -> None:
    ensure_dirs(TABLE_DIR)
    df = censor_at(pd.read_csv(SYNTHETIC_DATA), "time_to_os_days", "os_event", 5 * 365.25)
    main_encoded = encode_for_interaction(df, include_interactions=False)
    interaction_encoded = encode_for_interaction(df, include_interactions=True)
    main_model = fit(main_encoded)
    interaction_model = fit(interaction_encoded)

    summary = interaction_model.summary.reset_index().rename(columns={"covariate": "term"})
    summary["HR"] = np.exp(summary["coef"])
    summary["HR_lower_95"] = np.exp(summary["coef lower 95%"])
    summary["HR_upper_95"] = np.exp(summary["coef upper 95%"])
    summary["E_value"] = summary["HR"].map(e_value)
    out = summary[["term", "HR", "HR_lower_95", "HR_upper_95", "p", "E_value"]]
    out_path = TABLE_DIR / "site_time_interaction_cox.csv"
    out.to_csv(out_path, index=False)

    statistic = 2 * (interaction_model.log_likelihood_ - main_model.log_likelihood_)
    df_diff = interaction_model.params_.shape[0] - main_model.params_.shape[0]
    p_value = chi2.sf(statistic, df_diff)
    lrt_path = TABLE_DIR / "site_time_interaction_lrt.txt"
    lrt_path.write_text(
        f"Likelihood ratio test for cancer site by treatment timing interaction\n"
        f"Chi-square: {statistic:.3f}\n"
        f"Degrees of freedom: {df_diff}\n"
        f"P value: {p_value:.6g}\n",
        encoding="utf-8",
    )
    print(f"Saved interaction model: {out_path}")
    print(f"Saved interaction LRT: {lrt_path}")


if __name__ == "__main__":
    main()
