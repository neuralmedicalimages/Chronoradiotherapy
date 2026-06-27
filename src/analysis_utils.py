from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from config import CONTINUOUS_COVARIATES, COVARIATE_CATEGORIES


def ensure_dirs(*paths) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def censor_at(df: pd.DataFrame, time_col: str, event_col: str, days: float) -> pd.DataFrame:
    out = df.copy()
    over = out[time_col] > days
    out.loc[over, time_col] = days
    out.loc[over, event_col] = 0
    return out


def e_value(hr: float) -> float:
    if pd.isna(hr) or hr <= 0:
        return np.nan
    value = 1.0 / hr if hr < 1 else hr
    return float(value + np.sqrt(value * (value - 1)))


def format_p(value: float) -> str:
    if pd.isna(value):
        return ""
    return "<.001" if value < 0.001 else f"{value:.3f}"


def prepare_model_frame(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    categorical_cols: list[str] | None = None,
    continuous_cols: list[str] | None = None,
) -> pd.DataFrame:
    categorical_cols = categorical_cols or list(COVARIATE_CATEGORIES)
    continuous_cols = continuous_cols or CONTINUOUS_COVARIATES
    cols = [time_col, event_col] + continuous_cols + categorical_cols
    model = df[cols].copy()
    for col, order in COVARIATE_CATEGORIES.items():
        if col in model.columns:
            present = [x for x in order if x in set(model[col].dropna())]
            model[col] = pd.Categorical(model[col], categories=present, ordered=False)
    return model.dropna()


def fit_cox_model(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    categorical_cols: list[str] | None = None,
    continuous_cols: list[str] | None = None,
    robust: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = prepare_model_frame(df, time_col, event_col, categorical_cols, continuous_cols)
    categorical_cols = categorical_cols or list(COVARIATE_CATEGORIES)
    candidate_categoricals = [c for c in categorical_cols if c in model.columns]
    categorical_cols = [c for c in candidate_categoricals if model[c].nunique() > 1]
    constant_categoricals = [c for c in candidate_categoricals if c not in categorical_cols]
    if constant_categoricals:
        model = model.drop(columns=constant_categoricals)
    encoded = pd.get_dummies(model, columns=categorical_cols, drop_first=True)
    for col in encoded.columns:
        encoded[col] = pd.to_numeric(encoded[col], errors="coerce")
    encoded = encoded.dropna()
    feature_cols = [c for c in encoded.columns if c not in {time_col, event_col}]
    zero_var = [c for c in feature_cols if encoded[c].nunique(dropna=False) <= 1]
    if zero_var:
        encoded = encoded.drop(columns=zero_var)

    cph = CoxPHFitter()
    try:
        cph.fit(encoded, duration_col=time_col, event_col=event_col, robust=robust)
    except Exception:
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(encoded, duration_col=time_col, event_col=event_col, robust=robust)

    out = cph.summary.reset_index().rename(columns={"covariate": "term"})
    if "term" not in out.columns:
        out = out.rename(columns={out.columns[0]: "term"})
    out["HR"] = np.exp(out["coef"])
    out["HR_lower_95"] = np.exp(out["coef lower 95%"])
    out["HR_upper_95"] = np.exp(out["coef upper 95%"])
    out["E_value"] = out["HR"].map(e_value)
    keep = ["term", "HR", "HR_lower_95", "HR_upper_95", "p", "E_value"]
    return out[keep], model


def coefficient_row(summary: pd.DataFrame, term: str) -> dict:
    match = summary.loc[summary["term"].eq(term)]
    if match.empty:
        return {"HR": np.nan, "HR_lower_95": np.nan, "HR_upper_95": np.nan, "p": np.nan}
    row = match.iloc[0]
    return {
        "HR": row["HR"],
        "HR_lower_95": row["HR_lower_95"],
        "HR_upper_95": row["HR_upper_95"],
        "p": row["p"],
    }
