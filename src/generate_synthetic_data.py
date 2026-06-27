from __future__ import annotations

import numpy as np
import pandas as pd

from analysis_utils import ensure_dirs
from config import CANCER_ORDER, DATA_DIR, SYNTHETIC_DATA, YEAR_ORDER
import matplotlib.pyplot as plt

def year_bin(year: int) -> str:
    if year <= 2014:
        return "2010-2014"
    if year <= 2019:
        return "2015-2019"
    return "2020-2025"


def timing_group(minutes: np.ndarray) -> np.ndarray:
    hour = (minutes / 60.0) % 24
    return np.select(
        [hour < 6, hour < 12, hour < 18],
        ["Night", "Morning", "Afternoon"],
        default="Evening",
    )


def season_from_month(month: np.ndarray) -> np.ndarray:
    return np.select(
        [np.isin(month, [3, 4, 5]), np.isin(month, [6, 7, 8]), np.isin(month, [9, 10, 11])],
        ["Spring", "Summer", "Autumn"],
        default="Winter",
    )


def dose_groups(values: np.ndarray, cuts: tuple[float, float]) -> np.ndarray:
    return np.select([values <= cuts[0], values <= cuts[1]], ["Low", "Middle"], default="High")


def simulate_endpoint(rng: np.random.Generator, linear_predictor: np.ndarray, max_follow: float) -> tuple[np.ndarray, np.ndarray]:
    baseline = 1 / 2800
    event_time = rng.exponential(scale=1 / (baseline * np.exp(linear_predictor)))
    censor_time = rng.uniform(365, max_follow, size=len(linear_predictor))
    observed_time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)
    return observed_time.round(1), event


def build_synthetic_dataset(n: int = 80000, seed: int = 20260627) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    cancer_probs = np.array([0.23, 0.10, 0.11, 0.08, 0.08, 0.07, 0.06, 0.05, 0.05, 0.035, 0.035, 0.025, 0.025, 0.06])
    cancer_probs = cancer_probs / cancer_probs.sum()
    cancer = rng.choice(CANCER_ORDER, size=n, p=cancer_probs)

    year_probs = np.array([0.035] * 5 + [0.055] * 5 + [0.075] * 6)
    year = rng.choice(np.arange(2010, 2026), size=n, p=year_probs / year_probs.sum())
    month = rng.integers(1, 13, size=n)
    timing_component = rng.choice(["morning", "afternoon", "evening", "night"], size=n, p=[0.29, 0.32, 0.27, 0.12])
    rt_minutes = np.empty(n)
    component_params = {
        "morning": (10 * 60, 70 * 2),
        "afternoon": (15 * 60, 70 * 2),
        "evening": (21 * 60, 80 * 2),
        "night": (1 * 60, 100),
    }
    for component, (mean, sd) in component_params.items():
        mask = timing_component == component
        rt_minutes[mask] = rng.normal(mean, sd, mask.sum())
    rt_minutes[rt_minutes < 0] = rt_minutes[rt_minutes < 0] + 24 * 60
    rt_minutes[rt_minutes > 24 * 60] = rt_minutes[rt_minutes > 24 * 60] - 24 * 60
    timing = timing_group(rt_minutes)
    plt.hist(rt_minutes, bins=240)

    age_base = rng.normal(59, 13, n)
    age_shift = pd.Series(cancer).map(
        {
            "Breast Cancer": -6,
            "Nasopharyngeal Carcinoma": -8,
            "Prostate Cancer": 10,
            "Lung Cancer": 5,
            "Pancreatic Cancer": 6,
        }
    ).fillna(0).to_numpy()
    age = np.clip(age_base + age_shift, 18, 90).round(1)

    female_prob = pd.Series(cancer).map(
        {
            "Breast Cancer": 0.98,
            "Cervical Cancer": 1.0,
            "Prostate Cancer": 0.0,
            "Lung Cancer": 0.38,
            "Esophageal Cancer": 0.25,
            "Nasopharyngeal Carcinoma": 0.32,
        }
    ).fillna(0.48).to_numpy()
    sex = np.where(rng.random(n) < female_prob, "Female", "Male")

    stage = rng.choice(
        ["Stage I", "Stage II", "Stage III", "Stage IV", "Recurrence", "Unknown"],
        size=n,
        p=[0.18, 0.26, 0.29, 0.18, 0.03, 0.06],
    )
    kps = rng.choice([">=90", "<90", "Not documented"], size=n, p=[0.58, 0.18, 0.24])
    chemo_prob = np.where(np.isin(cancer, ["Nasopharyngeal Carcinoma", "Esophageal Cancer", "Lung Cancer", "Cervical Cancer"]), 0.48, 0.18)
    concurrent_chemo = np.where(rng.random(n) < chemo_prob, "Yes", "Not documented")

    fraction_dose = rng.normal(210, 25, n)
    fraction_dose += np.where(np.isin(cancer, ["Lung Cancer", "Liver Cancer", "Pancreatic Cancer"]), rng.gamma(2.0, 65, n), 0)
    fraction_dose = np.clip(fraction_dose, 150, 1200).round(1)
    num_fractions = np.clip(np.round(rng.normal(28, 6, n)), 3, 40).astype(int)
    total_dose = np.clip(fraction_dose * num_fractions + rng.normal(0, 120, n), 800, 8000).round(1)

    technique = rng.choice(["IMRT", "VMAT", "3DCRT", "SBRT", "Other"], size=n, p=[0.46, 0.28, 0.13, 0.09, 0.04])
    sbrt_like = np.where((technique == "SBRT") | (fraction_dose >= 500) | (num_fractions <= 8), "SBRT-like", "Non-SBRT")

    lung_subtype = np.where(
        cancer == "Lung Cancer",
        rng.choice(["NSCLC", "SCLC", "Unknown"], size=n, p=[0.70, 0.15, 0.15]),
        "",
    )
    colorectal_subtype = np.where(
        cancer == "Colorectal Cancer",
        rng.choice(["Colon", "Rectum", "Unclassified"], size=n, p=[0.47, 0.43, 0.10]),
        "",
    )
    lymphoma_subtype = np.where(
        cancer == "Lymphoma",
        rng.choice(["Aggressive B-cell", "Indolent B-cell", "T-cell/NK-cell", "Hodgkin", "Unclassified"], size=n, p=[0.32, 0.20, 0.12, 0.15, 0.21]),
        "",
    )

    stage_lp = pd.Series(stage).map({"Stage I": 0.0, "Stage II": 0.22, "Stage III": 0.55, "Stage IV": 1.05, "Recurrence": 0.85, "Unknown": 0.35}).to_numpy()
    cancer_lp = pd.Series(cancer).map(
        {
            "Breast Cancer": -0.65,
            "Prostate Cancer": -0.55,
            "Lung Cancer": 0.55,
            "Pancreatic Cancer": 0.95,
            "Liver Cancer": 0.70,
            "Esophageal Cancer": 0.35,
            "Nasopharyngeal Carcinoma": -0.10,
            "Lymphoma": 0.20,
        }
    ).fillna(0.0).to_numpy()
    timing_lp = np.where(timing == "Night", -0.04, 0.0)
    timing_lp += np.where((cancer == "Nasopharyngeal Carcinoma") & (timing == "Night"), -0.22, 0.0)
    timing_lp += np.where((cancer == "Lung Cancer") & (timing == "Night"), 0.18, 0.0)
    timing_lp += np.where(timing == "Evening", -0.02, 0.0)
    dose_lp = np.where(total_dose < 5000, 0.12, np.where(total_dose > 6600, -0.08, 0.0))
    kps_lp = np.where(kps == "<90", 0.35, np.where(kps == "Not documented", 0.10, 0.0))
    chemo_lp = np.where(concurrent_chemo == "Yes", -0.04, 0.0)
    age_lp = (age - 60) * 0.018
    lp_os = stage_lp + cancer_lp + timing_lp + dose_lp + kps_lp + chemo_lp + age_lp
    os_time, os_event = simulate_endpoint(rng, lp_os, 12 * 365.25)

    pfs_lp = lp_os + np.where(cancer == "Nasopharyngeal Carcinoma", -0.10, 0.05)
    pfs_lp += np.where((cancer == "Esophageal Cancer") & (timing == "Night"), -0.12, 0.0)
    pfs_time, pfs_event = simulate_endpoint(rng, pfs_lp + 0.35, 8 * 365.25)

    df = pd.DataFrame(
        {
            "patient_id": [f"SYN{idx:06d}" for idx in range(1, n + 1)],
            "cancer_type": cancer,
            "age_at_rt": age,
            "sex": sex,
            "year": year,
            "year_bin": [year_bin(int(y)) for y in year],
            "month": month,
            "season": season_from_month(month),
            "stage": stage,
            "approval_to_rt_days": np.clip(rng.gamma(2.2, 3.0, n), 0, 60).round(1),
            "rt_time_minutes": rt_minutes.round(1),
            "timing_group": timing,
            "kps_group": kps,
            "concurrent_chemo": concurrent_chemo,
            "total_dose": total_dose,
            "fraction_dose": fraction_dose,
            "num_fractions": num_fractions,
            "total_dose_group": dose_groups(total_dose, (5040, 6600)),
            "fraction_dose_group": dose_groups(fraction_dose, (220, 500)),
            "technique": technique,
            "sbrt_like": sbrt_like,
            "lung_subtype": lung_subtype,
            "colorectal_subtype": colorectal_subtype,
            "lymphoma_subtype": lymphoma_subtype,
            "time_to_os_days": os_time,
            "os_event": os_event,
            "time_to_pfs_days": pfs_time,
            "pfs_event": pfs_event,
        }
    )
    return df


def main() -> pd.DataFrame:
    ensure_dirs(DATA_DIR)
    df = build_synthetic_dataset()
    df.to_csv(SYNTHETIC_DATA, index=False)
    print(f"Saved synthetic analysis-ready data: {SYNTHETIC_DATA} ({len(df):,} rows)")
    return df


if __name__ == "__main__":
    main()
