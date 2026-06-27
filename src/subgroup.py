from __future__ import annotations

import pandas as pd

from analysis_utils import censor_at, coefficient_row, ensure_dirs, fit_cox_model
from config import SYNTHETIC_DATA, TABLE_DIR


SUBGROUP_CANCERS = ["Nasopharyngeal Carcinoma", "Esophageal Cancer", "Lung Cancer", "Colorectal Cancer"]
BASE_CATEGORICAL = ["timing_group", "sex", "year_bin", "stage", "kps_group", "concurrent_chemo", "total_dose_group", "fraction_dose_group"]
BASE_CONTINUOUS = ["age_at_rt", "approval_to_rt_days"]


def subgroup_specs(cancer: str) -> list[tuple[str, str]]:
    common = [
        ("sex", "Sex"),
        ("stage", "Tumor stage"),
        ("kps_group", "KPS group"),
        ("concurrent_chemo", "Concurrent chemotherapy"),
        ("total_dose_group", "Total dose group"),
        ("fraction_dose_group", "Dose per fraction group"),
        ("sbrt_like", "SBRT-like regimen"),
    ]
    if cancer == "Lung Cancer":
        common.append(("lung_subtype", "Lung subtype"))
    if cancer == "Colorectal Cancer":
        common.append(("colorectal_subtype", "Colorectal site"))
    return common


def run_subgroup(df: pd.DataFrame, cancer: str, variable: str, label: str) -> list[dict]:
    rows: list[dict] = []
    for level, subset in df.groupby(variable, dropna=True):
        if level == "" or len(subset) < 80 or subset["os_event"].sum() < 10:
            continue
        categorical = [c for c in BASE_CATEGORICAL if c != variable]
        summary, model = fit_cox_model(subset, "time_to_os_days", "os_event", categorical, BASE_CONTINUOUS)
        effect = coefficient_row(summary, "timing_group_Night")
        effect.update(
            {
                "cancer_type": cancer,
                "subgroup": label,
                "level": level,
                "n": len(model),
                "events": int(model["os_event"].sum()),
                "comparison": "Night vs Morning",
            }
        )
        rows.append(effect)
    return rows


def main() -> None:
    ensure_dirs(TABLE_DIR)
    df = pd.read_csv(SYNTHETIC_DATA)
    df = df[df["timing_group"].isin(["Morning", "Night"])].copy()
    df = censor_at(df, "time_to_os_days", "os_event", 5 * 365.25)
    rows: list[dict] = []
    for cancer in SUBGROUP_CANCERS:
        subset = df[df["cancer_type"] == cancer].copy()
        for variable, label in subgroup_specs(cancer):
            rows.extend(run_subgroup(subset, cancer, variable, label))
    out = pd.DataFrame(rows)
    out_path = TABLE_DIR / "subgroup_night_vs_morning.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved subgroup analysis: {out_path}")


if __name__ == "__main__":
    main()
