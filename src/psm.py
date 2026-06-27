from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from analysis_utils import censor_at, ensure_dirs, fit_cox_model
from config import COVARIATE_CATEGORIES, FIGURE_DIR, SYNTHETIC_DATA, TABLE_DIR
from plot_style import GREEN, ORANGE, add_panel_label, format_p, save_figure, set_publication_style


PSM_CANCERS = ["Nasopharyngeal Carcinoma", "Esophageal Cancer", "Lung Cancer"]
BASE_COVARIATES = [
    "age_at_rt",
    "sex",
    "year_bin",
    "stage",
    "approval_to_rt_days",
    "kps_group",
    "concurrent_chemo",
    "total_dose_group",
    "fraction_dose_group",
]


def propensity_frame(df: pd.DataFrame) -> pd.DataFrame:
    model = df[df["timing_group"].isin(["Morning", "Afternoon", "Night"])].copy()
    model["treatment_group"] = np.where(model["timing_group"] == "Night", "Night", "Daytime")
    model["night_treatment"] = (model["treatment_group"] == "Night").astype(int)
    encoded = pd.get_dummies(model[BASE_COVARIATES], drop_first=True)
    encoded = encoded.apply(pd.to_numeric, errors="coerce").fillna(0)
    return model, encoded


def match_one_to_one(model: pd.DataFrame, encoded: pd.DataFrame, caliper: float = 0.2) -> pd.DataFrame:
    treatment = model["night_treatment"].to_numpy()
    clf = LogisticRegression(max_iter=1000)
    clf.fit(encoded, treatment)
    ps = clf.predict_proba(encoded)[:, 1]
    model = model.copy()
    model["propensity_score"] = ps
    model["logit_ps"] = np.log(ps / (1 - ps))

    treated = model[model["night_treatment"] == 1].copy()
    controls = model[model["night_treatment"] == 0].copy()
    if treated.empty or controls.empty:
        return model.iloc[0:0].copy()

    nn = NearestNeighbors(n_neighbors=1).fit(controls[["logit_ps"]])
    distances, indices = nn.kneighbors(treated[["logit_ps"]])
    threshold = caliper * np.nanstd(model["logit_ps"])
    used_controls: set[int] = set()
    matched_rows = []
    for treated_pos, (distance, control_pos) in enumerate(zip(distances[:, 0], indices[:, 0])):
        control_index = controls.index[control_pos]
        if distance <= threshold and control_index not in used_controls:
            matched_rows.extend([treated.index[treated_pos], control_index])
            used_controls.add(control_index)
    matched = model.loc[matched_rows].copy()
    matched["matched_pair_count"] = len(matched_rows) // 2
    return matched


def add_risk_table(ax, data: pd.DataFrame, times: list[int]) -> None:
    ax.axis("off")
    ax.text(-4, 0.72, "At risk", fontsize=7, ha="right")
    for row_index, (group_value, label) in enumerate([(1, "Night"), (0, "Daytime")]):
        subset = data[data["night_treatment"] == group_value]
        y = 0.42 - row_index * 0.35
        ax.text(-4, y, label, fontsize=7, ha="right")
        for month in times:
            day = month * 30.4375
            count = int((subset["time_to_os_days"] >= day).sum())
            ax.text(month, y, str(count), fontsize=7, ha="center")
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.25, 1)


def plot_one_km(ax, table_ax, data: pd.DataFrame, title: str, panel_label: str) -> None:
    kmf = KaplanMeierFitter()
    for group, label, color in [(1, "Night (0-6)", GREEN), (0, "Daytime (6-18)", ORANGE)]:
        subset = data[data["night_treatment"] == group]
        if subset.empty:
            continue
        kmf.fit(subset["time_to_os_days"] / 30.4375, subset["os_event"], label=f"{label} (n={len(subset)})")
        survival = kmf.survival_function_.iloc[:, 0] * 100
        ci = kmf.confidence_interval_survival_function_ * 100
        ax.plot(survival.index, survival.values, color=color, linewidth=1.5, label=f"{label} (n={len(subset)})")
        ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color=color, alpha=0.16, linewidth=0)
    night = data[data["night_treatment"] == 1]
    daytime = data[data["night_treatment"] == 0]
    if len(night) > 5 and len(daytime) > 5:
        test = logrank_test(night["time_to_os_days"], daytime["time_to_os_days"], night["os_event"], daytime["os_event"])
        ax.text(3, 18, f"Log-rank P {format_p(test.p_value)}", fontsize=7)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 105)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("Time Since Treatment (Months)", fontweight="bold", labelpad=7)
    ax.set_ylabel("Survival Probability (%)", fontweight="bold")
    ax.grid(False)
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    add_panel_label(ax, panel_label, x=0.0, y=1.10)
    add_risk_table(table_ax, data, [0, 12, 24, 36, 48, 60])


def plot_km(original: pd.DataFrame, matched: pd.DataFrame, cancer: str) -> None:
    original = censor_at(original, "time_to_os_days", "os_event", 5 * 365.25)
    matched = censor_at(matched, "time_to_os_days", "os_event", 5 * 365.25)
    fig = plt.figure(figsize=(10.8, 4.2))
    grid = GridSpec(2, 2, figure=fig, height_ratios=[4, 0.75], hspace=0.28, wspace=0.28)
    ax_a = fig.add_subplot(grid[0, 0])
    risk_a = fig.add_subplot(grid[1, 0], sharex=ax_a)
    ax_b = fig.add_subplot(grid[0, 1])
    risk_b = fig.add_subplot(grid[1, 1], sharex=ax_b)
    plot_one_km(ax_a, risk_a, original, f"{cancer} Pre-PSM Survival", "A")
    plot_one_km(ax_b, risk_b, matched, f"{cancer} PSM Matched Survival", "B")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.13, wspace=0.28, hspace=0.28)
    save_figure(fig, FIGURE_DIR / f"psm_km_{cancer.lower().replace(' ', '_')}")
    plt.close(fig)


def run_cancer(df: pd.DataFrame, cancer: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    model, encoded = propensity_frame(df[df["cancer_type"] == cancer])
    matched = match_one_to_one(model, encoded)
    matched = censor_at(matched, "time_to_os_days", "os_event", 5 * 365.25)
    if matched.empty:
        return matched, pd.DataFrame()
    categorical = ["treatment_group", "sex", "year_bin", "stage", "kps_group", "concurrent_chemo", "total_dose_group", "fraction_dose_group"]
    summary, model_frame = fit_cox_model(matched, "time_to_os_days", "os_event", categorical, ["age_at_rt", "approval_to_rt_days"])
    summary.insert(0, "events", int(model_frame["os_event"].sum()))
    summary.insert(0, "n", len(model_frame))
    summary.insert(0, "cancer_type", cancer)
    plot_km(model, matched, cancer)
    return matched, summary


def main() -> None:
    set_publication_style()
    ensure_dirs(TABLE_DIR, FIGURE_DIR)
    df = pd.read_csv(SYNTHETIC_DATA)
    matched_frames = []
    result_frames = []
    for cancer in PSM_CANCERS:
        matched, result = run_cancer(df, cancer)
        if not matched.empty:
            matched_frames.append(matched)
        if not result.empty:
            result_frames.append(result)
    if matched_frames:
        pd.concat(matched_frames, ignore_index=True).to_csv(TABLE_DIR / "psm_matched_synthetic_cohort.csv", index=False)
    if result_frames:
        out = pd.concat(result_frames, ignore_index=True)
        out_path = TABLE_DIR / "psm_matched_cox.csv"
        out.to_csv(out_path, index=False)
        print(f"Saved PSM Cox models: {out_path}")


if __name__ == "__main__":
    main()
