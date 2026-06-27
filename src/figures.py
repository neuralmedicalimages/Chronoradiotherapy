from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.patches as patches
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from matplotlib import pyplot as plt

from analysis_utils import ensure_dirs
from config import FIGURE_DIR, MAJOR_CANCERS, SYNTHETIC_DATA, TABLE_DIR, TIMING_ORDER
from plot_style import CANCER_COLORS, add_panel_label, save_figure, set_publication_style


# 确保 PDF 导出时文本不变成路径
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

EPS = 1e-12


def cw_to_mpl(cw_deg: float) -> float:
    """
    顺时针角度转 matplotlib 角度。

    cw_deg:
        0° 在正上方，顺时针增加。

    matplotlib:
        0° 在正右方，逆时针增加。
    """
    return (90.0 - cw_deg) % 360.0


def wedge_angles_from_cw(cw_start: float, cw_end: float) -> tuple[float, float]:
    """
    把顺时针角度区间转换为 patches.Wedge 需要的角度。
    """
    theta1 = cw_to_mpl(cw_end)
    theta2 = cw_to_mpl(cw_start)

    if theta2 < theta1:
        theta2 += 360.0

    return theta1, theta2


def polar_xy(radius: float, cw_deg: float) -> tuple[float, float]:
    """
    根据顺时针角度和半径得到平面坐标。
    """
    angle = np.radians(cw_to_mpl(cw_deg))
    return radius * np.cos(angle), radius * np.sin(angle)


def tangent_rotation_for_text(cw_deg: float) -> float:
    """
    让圆周文字沿切线方向旋转，同时避免倒置。
    """
    mpl_deg = cw_to_mpl(cw_deg)
    rot = mpl_deg - 90.0
    r = rot % 360.0

    if 90.0 < r < 270.0:
        rot += 180.0

    return rot


def scale_count_to_radius(
    value: float,
    max_value: float,
    inner_radius: float,
    outer_radius: float,
) -> float:
    """
    把患者数映射到半径。

    注意：
    这里强制以 0 为起点。
    inner_radius 对应 count = 0。
    outer_radius 对应 count = max_value。
    """
    if max_value <= EPS:
        return inner_radius

    rel = value / max_value
    rel = max(0.0, min(1.0, rel))

    return inner_radius + rel * (outer_radius - inner_radius)


def nice_ref_values(max_value: float, n: int = 4) -> list[int]:
    """
    自动生成参考圈刻度。
    """
    if max_value <= 0:
        return []

    raw_step = max_value / n
    magnitude = 10 ** np.floor(np.log10(raw_step))
    residual = raw_step / magnitude

    if residual <= 1:
        nice_step = 1 * magnitude
    elif residual <= 2:
        nice_step = 2 * magnitude
    elif residual <= 5:
        nice_step = 5 * magnitude
    else:
        nice_step = 10 * magnitude

    values = []
    current = nice_step

    while current < max_value * 1.001:
        values.append(int(round(current)))
        current += nice_step

    return values[:n]


def timing_distribution(df: pd.DataFrame) -> None:
    counts = pd.crosstab(df["year_bin"], df["timing_group"], normalize="index") * 100
    counts = counts.reindex(columns=[c for c in TIMING_ORDER if c in counts.columns])

    ax = counts.plot(kind="line", marker="o", figsize=(6, 4))
    ax.set_xlabel("Year of treatment")
    ax.set_ylabel("Patients (%)")
    ax.set_title("Treatment timing distribution over calendar periods")
    ax.grid(alpha=0.2)
    ax.legend(title="")
    ax.figure.tight_layout()

    save_figure(ax.figure, FIGURE_DIR / "timing_distribution_by_year")
    plt.close(ax.figure)


def polar_timing_distribution(df: pd.DataFrame) -> None:
    """
    用 patches.Wedge 手动画圆环时间分布图。

    关键逻辑：
    1. 00:00 在正上方。
    2. 时间顺时针增加。
    3. 144 个 bin，每个 bin 代表 10 分钟。
    4. 4 个 6 小时背景扇区。
    5. 内圈边缘代表 count = 0。
    6. 柱子的外半径代表该 10 分钟 bin 的患者数。
    """

    # =========================
    # 1. 基础参数
    # =========================
    bins_total = 144
    bins_per_segment = 36
    n_segments = 4

    outer_radius = 1.00
    inner_radius = 0.28
    plotting_outer_radius = 0.95

    gap_degree = 2.0
    side_padding_degree = 0.5
    intra_bar_gap_factor = 0.12

    segment_span_cw = 360.0 / n_segments
    lim_expand = 1.12

    segment_colors = ["#F3F6FA", "#DDE7F1", "#DEE7F0", "#D4DFEB"]
    segment_labels = [
        "Night\n00:00-06:00",
        "Morning\n06:00-12:00",
        "Afternoon\n12:00-18:00",
        "Evening\n18:00-24:00",
    ]

    # =========================
    # 2. 整理数据
    # =========================
    plot_df = df.copy()

    plot_df["plot_cancer"] = plot_df["cancer_type"].where(
        plot_df["cancer_type"].isin(MAJOR_CANCERS),
        "Other",
    )

    # 防止 rt_time_minutes = 1440 时变成第 144 个 bin
    plot_df["rt_time_minutes_plot"] = plot_df["rt_time_minutes"] % 1440

    counts = (
        plot_df.assign(
            bin_index=np.floor(plot_df["rt_time_minutes_plot"] / 10).astype(int)
        )
        .groupby(["bin_index", "plot_cancer"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(bins_total), fill_value=0)
        .reindex(columns=[*MAJOR_CANCERS, "Other"], fill_value=0)
    )

    counts_total = counts.sum(axis=1).to_numpy(dtype=float)
    max_total = float(np.max(counts_total))

    if max_total <= 0:
        return

    # =========================
    # 3. 计算每个柱子的角宽
    # =========================
    segment_draw_span = segment_span_cw - gap_degree
    available_bar_angle = segment_draw_span - 2 * side_padding_degree

    if available_bar_angle <= 0:
        raise ValueError("side_padding_degree 太大，扇区内部没有可用空间放柱子。")

    total_units = bins_per_segment + (bins_per_segment - 1) * intra_bar_gap_factor
    unit_width = available_bar_angle / total_units
    bar_angle_width = unit_width
    bar_gap_width = unit_width * intra_bar_gap_factor

    # =========================
    # 4. 创建普通平面坐标轴，不用 polar=True
    # =========================
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    lim = outer_radius * lim_expand
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # =========================
    # 5. 背景 6 小时扇区
    # =========================
    segment_label_radius = outer_radius * 1.08

    for seg in range(n_segments):
        seg_cw_start = seg * segment_span_cw + gap_degree / 2
        seg_cw_end = (seg + 1) * segment_span_cw - gap_degree / 2

        theta1, theta2 = wedge_angles_from_cw(seg_cw_start, seg_cw_end)

        bg = patches.Wedge(
            center=(0, 0),
            r=outer_radius,
            theta1=theta1,
            theta2=theta2,
            width=outer_radius - inner_radius,
            facecolor=segment_colors[seg % len(segment_colors)],
            edgecolor="#666666",
            linewidth=0.8,
            alpha=1.0,
            zorder=1,
        )
        ax.add_patch(bg)

        seg_cw_mid = (seg_cw_start + seg_cw_end) / 2
        lx, ly = polar_xy(segment_label_radius, seg_cw_mid)

        ax.text(
            lx,
            ly,
            segment_labels[seg],
            ha="center",
            va="center",
            fontsize=9,
            rotation=tangent_rotation_for_text(seg_cw_mid),
            rotation_mode="anchor",
            zorder=8,
        )

    # =========================
    # 6. 参考圈
    # =========================
    ref_values = nice_ref_values(max_total, n=4)

    for seg in range(n_segments):
        seg_cw_start = seg * segment_span_cw + gap_degree / 2
        seg_cw_end = (seg + 1) * segment_span_cw - gap_degree / 2

        theta1, theta2 = wedge_angles_from_cw(seg_cw_start, seg_cw_end)
        angles = np.linspace(theta1, theta2, 300)
        angles_rad = np.radians(angles)

        for ref_value in ref_values:
            rr = scale_count_to_radius(
                value=ref_value,
                max_value=max_total,
                inner_radius=inner_radius,
                outer_radius=plotting_outer_radius,
            )

            xs = rr * np.cos(angles_rad)
            ys = rr * np.sin(angles_rad)

            ax.plot(
                xs,
                ys,
                linestyle="--",
                color="#555555",
                linewidth=0.55,
                alpha=0.75,
                zorder=3,
            )

    # 参考圈文字
    ref_text_angle = 60.0

    for ref_value in ref_values:
        rr = scale_count_to_radius(
            value=ref_value,
            max_value=max_total,
            inner_radius=inner_radius,
            outer_radius=plotting_outer_radius,
        )

        tx, ty = polar_xy(rr, ref_text_angle)

        ax.text(
            tx,
            ty,
            f"{ref_value:,}",
            ha="center",
            va="bottom",
            color="black",
            fontsize=7,
            rotation=tangent_rotation_for_text(ref_text_angle),
            rotation_mode="anchor",
            zorder=7,
        )

    # =========================
    # 7. 堆叠柱
    # =========================
    group_order = [*MAJOR_CANCERS, "Other"]

    for i in range(bins_total):
        total = float(counts_total[i])

        if total <= 0:
            continue

        seg = int(i // bins_per_segment)
        j = int(i % bins_per_segment)

        seg_cw_start = (
            seg * segment_span_cw
            + gap_degree / 2
            + side_padding_degree
        )

        bar_cw_start = seg_cw_start + j * (bar_angle_width + bar_gap_width)
        bar_cw_end = bar_cw_start + bar_angle_width

        theta1, theta2 = wedge_angles_from_cw(bar_cw_start, bar_cw_end)

        # 关键：
        # 内圈 inner_radius 对应 0。
        # 当前 bin 的总高度按 total / max_total 映射。
        r_top_total = scale_count_to_radius(
            value=total,
            max_value=max_total,
            inner_radius=inner_radius,
            outer_radius=plotting_outer_radius,
        )

        thickness_total = r_top_total - inner_radius

        if thickness_total <= 1e-8:
            continue

        r_base = inner_radius
        row = counts.iloc[i]

        for cancer in group_order:
            part = float(row.get(cancer, 0.0))

            if part <= 0:
                continue

            part_thickness = thickness_total * (part / total)
            r_top_part = r_base + part_thickness

            wedge = patches.Wedge(
                center=(0, 0),
                r=r_top_part,
                theta1=theta1,
                theta2=theta2,
                width=part_thickness,
                facecolor=CANCER_COLORS.get(cancer, "#999999"),
                edgecolor="white",
                linewidth=0.35,
                alpha=0.95,
                joinstyle="round",
                zorder=5,
            )

            ax.add_patch(wedge)
            r_base = r_top_part

    # =========================
    # 8. 中心文字
    # =========================
    ax.text(
        0,
        0,
        "Patient-level\nMedian RT Time\nDistribution",
        ha="center",
        va="center",
        fontsize=8,
        zorder=10,
    )

    # =========================
    # 9. 图例
    # =========================
    handles = [
        patches.Patch(
            facecolor=CANCER_COLORS.get(cancer, "#999999"),
            label=cancer.replace("Nasopharyngeal Carcinoma", "NPC"),
            alpha=0.95,
        )
        for cancer in group_order
    ]

    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        frameon=False,
        fontsize=7,
    )

    # =========================
    # 10. 标题与保存
    # =========================
    ax.set_title(
        "Course Median Treatment Time\n(one value per patient)",
        fontsize=9,
        y=1.04,
    )

    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "figure1_timing_polar")
    plt.close(fig)


def night_effect_forest() -> None:
    path = TABLE_DIR / "main_multivariable_cox.csv"

    if not path.exists():
        return

    cox = pd.read_csv(path)

    plot = cox[
        (cox["follow_up"] == "5 years")
        & (cox["term"] == "timing_group_Night")
    ].copy()

    plot = plot.sort_values("HR")

    if plot.empty:
        return

    y = range(len(plot))

    fig, ax = plt.subplots(figsize=(6.6, max(4.2, 0.34 * len(plot))))

    ax.errorbar(
        plot["HR"],
        y,
        xerr=[
            plot["HR"] - plot["HR_lower_95"],
            plot["HR_upper_95"] - plot["HR"],
        ],
        fmt="s",
        color="black",
        ecolor="black",
        markersize=4.8,
        elinewidth=1.1,
        capsize=3,
    )

    ax.axvline(1, color="#999999", linestyle="--", linewidth=0.9)

    ax.set_xscale("log")
    ax.set_xlim(0.45, 2.2)

    ticks = [0.5, 0.75, 1.0, 1.5, 2.0]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.5", "0.75", "1", "1.5", "2"]))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.set_yticks(list(y))
    ax.set_yticklabels(plot["population"])

    ax.set_xlabel("Hazard ratio for night vs morning treatment")
    ax.set_title("Cancer-specific multivariable Cox regression")
    ax.grid(axis="x", alpha=0.2)

    add_panel_label(ax, "A", x=-0.18, y=1.03)

    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "figure2_night_effect_forest")
    plt.close(fig)


def main() -> None:
    set_publication_style()
    ensure_dirs(FIGURE_DIR)

    df = pd.read_csv(SYNTHETIC_DATA)

    polar_timing_distribution(df)
    timing_distribution(df)
    night_effect_forest()

    print(f"Saved figures to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()