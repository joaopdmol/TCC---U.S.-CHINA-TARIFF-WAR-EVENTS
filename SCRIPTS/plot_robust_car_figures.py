from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from academic_plot_style import (
    AXIS_FONTSIZE,
    ESCALATION_COLOR,
    LEGEND_FONTSIZE,
    RELIEF_COLOR,
    TITLE_FONTSIZE,
    add_reference_lines,
    apply_axis_style,
    configure_matplotlib,
    place_figure_legend,
    save_figure,
    shared_limits,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DATA"
FIGURES_DIR = BASE_DIR / "FIGURES"

SUMMARY_FILE = DATA_DIR / "robust_car_summary.csv"
BY_GROUP_FILE = DATA_DIR / "robust_car_by_group.csv"
LOO_FILE = DATA_DIR / "leave_one_out_car_influence.csv"
MASTER_FILE = DATA_DIR / "formal_car_master.csv"

WINDOW_ORDER = ["m1_p1", "m3_p3", "m5_p5"]
WINDOW_LABELS = {"m1_p1": "[-1,+1]", "m3_p3": "[-3,+3]", "m5_p5": "[-5,+5]"}
WINDOW_X = {w: i for i, w in enumerate(WINDOW_ORDER)}

MEAN_COLOR = "#1f77b4"
MEDIAN_COLOR = "#ff7f0e"
TRIMMED_COLOR = "#2ca02c"
WINSOR_COLOR = "#9467bd"
E08_COLOR = "#d62728"
OTHER_COLOR = "#aec7e8"

STRUCTURAL_COLOR = "#7f7f7f"


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in [SUMMARY_FILE, BY_GROUP_FILE, LOO_FILE, MASTER_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo nao encontrado: {path}. "
                "Rode primeiro build_robust_car_statistics.py."
            )
    return (
        pd.read_csv(SUMMARY_FILE),
        pd.read_csv(BY_GROUP_FILE),
        pd.read_csv(LOO_FILE),
        pd.read_csv(MASTER_FILE),
    )


def plot_robust_car_intervals_by_window(summary: pd.DataFrame) -> None:
    """Mean + bootstrap CI + median per window, single panel."""
    output = FIGURES_DIR / "robust_car_intervals_by_window.png"
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.array([WINDOW_X[w] for w in WINDOW_ORDER])
    offset = 0.12

    rows = {row["window_label"]: row for _, row in summary.iterrows()}
    means = [rows[w]["mean_car"] for w in WINDOW_ORDER]
    medians = [rows[w]["median_car"] for w in WINDOW_ORDER]
    ci_lo = [rows[w]["bootstrap_ci_mean_lower"] for w in WINDOW_ORDER]
    ci_hi = [rows[w]["bootstrap_ci_mean_upper"] for w in WINDOW_ORDER]
    yerr_lo = [m - lo for m, lo in zip(means, ci_lo)]
    yerr_hi = [hi - m for m, hi in zip(means, ci_hi)]

    ax.errorbar(
        x - offset,
        means,
        yerr=[yerr_lo, yerr_hi],
        fmt="o",
        color=MEAN_COLOR,
        markersize=9,
        capsize=5,
        linewidth=1.4,
        label="Mean CAR (bootstrap 95% CI)",
        zorder=3,
    )
    ax.scatter(
        x + offset,
        medians,
        marker="s",
        color=MEDIAN_COLOR,
        s=80,
        zorder=3,
        label="Median CAR",
    )

    add_reference_lines(ax, horizontal_zero=True)
    ax.set_xticks(x)
    ax.set_xticklabels([WINDOW_LABELS[w] for w in WINDOW_ORDER], fontsize=AXIS_FONTSIZE)
    apply_axis_style(
        ax,
        xlabel="Event window",
        ylabel="CAR (S&P 500)",
        title="Robust CAR Estimates by Event Window with Bootstrap 95% Confidence Interval",
        percent_y=True,
        grid_axis="y",
    )
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False)
    fig.suptitle("", fontsize=TITLE_FONTSIZE)
    save_figure(fig, output, top_rect=0.93)
    print(f"Figura salva em: {output}")


def plot_car_distribution_by_window(master: pd.DataFrame) -> None:
    """Boxplot + individual points per window, E08 highlighted."""
    output = FIGURES_DIR / "car_distribution_by_window.png"
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

    all_vals = master["formal_car_sp500"].dropna().tolist()
    y_limits = shared_limits(all_vals, pad_ratio=0.12)

    for ax, window in zip(axes, WINDOW_ORDER, strict=True):
        subset = master.loc[master["window_label"] == window].copy()
        values = subset["formal_car_sp500"].dropna().values

        bp = ax.boxplot(
            values,
            patch_artist=True,
            widths=0.45,
            medianprops={"color": "black", "linewidth": 1.6},
            whiskerprops={"linewidth": 1.1},
            capprops={"linewidth": 1.1},
            showfliers=False,
            positions=[0],
        )
        bp["boxes"][0].set_facecolor("#aec7e8")
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][0].set_edgecolor("black")
        bp["boxes"][0].set_linewidth(0.8)

        non_e08 = subset.loc[subset["event_id"] != "E08"]
        e08 = subset.loc[subset["event_id"] == "E08"]

        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.08, 0.08, size=len(non_e08))
        ax.scatter(
            jitter,
            non_e08["formal_car_sp500"].values,
            color=OTHER_COLOR,
            edgecolors="#444",
            linewidths=0.5,
            s=36,
            alpha=0.85,
            zorder=3,
        )

        if not e08.empty:
            e08_val = float(e08["formal_car_sp500"].values[0])
            ax.scatter(
                [0],
                [e08_val],
                color=E08_COLOR,
                edgecolors="black",
                linewidths=0.8,
                s=80,
                zorder=4,
                label="E08",
            )
            ax.annotate(
                "E08",
                xy=(0, e08_val),
                xytext=(0.18, e08_val),
                fontsize=8,
                color=E08_COLOR,
                arrowprops={"arrowstyle": "-", "color": E08_COLOR, "lw": 0.7},
            )

        add_reference_lines(ax, horizontal_zero=True)
        ax.set_ylim(*y_limits)
        ax.set_xticks([])
        apply_axis_style(
            ax,
            title=WINDOW_LABELS[window],
            percent_y=True,
            grid_axis="y",
        )

    apply_axis_style(axes[0], ylabel="Formal CAR (S&P 500)", percent_y=True, grid_axis="y")
    fig.suptitle(
        "Distribution of Formal CAR by Event Window (all events, E08 highlighted)",
        fontsize=TITLE_FONTSIZE,
        y=0.99,
    )
    save_figure(fig, output, top_rect=0.91)
    print(f"Figura salva em: {output}")


def plot_mean_median_robust_comparison(summary: pd.DataFrame) -> None:
    """Line chart comparing mean, median, trimmed mean, winsorized mean per window."""
    output = FIGURES_DIR / "mean_median_robust_car_comparison.png"
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.array([WINDOW_X[w] for w in WINDOW_ORDER])
    rows = {row["window_label"]: row for _, row in summary.iterrows()}

    series = {
        "Mean CAR": ("mean_car", MEAN_COLOR, "o-"),
        "Median CAR": ("median_car", MEDIAN_COLOR, "s--"),
        "10% Trimmed Mean": ("trimmed_mean_car", TRIMMED_COLOR, "^-."),
        "10% Winsorized Mean": ("winsorized_mean_car", WINSOR_COLOR, "D:"),
    }

    for label, (col, color, fmt) in series.items():
        vals = [rows[w][col] for w in WINDOW_ORDER]
        ax.plot(
            x,
            vals,
            fmt,
            color=color,
            markersize=8,
            linewidth=1.6,
            label=label,
        )

    add_reference_lines(ax, horizontal_zero=True)
    ax.set_xticks(x)
    ax.set_xticklabels([WINDOW_LABELS[w] for w in WINDOW_ORDER], fontsize=AXIS_FONTSIZE)
    apply_axis_style(
        ax,
        xlabel="Event window",
        ylabel="CAR (S&P 500)",
        title="Comparison of Central Tendency Estimators Across Event Windows",
        percent_y=True,
        grid_axis="y",
    )
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc="upper left")
    save_figure(fig, output, top_rect=0.93)
    print(f"Figura salva em: {output}")


def plot_leave_one_out_influence(loo: pd.DataFrame) -> None:
    """Horizontal bar chart of the largest LOO influence values per event window."""
    output = FIGURES_DIR / "leave_one_out_car_influence.png"
    top_n = 10
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharex=True, sharey=False)

    loo = loo.copy()
    loo["delta_mean_car_pp"] = loo["delta_mean_car"] * 100
    max_abs_delta = float(loo["delta_mean_car_pp"].abs().max())
    x_limit = max_abs_delta * 1.18 if max_abs_delta > 0 else 0.1

    for ax, window in zip(axes, WINDOW_ORDER, strict=True):
        subset = loo.loc[loo["window_label"] == window].copy()
        subset = (
            subset.assign(abs_delta=lambda df: df["delta_mean_car_pp"].abs())
            .sort_values("abs_delta", ascending=False)
            .head(top_n)
            .sort_values("delta_mean_car_pp", ascending=True)
            .reset_index(drop=True)
        )
        highlight_event = subset.loc[subset["abs_delta"].idxmax(), "excluded_event_id"]

        colors = [
            E08_COLOR if eid == highlight_event else OTHER_COLOR
            for eid in subset["excluded_event_id"]
        ]
        ax.barh(
            subset["excluded_event_id"],
            subset["delta_mean_car_pp"],
            color=colors,
            edgecolor="#333333",
            linewidth=0.45,
            height=0.62,
        )

        add_reference_lines(ax, vertical_zero=True)
        apply_axis_style(
            ax,
            title=WINDOW_LABELS[window],
            grid_axis="x",
        )
        ax.set_xlim(-x_limit, x_limit)
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=9)
        ax.text(
            0.02,
            0.03,
            f"Top {top_n} by |change|",
            transform=ax.transAxes,
            fontsize=9,
            color="#555555",
            ha="left",
            va="bottom",
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=E08_COLOR, label="Largest absolute change"),
        plt.Rectangle((0, 0), 1, 1, color=OTHER_COLOR, label="Other displayed events"),
    ]
    place_figure_legend(fig, handles, [h.get_label() for h in handles], ncol=2, anchor_y=0.93)
    fig.suptitle(
        "Leave-One-Out Influence on Mean CAR",
        fontsize=TITLE_FONTSIZE,
        y=0.98,
    )
    fig.supxlabel(
        "Change in mean CAR when event is excluded (percentage points)",
        fontsize=AXIS_FONTSIZE,
        y=0.035,
    )
    save_figure(fig, output, top_rect=0.86)
    print(f"Figura salva em: {output}")


def plot_robust_car_by_group(by_group: pd.DataFrame) -> None:
    """Median and trimmed mean for escalation vs relief per window."""
    output = FIGURES_DIR / "robust_car_by_group.png"
    groups_to_plot = ["escalation", "relief"]
    group_labels = {"escalation": "Escalation", "relief": "Relief"}
    group_colors = {"escalation": ESCALATION_COLOR, "relief": RELIEF_COLOR}

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

    all_vals = []
    for window in WINDOW_ORDER:
        for group in groups_to_plot:
            row = by_group.loc[
                (by_group["event_group"] == group) & (by_group["window_label"] == window)
            ]
            if not row.empty:
                all_vals += [float(row["median_car"].values[0]), float(row["trimmed_mean_car"].values[0])]
                ci_lo = float(row["bootstrap_ci_mean_lower"].values[0])
                ci_hi = float(row["bootstrap_ci_mean_upper"].values[0])
                all_vals += [ci_lo, ci_hi]

    valid_vals = [v for v in all_vals if not np.isnan(v)]
    y_limits = shared_limits(valid_vals, pad_ratio=0.15)

    bar_width = 0.32
    offsets = {"escalation": -0.18, "relief": 0.18}

    for ax, window in zip(axes, WINDOW_ORDER, strict=True):
        for group in groups_to_plot:
            row = by_group.loc[
                (by_group["event_group"] == group) & (by_group["window_label"] == window)
            ]
            if row.empty:
                continue
            row = row.iloc[0]
            n = int(row["n"])
            median_val = float(row["median_car"])
            trimmed_val = float(row["trimmed_mean_car"])
            ci_lo = float(row["bootstrap_ci_mean_lower"])
            ci_hi = float(row["bootstrap_ci_mean_upper"])
            color = group_colors[group]
            xpos = offsets[group]

            ax.bar(
                xpos - 0.08,
                median_val,
                width=bar_width * 0.45,
                color=color,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.7,
            )
            ax.bar(
                xpos + 0.08,
                trimmed_val if not np.isnan(trimmed_val) else 0,
                width=bar_width * 0.45,
                color=color,
                alpha=0.45,
                edgecolor="black",
                linewidth=0.7,
                hatch="//",
            )
            if not (np.isnan(ci_lo) or np.isnan(ci_hi)):
                mean_val = float(row["mean_car"])
                ax.errorbar(
                    xpos,
                    mean_val,
                    yerr=[[mean_val - ci_lo], [ci_hi - mean_val]],
                    fmt="none",
                    color="black",
                    capsize=4,
                    linewidth=1.2,
                )

            ax.text(
                xpos,
                y_limits[0] * 0.88,
                f"N={n}",
                ha="center",
                fontsize=7,
                color="#444",
            )

        ax.set_xticks([-0.18, 0.18])
        ax.set_xticklabels(
            [group_labels[g] for g in groups_to_plot],
            fontsize=AXIS_FONTSIZE - 1,
        )
        add_reference_lines(ax, horizontal_zero=True)
        ax.set_ylim(*y_limits)
        apply_axis_style(ax, title=WINDOW_LABELS[window], percent_y=True, grid_axis="y")

    apply_axis_style(axes[0], ylabel="CAR (S&P 500)", percent_y=True, grid_axis="y")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=ESCALATION_COLOR, alpha=0.85, label="Escalation — Median"),
        plt.Rectangle((0, 0), 1, 1, color=ESCALATION_COLOR, alpha=0.45, hatch="//", label="Escalation — 10% Trimmed Mean"),
        plt.Rectangle((0, 0), 1, 1, color=RELIEF_COLOR, alpha=0.85, label="Relief — Median"),
        plt.Rectangle((0, 0), 1, 1, color=RELIEF_COLOR, alpha=0.45, hatch="//", label="Relief — 10% Trimmed Mean"),
    ]
    place_figure_legend(
        fig,
        legend_handles,
        [h.get_label() for h in legend_handles],
        ncol=2,
        anchor_y=1.06,
    )
    fig.suptitle(
        "Robust CAR by Event Group: Median and Trimmed Mean (bootstrap CI on mean shown as error bar)",
        fontsize=TITLE_FONTSIZE - 1,
        y=1.00,
    )
    save_figure(fig, output, top_rect=0.87)
    print(f"Figura salva em: {output}")


def main() -> None:
    configure_matplotlib()
    summary, by_group, loo, master = _load_inputs()

    plot_robust_car_intervals_by_window(summary)
    plot_car_distribution_by_window(master)
    plot_mean_median_robust_comparison(summary)
    plot_leave_one_out_influence(loo)
    plot_robust_car_by_group(by_group)


if __name__ == "__main__":
    main()
