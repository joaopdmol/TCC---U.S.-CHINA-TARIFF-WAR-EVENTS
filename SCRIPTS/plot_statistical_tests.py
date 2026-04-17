from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from academic_plot_style import (
    ESCALATION_COLOR,
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
FORMAL_CAR_BY_EVENT_FILE = BASE_DIR / "DATA" / "formal_car_by_event.csv"
RETURN_VS_VOLATILITY_FILE = BASE_DIR / "DATA" / "return_vs_volatility.csv"

CAR_DISTRIBUTION_OUTPUT = BASE_DIR / "FIGURES" / "car_distribution_by_group.png"
SCATTER_OUTPUT = BASE_DIR / "FIGURES" / "scatter_return_vs_volatility.png"

WINDOW_ORDER = ["m1_p1", "m3_p3", "m5_p5"]
WINDOW_LABELS = {
    "m1_p1": "[-1,+1]",
    "m3_p3": "[-3,+3]",
    "m5_p5": "[-5,+5]",
}
GROUP_ORDER = ["escalation", "relief"]
GROUP_LABELS = {
    "escalation": "Escalation",
    "relief": "Relief",
}
GROUP_COLORS = {
    "escalation": ESCALATION_COLOR,
    "relief": RELIEF_COLOR,
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not FORMAL_CAR_BY_EVENT_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {FORMAL_CAR_BY_EVENT_FILE}. "
            "Rode primeiro build_formal_car_tables.py."
        )
    if not RETURN_VS_VOLATILITY_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {RETURN_VS_VOLATILITY_FILE}. "
            "Rode primeiro build_return_vs_volatility.py."
        )

    return pd.read_csv(FORMAL_CAR_BY_EVENT_FILE), pd.read_csv(RETURN_VS_VOLATILITY_FILE)


def plot_car_distribution_by_group(formal_car_by_event: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    all_values = formal_car_by_event["formal_car_sp500"].dropna().tolist()
    y_limits = shared_limits(all_values, pad_ratio=0.15)

    for ax, window_label in zip(axes, WINDOW_ORDER, strict=True):
        subset = formal_car_by_event.loc[formal_car_by_event["window_label"] == window_label]
        data = [
            subset.loc[subset["event_group"] == event_group, "formal_car_sp500"].dropna().values
            for event_group in GROUP_ORDER
        ]

        box = ax.boxplot(
            data,
            patch_artist=True,
            tick_labels=[GROUP_LABELS[event_group] for event_group in GROUP_ORDER],
            widths=0.55,
            medianprops={"color": "black", "linewidth": 1.6},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
        )
        for patch, event_group in zip(box["boxes"], GROUP_ORDER, strict=True):
            patch.set_facecolor(GROUP_COLORS[event_group])
            patch.set_alpha(0.75)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)

        add_reference_lines(ax, horizontal_zero=True)
        ax.set_ylim(*y_limits)
        apply_axis_style(
            ax,
            title=WINDOW_LABELS[window_label],
            percent_y=True,
            grid_axis="y",
        )

    apply_axis_style(axes[0], ylabel="CAR formal do S&P 500", percent_y=True, grid_axis="y")
    fig.suptitle("Distribuicao do CAR formal por grupo", fontsize=TITLE_FONTSIZE, y=0.98)
    save_figure(fig, CAR_DISTRIBUTION_OUTPUT, top_rect=0.92)


def plot_scatter_return_vs_volatility(return_vs_volatility: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    x_limits = shared_limits(return_vs_volatility["formal_car_sp500"].dropna().tolist(), pad_ratio=0.12)
    y_limits = shared_limits(return_vs_volatility["volatility_sp500"].dropna().tolist(), pad_ratio=0.12)

    for ax, window_label in zip(axes, WINDOW_ORDER, strict=True):
        subset = return_vs_volatility.loc[return_vs_volatility["window_label"] == window_label]
        for event_group in GROUP_ORDER:
            group_subset = subset.loc[subset["event_group"] == event_group]
            ax.scatter(
                group_subset["formal_car_sp500"],
                group_subset["volatility_sp500"],
                color=GROUP_COLORS[event_group],
                alpha=0.70,
                s=56,
                edgecolors="white",
                linewidths=0.5,
                label=GROUP_LABELS[event_group],
            )

        add_reference_lines(ax, vertical_zero=True)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        apply_axis_style(
            ax,
            xlabel="CAR formal do S&P 500",
            title=WINDOW_LABELS[window_label],
            percent_x=True,
            percent_y=True,
            grid_axis="both",
        )

    apply_axis_style(
        axes[0],
        ylabel="Volatilidade do S&P 500",
        percent_x=True,
        percent_y=True,
        grid_axis="both",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    place_figure_legend(fig, handles, labels, ncol=2, anchor_y=1.03)
    fig.suptitle("CAR formal versus volatilidade por janela", fontsize=TITLE_FONTSIZE, y=0.98)
    save_figure(fig, SCATTER_OUTPUT, top_rect=0.86)


def main() -> None:
    configure_matplotlib()
    formal_car_by_event, return_vs_volatility = load_inputs()

    plot_car_distribution_by_group(formal_car_by_event)
    plot_scatter_return_vs_volatility(return_vs_volatility)

    print(f"Figura salva em: {CAR_DISTRIBUTION_OUTPUT}")
    print(f"Figura salva em: {SCATTER_OUTPUT}")


if __name__ == "__main__":
    main()
