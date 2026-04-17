from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
WINDOW_COMPARISON_BY_GROUP_FILE = BASE_DIR / "DATA" / "window_comparison_by_group.csv"
GROUP_SIGN_SUMMARY_FILE = BASE_DIR / "DATA" / "group_sign_summary.csv"
RETURN_VS_VOLATILITY_FILE = BASE_DIR / "DATA" / "return_vs_volatility.csv"

WINDOW_COMPARISON_OUTPUT = BASE_DIR / "FIGURES" / "window_comparison.png"
EVENT_SIGN_DISTRIBUTION_OUTPUT = BASE_DIR / "FIGURES" / "event_sign_distribution.png"
RETURN_VS_VOLATILITY_OUTPUT = BASE_DIR / "FIGURES" / "return_vs_volatility.png"

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


def ensure_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado para {description}: {path}")


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_file(WINDOW_COMPARISON_BY_GROUP_FILE, "comparacao entre janelas")
    ensure_file(GROUP_SIGN_SUMMARY_FILE, "analise de sinal")
    ensure_file(RETURN_VS_VOLATILITY_FILE, "retorno vs volatilidade")

    return (
        pd.read_csv(WINDOW_COMPARISON_BY_GROUP_FILE),
        pd.read_csv(GROUP_SIGN_SUMMARY_FILE),
        pd.read_csv(RETURN_VS_VOLATILITY_FILE),
    )


def plot_window_comparison(window_comparison_by_group: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    x_positions = list(range(len(WINDOW_ORDER)))

    for event_group in GROUP_ORDER:
        subset = window_comparison_by_group.loc[
            window_comparison_by_group["event_group"] == event_group
        ]
        values = [
            subset[f"mean_formal_car_{window}"].iloc[0]
            if f"mean_formal_car_{window}" in subset.columns
            else np.nan
            for window in WINDOW_ORDER
        ]
        ax.plot(
            x_positions,
            values,
            color=GROUP_COLORS[event_group],
            marker="o",
            linewidth=2.6,
            markersize=8,
            label=GROUP_LABELS[event_group],
        )

    add_reference_lines(ax, horizontal_zero=True)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([WINDOW_LABELS[window] for window in WINDOW_ORDER])
    apply_axis_style(
        ax,
        xlabel="Janela de evento",
        ylabel="CAR formal medio",
        percent_y=True,
        grid_axis="y",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Comparacao do CAR formal medio entre janelas", fontsize=TITLE_FONTSIZE, y=0.98)
    save_figure(fig, WINDOW_COMPARISON_OUTPUT, top_rect=0.88)


def plot_event_sign_distribution(group_sign_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    sign_order = ["negative", "neutral", "positive"]
    sign_labels = ["Negativo", "Neutro", "Positivo"]
    x_positions = np.arange(len(sign_order))
    bar_width = 0.34

    all_values = group_sign_summary["event_share"].dropna().tolist()
    y_limits = shared_limits(all_values, pad_ratio=0.15)
    y_limits = (0, max(y_limits[1], 0.65))

    for ax, window_label in zip(axes, WINDOW_ORDER, strict=True):
        subset = group_sign_summary.loc[group_sign_summary["window_label"] == window_label]

        for offset_index, event_group in enumerate(GROUP_ORDER):
            group_subset = (
                subset.loc[subset["event_group"] == event_group]
                .set_index("sign_label")
                .reindex(sign_order)
            )
            ax.bar(
                x_positions + ((offset_index - 0.5) * bar_width),
                group_subset["event_share"],
                width=bar_width,
                color=GROUP_COLORS[event_group],
                edgecolor="black",
                linewidth=0.5,
                label=GROUP_LABELS[event_group],
            )

        ax.set_ylim(*y_limits)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(sign_labels)
        apply_axis_style(
            ax,
            xlabel="Sinal do CAR",
            title=WINDOW_LABELS[window_label],
            percent_y=True,
            grid_axis="y",
        )

    apply_axis_style(axes[0], ylabel="Proporcao de eventos", percent_y=True, grid_axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    place_figure_legend(fig, handles, labels, ncol=2, anchor_y=1.03)
    fig.suptitle("Distribuicao do sinal do CAR por grupo", fontsize=TITLE_FONTSIZE, y=0.98)
    save_figure(fig, EVENT_SIGN_DISTRIBUTION_OUTPUT, top_rect=0.86)


def plot_return_vs_volatility(return_vs_volatility: pd.DataFrame) -> None:
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
                s=54,
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
    fig.suptitle("Retorno acumulado e volatilidade por janela", fontsize=TITLE_FONTSIZE, y=0.98)
    save_figure(fig, RETURN_VS_VOLATILITY_OUTPUT, top_rect=0.86)


def main() -> None:
    configure_matplotlib()
    window_comparison_by_group, group_sign_summary, return_vs_volatility = load_tables()

    plot_window_comparison(window_comparison_by_group)
    plot_event_sign_distribution(group_sign_summary)
    plot_return_vs_volatility(return_vs_volatility)

    print(f"Figura salva em: {WINDOW_COMPARISON_OUTPUT}")
    print(f"Figura salva em: {EVENT_SIGN_DISTRIBUTION_OUTPUT}")
    print(f"Figura salva em: {RETURN_VS_VOLATILITY_OUTPUT}")


if __name__ == "__main__":
    main()
