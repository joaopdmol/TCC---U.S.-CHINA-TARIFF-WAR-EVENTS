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
CAR_BY_EVENT_FILE = BASE_DIR / "DATA" / "car_by_event.csv"
SUMMARY_BY_GROUP_FILE = BASE_DIR / "DATA" / "event_summary_by_group.csv"
VOLATILITY_BY_GROUP_FILE = BASE_DIR / "DATA" / "volatility_by_group.csv"

CAR_OUTPUT_FILE = BASE_DIR / "FIGURES" / "car_by_event.png"
RETURNS_OUTPUT_FILE = BASE_DIR / "FIGURES" / "returns_by_group.png"
VOLATILITY_OUTPUT_FILE = BASE_DIR / "FIGURES" / "volatility_by_group.png"

WINDOW_ORDER = ["m1_p1", "m3_p3", "m5_p5"]
WINDOW_LABELS = {
    "m1_p1": "[-1,+1]",
    "m3_p3": "[-3,+3]",
    "m5_p5": "[-5,+5]",
}
PERIOD_ORDER = ["before", "event_day", "after"]
PERIOD_LABELS = {
    "before": "Before",
    "event_day": "Event day",
    "after": "After",
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
WINDOW_COLORS = {
    "m1_p1": "#1f4e79",
    "m3_p3": "#2e86c1",
    "m5_p5": "#85c1e9",
}


def ensure_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado para {description}: {path}")


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_file(CAR_BY_EVENT_FILE, "CAR simples por evento")
    ensure_file(SUMMARY_BY_GROUP_FILE, "resumo por grupo")
    ensure_file(VOLATILITY_BY_GROUP_FILE, "volatilidade por grupo")

    return (
        pd.read_csv(CAR_BY_EVENT_FILE),
        pd.read_csv(SUMMARY_BY_GROUP_FILE),
        pd.read_csv(VOLATILITY_BY_GROUP_FILE),
    )


def plot_car_by_event(car_by_event: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(18, 10))
    event_order = sorted(car_by_event["event_id"].unique().tolist())
    y_positions = np.arange(len(event_order))
    bar_height = 0.22
    offsets = [-bar_height, 0.0, bar_height]

    for offset, window_label in zip(offsets, WINDOW_ORDER, strict=True):
        subset = (
            car_by_event.loc[car_by_event["window_label"] == window_label]
            .set_index("event_id")
            .reindex(event_order)
        )
        ax.barh(
            y_positions + offset,
            subset["car_simple_sp500"],
            height=bar_height,
            color=WINDOW_COLORS[window_label],
            edgecolor="black",
            linewidth=0.5,
            label=WINDOW_LABELS[window_label],
        )

    add_reference_lines(ax, vertical_zero=True)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(event_order)
    ax.invert_yaxis()
    apply_axis_style(
        ax,
        xlabel="Retorno acumulado simples do S&P 500",
        ylabel="Evento",
        percent_x=True,
        grid_axis="x",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Retorno acumulado simples por evento e janela", fontsize=TITLE_FONTSIZE, y=0.98)
    save_figure(fig, CAR_OUTPUT_FILE, top_rect=0.92)


def plot_returns_by_group(summary_by_group: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    x_positions = np.arange(len(PERIOD_ORDER))
    bar_width = 0.34

    all_values = summary_by_group["sp500_return_mean"].dropna().tolist()
    y_limits = shared_limits(all_values, pad_ratio=0.25)

    for ax, window_label in zip(axes, WINDOW_ORDER, strict=True):
        subset = summary_by_group.loc[summary_by_group["window_label"] == window_label].copy()
        for offset_index, event_group in enumerate(GROUP_ORDER):
            group_subset = (
                subset.loc[subset["event_group"] == event_group]
                .set_index("period")
                .reindex(PERIOD_ORDER)
            )
            ax.bar(
                x_positions + ((offset_index - 0.5) * bar_width),
                group_subset["sp500_return_mean"],
                width=bar_width,
                color=GROUP_COLORS[event_group],
                edgecolor="black",
                linewidth=0.5,
                label=GROUP_LABELS[event_group],
            )

        add_reference_lines(ax, horizontal_zero=True)
        ax.set_ylim(*y_limits)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([PERIOD_LABELS[period] for period in PERIOD_ORDER], rotation=0)
        apply_axis_style(
            ax,
            xlabel="Fase da janela",
            title=WINDOW_LABELS[window_label],
            percent_y=True,
            grid_axis="y",
        )

    apply_axis_style(axes[0], ylabel="Retorno medio diario do S&P 500", percent_y=True, grid_axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    place_figure_legend(fig, handles, labels, ncol=2, anchor_y=1.03)
    fig.suptitle("Retorno medio diario por fase da janela e grupo", fontsize=TITLE_FONTSIZE, y=0.98)
    save_figure(fig, RETURNS_OUTPUT_FILE, top_rect=0.86)


def plot_volatility_by_group(volatility_by_group: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    x_positions = list(range(len(WINDOW_ORDER)))

    for event_group in GROUP_ORDER:
        subset = (
            volatility_by_group.loc[volatility_by_group["event_group"] == event_group]
            .set_index("window_label")
            .reindex(WINDOW_ORDER)
        )
        ax.plot(
            x_positions,
            subset["volatility_sp500_mean"],
            color=GROUP_COLORS[event_group],
            marker="o",
            linewidth=2.6,
            markersize=8,
            label=GROUP_LABELS[event_group],
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([WINDOW_LABELS[window] for window in WINDOW_ORDER])
    apply_axis_style(
        ax,
        xlabel="Janela de evento",
        ylabel="Volatilidade media do S&P 500",
        percent_y=True,
        grid_axis="y",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Volatilidade media por grupo e janela", fontsize=TITLE_FONTSIZE, y=0.98)
    save_figure(fig, VOLATILITY_OUTPUT_FILE, top_rect=0.88)


def main() -> None:
    configure_matplotlib()
    car_by_event, summary_by_group, volatility_by_group = load_tables()

    plot_car_by_event(car_by_event)
    plot_returns_by_group(summary_by_group)
    plot_volatility_by_group(volatility_by_group)

    print(f"Figura salva em: {CAR_OUTPUT_FILE}")
    print(f"Figura salva em: {RETURNS_OUTPUT_FILE}")
    print(f"Figura salva em: {VOLATILITY_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
