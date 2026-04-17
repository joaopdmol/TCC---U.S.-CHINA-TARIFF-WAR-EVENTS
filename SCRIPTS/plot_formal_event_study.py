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
ABNORMAL_RETURNS_FILE = BASE_DIR / "DATA" / "abnormal_returns_long.csv"
FORMAL_CAR_BY_EVENT_FILE = BASE_DIR / "DATA" / "formal_car_by_event.csv"
FORMAL_CAR_BY_GROUP_FILE = BASE_DIR / "DATA" / "formal_car_by_group.csv"

FORMAL_CAR_BY_EVENT_OUTPUT = BASE_DIR / "FIGURES" / "formal_car_by_event.png"
FORMAL_CAR_BY_GROUP_OUTPUT = BASE_DIR / "FIGURES" / "formal_car_by_group.png"
ABNORMAL_RETURNS_BY_GROUP_OUTPUT = BASE_DIR / "FIGURES" / "abnormal_returns_by_group.png"

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
WINDOW_COLORS = {
    "m1_p1": "#1f4e79",
    "m3_p3": "#2e86c1",
    "m5_p5": "#85c1e9",
}


def ensure_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado para {description}: {path}")


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_file(ABNORMAL_RETURNS_FILE, "abnormal returns")
    ensure_file(FORMAL_CAR_BY_EVENT_FILE, "formal CAR por evento")
    ensure_file(FORMAL_CAR_BY_GROUP_FILE, "formal CAR por grupo")

    abnormal_returns = pd.read_csv(ABNORMAL_RETURNS_FILE)
    formal_car_by_event = pd.read_csv(FORMAL_CAR_BY_EVENT_FILE)
    formal_car_by_group = pd.read_csv(FORMAL_CAR_BY_GROUP_FILE)
    return abnormal_returns, formal_car_by_event, formal_car_by_group


def plot_formal_car_by_event(formal_car_by_event: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(18, 10))
    event_order = sorted(formal_car_by_event["event_id"].unique().tolist())
    y_positions = np.arange(len(event_order))
    bar_height = 0.22
    offsets = [-bar_height, 0.0, bar_height]

    for offset, window_label in zip(offsets, WINDOW_ORDER, strict=True):
        subset = (
            formal_car_by_event.loc[formal_car_by_event["window_label"] == window_label]
            .set_index("event_id")
            .reindex(event_order)
        )
        ax.barh(
            y_positions + offset,
            subset["formal_car_sp500"],
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
        xlabel="CAR formal do S&P 500",
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
    fig.suptitle("CAR formal do S&P 500 por evento e janela", fontsize=TITLE_FONTSIZE, y=0.98)
    save_figure(fig, FORMAL_CAR_BY_EVENT_OUTPUT, top_rect=0.92)


def plot_formal_car_by_group(formal_car_by_group: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    x_positions = list(range(len(WINDOW_ORDER)))

    for event_group in GROUP_ORDER:
        subset = (
            formal_car_by_group.loc[formal_car_by_group["event_group"] == event_group]
            .set_index("window_label")
            .reindex(WINDOW_ORDER)
        )
        ax.plot(
            x_positions,
            subset["mean_formal_car_sp500"],
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
        ylabel="CAR formal medio do S&P 500",
        percent_y=True,
        grid_axis="y",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("CAR formal medio do S&P 500 por grupo", fontsize=TITLE_FONTSIZE, y=0.98)
    save_figure(fig, FORMAL_CAR_BY_GROUP_OUTPUT, top_rect=0.88)


def plot_abnormal_returns_by_group(abnormal_returns: pd.DataFrame) -> None:
    grouped = (
        abnormal_returns.groupby(
            ["window_label", "event_group", "relative_day"],
            dropna=False,
            observed=True,
        )
        .agg(mean_abnormal_return=("abnormal_return", "mean"))
        .reset_index()
    )

    all_y_values = grouped["mean_abnormal_return"].dropna().tolist()
    y_limits = shared_limits(all_y_values, pad_ratio=0.20)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, window_label in zip(axes, WINDOW_ORDER, strict=True):
        subset = grouped.loc[grouped["window_label"] == window_label].copy()
        for event_group in GROUP_ORDER:
            group_subset = subset.loc[subset["event_group"] == event_group].sort_values("relative_day")
            ax.plot(
                group_subset["relative_day"],
                group_subset["mean_abnormal_return"],
                marker="o",
                linewidth=2.4,
                markersize=6,
                color=GROUP_COLORS[event_group],
                label=GROUP_LABELS[event_group],
            )

        add_reference_lines(ax, horizontal_zero=True, vertical_zero=True)
        ax.set_ylim(*y_limits)
        ax.set_xticks(sorted(subset["relative_day"].unique().tolist()))
        apply_axis_style(
            ax,
            xlabel="Dia relativo ao evento",
            title=WINDOW_LABELS[window_label],
            percent_y=True,
            grid_axis="y",
        )

    apply_axis_style(axes[0], ylabel="Retorno anormal medio", percent_y=True, grid_axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    place_figure_legend(fig, handles, labels, ncol=2, anchor_y=1.03)
    fig.suptitle("Retorno anormal medio do S&P 500 por grupo", fontsize=TITLE_FONTSIZE, y=0.98)
    save_figure(fig, ABNORMAL_RETURNS_BY_GROUP_OUTPUT, top_rect=0.86)


def main() -> None:
    configure_matplotlib()
    abnormal_returns, formal_car_by_event, formal_car_by_group = load_tables()

    plot_formal_car_by_event(formal_car_by_event)
    plot_formal_car_by_group(formal_car_by_group)
    plot_abnormal_returns_by_group(abnormal_returns)

    print(f"Figura salva em: {FORMAL_CAR_BY_EVENT_OUTPUT}")
    print(f"Figura salva em: {FORMAL_CAR_BY_GROUP_OUTPUT}")
    print(f"Figura salva em: {ABNORMAL_RETURNS_BY_GROUP_OUTPUT}")


if __name__ == "__main__":
    main()
