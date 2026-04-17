from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
ABNORMAL_RETURNS_FILE = BASE_DIR / "DATA" / "abnormal_returns_long.csv"
FORMAL_CAR_BY_EVENT_FILE = BASE_DIR / "DATA" / "formal_car_by_event.csv"
FORMAL_CAR_BY_GROUP_FILE = BASE_DIR / "DATA" / "formal_car_by_group.csv"

FORMAL_CAR_BY_EVENT_OUTPUT = BASE_DIR / "FIGURES" / "formal_car_by_event.png"
FORMAL_CAR_BY_GROUP_OUTPUT = BASE_DIR / "FIGURES" / "formal_car_by_group.png"
ABNORMAL_RETURNS_BY_GROUP_OUTPUT = BASE_DIR / "FIGURES" / "abnormal_returns_by_group.png"

WINDOW_ORDER = ["m1_p1", "m3_p3", "m5_p5"]
WINDOW_LABELS = {
    "m1_p1": "[-1, +1]",
    "m3_p3": "[-3, +3]",
    "m5_p5": "[-5, +5]",
}
GROUP_ORDER = ["escalation", "relief"]
GROUP_LABELS = {
    "escalation": "Escalation",
    "relief": "Relief",
}
GROUP_COLORS = {
    "escalation": "#B03A2E",
    "relief": "#1E8449",
}
WINDOW_COLORS = {
    "m1_p1": "#1F4E79",
    "m3_p3": "#2E86C1",
    "m5_p5": "#85C1E9",
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
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
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
            label=WINDOW_LABELS[window_label],
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(event_order)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("CAR formal simples do S&P 500")
    ax.set_ylabel("Evento")
    ax.set_title("CAR formal simples do S&P 500 por evento e janela")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FORMAL_CAR_BY_EVENT_OUTPUT, bbox_inches="tight")
    plt.close(fig)


def plot_formal_car_by_group(formal_car_by_group: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    x_positions = np.arange(len(WINDOW_ORDER))
    bar_width = 0.35

    for offset_index, event_group in enumerate(GROUP_ORDER):
        subset = (
            formal_car_by_group.loc[formal_car_by_group["event_group"] == event_group]
            .set_index("window_label")
            .reindex(WINDOW_ORDER)
        )
        ax.bar(
            x_positions + ((offset_index - 0.5) * bar_width),
            subset["mean_formal_car_sp500"],
            width=bar_width,
            color=GROUP_COLORS[event_group],
            label=GROUP_LABELS[event_group],
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([WINDOW_LABELS[window] for window in WINDOW_ORDER])
    ax.xaxis.grid(False)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Janela do evento")
    ax.set_ylabel("CAR formal simples medio")
    ax.set_title("CAR formal simples medio do S&P 500 por grupo de evento")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FORMAL_CAR_BY_GROUP_OUTPUT, bbox_inches="tight")
    plt.close(fig)


def plot_abnormal_returns_by_group(abnormal_returns: pd.DataFrame) -> None:
    grouped = (
        abnormal_returns.groupby(["window_label", "event_group", "relative_day"], dropna=False, observed=True)
        .agg(mean_abnormal_return=("abnormal_return", "mean"))
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300, sharey=True)

    for ax, window_label in zip(axes, WINDOW_ORDER, strict=True):
        subset = grouped.loc[grouped["window_label"] == window_label].copy()
        for event_group in GROUP_ORDER:
            group_subset = subset.loc[subset["event_group"] == event_group].sort_values("relative_day")
            ax.plot(
                group_subset["relative_day"],
                group_subset["mean_abnormal_return"],
                marker="o",
                linewidth=1.8,
                markersize=4,
                color=GROUP_COLORS[event_group],
                label=GROUP_LABELS[event_group],
            )

        ax.axvline(0, color="#5D6D7E", linestyle="--", linewidth=1)
        ax.axhline(0, color="#AEB6BF", linestyle=":", linewidth=1)
        ax.set_title(WINDOW_LABELS[window_label])
        ax.set_xlabel("Dia relativo ao evento")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    axes[0].set_ylabel("Retorno anormal medio")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
    fig.suptitle("Retorno anormal medio do S&P 500 por grupo de evento", y=1.04)
    fig.tight_layout()
    fig.savefig(ABNORMAL_RETURNS_BY_GROUP_OUTPUT, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    abnormal_returns, formal_car_by_event, formal_car_by_group = load_tables()
    plt.style.use("default")

    plot_formal_car_by_event(formal_car_by_event)
    plot_formal_car_by_group(formal_car_by_group)
    plot_abnormal_returns_by_group(abnormal_returns)

    print(f"Figura salva em: {FORMAL_CAR_BY_EVENT_OUTPUT}")
    print(f"Figura salva em: {FORMAL_CAR_BY_GROUP_OUTPUT}")
    print(f"Figura salva em: {ABNORMAL_RETURNS_BY_GROUP_OUTPUT}")


if __name__ == "__main__":
    main()
