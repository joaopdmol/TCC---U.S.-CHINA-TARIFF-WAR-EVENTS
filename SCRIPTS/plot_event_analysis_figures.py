from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
CAR_BY_EVENT_FILE = BASE_DIR / "DATA" / "car_by_event.csv"
SUMMARY_BY_GROUP_FILE = BASE_DIR / "DATA" / "event_summary_by_group.csv"
VOLATILITY_BY_GROUP_FILE = BASE_DIR / "DATA" / "volatility_by_group.csv"

CAR_OUTPUT_FILE = BASE_DIR / "FIGURES" / "car_by_event.png"
RETURNS_OUTPUT_FILE = BASE_DIR / "FIGURES" / "returns_by_group.png"
VOLATILITY_OUTPUT_FILE = BASE_DIR / "FIGURES" / "volatility_by_group.png"

WINDOW_ORDER = ["m1_p1", "m3_p3", "m5_p5"]
WINDOW_LABELS = {
    "m1_p1": "[-1, +1]",
    "m3_p3": "[-3, +3]",
    "m5_p5": "[-5, +5]",
}
PERIOD_ORDER = ["before", "event_day", "after"]
PERIOD_LABELS = {
    "before": "Antes",
    "event_day": "Dia do evento",
    "after": "Depois",
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
    ensure_file(CAR_BY_EVENT_FILE, "CAR simples por evento")
    ensure_file(SUMMARY_BY_GROUP_FILE, "resumo por grupo")
    ensure_file(VOLATILITY_BY_GROUP_FILE, "volatilidade por grupo")

    car_by_event = pd.read_csv(CAR_BY_EVENT_FILE)
    summary_by_group = pd.read_csv(SUMMARY_BY_GROUP_FILE)
    volatility_by_group = pd.read_csv(VOLATILITY_BY_GROUP_FILE)

    return car_by_event, summary_by_group, volatility_by_group


def plot_car_by_event(car_by_event: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
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
            label=WINDOW_LABELS[window_label],
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(event_order)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Retorno acumulado simples do S&P 500")
    ax.set_ylabel("Evento")
    ax.set_title("Retorno acumulado simples do S&P 500 por evento e janela")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    CAR_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CAR_OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)


def plot_returns_by_group(summary_by_group: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300, sharey=True)
    x_positions = np.arange(len(PERIOD_ORDER))
    bar_width = 0.35

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
                label=GROUP_LABELS[event_group],
            )

        ax.set_title(WINDOW_LABELS[window_label])
        ax.set_xticks(x_positions)
        ax.set_xticklabels([PERIOD_LABELS[period] for period in PERIOD_ORDER], rotation=10)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    axes[0].set_ylabel("Retorno medio diario do S&P 500")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
    fig.suptitle("Retorno medio diario do S&P 500 por grupo de evento e fase da janela", y=1.04)
    fig.tight_layout()
    fig.savefig(RETURNS_OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)


def plot_volatility_by_group(volatility_by_group: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    x_positions = np.arange(len(WINDOW_ORDER))
    bar_width = 0.35

    for offset_index, event_group in enumerate(GROUP_ORDER):
        subset = (
            volatility_by_group.loc[volatility_by_group["event_group"] == event_group]
            .set_index("window_label")
            .reindex(WINDOW_ORDER)
        )
        ax.bar(
            x_positions + ((offset_index - 0.5) * bar_width),
            subset["volatility_sp500_mean"],
            width=bar_width,
            color=GROUP_COLORS[event_group],
            label=GROUP_LABELS[event_group],
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([WINDOW_LABELS[window] for window in WINDOW_ORDER])
    ax.set_xlabel("Janela do evento")
    ax.set_ylabel("Volatilidade media do S&P 500")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Volatilidade media do S&P 500 por grupo de evento e janela")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(VOLATILITY_OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    car_by_event, summary_by_group, volatility_by_group = load_tables()
    plt.style.use("default")

    plot_car_by_event(car_by_event)
    plot_returns_by_group(summary_by_group)
    plot_volatility_by_group(volatility_by_group)

    print(f"Figura salva em: {CAR_OUTPUT_FILE}")
    print(f"Figura salva em: {RETURNS_OUTPUT_FILE}")
    print(f"Figura salva em: {VOLATILITY_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
