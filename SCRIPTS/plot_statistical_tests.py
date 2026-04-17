from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
FORMAL_CAR_BY_EVENT_FILE = BASE_DIR / "DATA" / "formal_car_by_event.csv"
RETURN_VS_VOLATILITY_FILE = BASE_DIR / "DATA" / "return_vs_volatility.csv"

CAR_DISTRIBUTION_OUTPUT = BASE_DIR / "FIGURES" / "car_distribution_by_group.png"
SCATTER_OUTPUT = BASE_DIR / "FIGURES" / "scatter_return_vs_volatility.png"

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
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300, sharey=True)

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
        )
        for patch, event_group in zip(box["boxes"], GROUP_ORDER, strict=True):
            patch.set_facecolor(GROUP_COLORS[event_group])
            patch.set_alpha(0.7)

        ax.axhline(0, color="#5D6D7E", linestyle="--", linewidth=1)
        ax.set_title(WINDOW_LABELS[window_label])
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    axes[0].set_ylabel("CAR formal simples do S&P 500")
    fig.suptitle("Distribuicao do CAR formal simples por grupo de evento", y=1.03)
    fig.tight_layout()
    fig.savefig(CAR_DISTRIBUTION_OUTPUT, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_return_vs_volatility(return_vs_volatility: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300, sharey=True)

    for ax, window_label in zip(axes, WINDOW_ORDER, strict=True):
        subset = return_vs_volatility.loc[return_vs_volatility["window_label"] == window_label]
        for event_group in GROUP_ORDER:
            group_subset = subset.loc[subset["event_group"] == event_group]
            ax.scatter(
                group_subset["formal_car_sp500"],
                group_subset["volatility_sp500"],
                color=GROUP_COLORS[event_group],
                alpha=0.8,
                s=42,
                label=GROUP_LABELS[event_group],
            )

        ax.axvline(0, color="#5D6D7E", linestyle="--", linewidth=1)
        ax.set_title(WINDOW_LABELS[window_label])
        ax.set_xlabel("CAR formal simples")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(linestyle=":", alpha=0.35)

    axes[0].set_ylabel("Volatilidade do S&P 500")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
    fig.suptitle("CAR formal simples versus volatilidade por janela", y=1.04)
    fig.tight_layout()
    fig.savefig(SCATTER_OUTPUT, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    formal_car_by_event, return_vs_volatility = load_inputs()
    plt.style.use("default")

    plot_car_distribution_by_group(formal_car_by_event)
    plot_scatter_return_vs_volatility(return_vs_volatility)

    print(f"Figura salva em: {CAR_DISTRIBUTION_OUTPUT}")
    print(f"Figura salva em: {SCATTER_OUTPUT}")


if __name__ == "__main__":
    main()
