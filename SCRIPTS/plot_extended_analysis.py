from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
WINDOW_COMPARISON_BY_GROUP_FILE = BASE_DIR / "DATA" / "window_comparison_by_group.csv"
GROUP_SIGN_SUMMARY_FILE = BASE_DIR / "DATA" / "group_sign_summary.csv"
RETURN_VS_VOLATILITY_FILE = BASE_DIR / "DATA" / "return_vs_volatility.csv"

WINDOW_COMPARISON_OUTPUT = BASE_DIR / "FIGURES" / "window_comparison.png"
EVENT_SIGN_DISTRIBUTION_OUTPUT = BASE_DIR / "FIGURES" / "event_sign_distribution.png"
RETURN_VS_VOLATILITY_OUTPUT = BASE_DIR / "FIGURES" / "return_vs_volatility.png"

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
SIGN_COLORS = {
    "positive": "#1E8449",
    "neutral": "#B7950B",
    "negative": "#B03A2E",
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
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    x_positions = np.arange(len(WINDOW_ORDER))
    bar_width = 0.35

    for offset_index, event_group in enumerate(GROUP_ORDER):
        subset = window_comparison_by_group.loc[
            window_comparison_by_group["event_group"] == event_group
        ]
        values = [
            subset[f"mean_formal_car_{window}"].iloc[0] if f"mean_formal_car_{window}" in subset.columns else np.nan
            for window in WINDOW_ORDER
        ]
        ax.bar(
            x_positions + ((offset_index - 0.5) * bar_width),
            values,
            width=bar_width,
            color=GROUP_COLORS[event_group],
            label=GROUP_LABELS[event_group],
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([WINDOW_LABELS[window] for window in WINDOW_ORDER])
    ax.set_xlabel("Janela do evento")
    ax.set_ylabel("CAR formal simples medio")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Comparacao do CAR formal simples medio entre janelas")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(WINDOW_COMPARISON_OUTPUT, bbox_inches="tight")
    plt.close(fig)


def plot_event_sign_distribution(group_sign_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300, sharey=True)
    sign_order = ["negative", "neutral", "positive"]
    x_positions = np.arange(len(sign_order))
    bar_width = 0.35

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
                label=GROUP_LABELS[event_group],
            )

        ax.set_title(WINDOW_LABELS[window_label])
        ax.set_xticks(x_positions)
        ax.set_xticklabels(["Negativo", "Neutro", "Positivo"])
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    axes[0].set_ylabel("Proporcao de eventos")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
    fig.suptitle("Distribuicao do sinal do CAR formal simples por grupo de evento", y=1.04)
    fig.tight_layout()
    fig.savefig(EVENT_SIGN_DISTRIBUTION_OUTPUT, bbox_inches="tight")
    plt.close(fig)


def plot_return_vs_volatility(return_vs_volatility: pd.DataFrame) -> None:
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
                s=40,
                label=GROUP_LABELS[event_group],
            )

        ax.axvline(0, color="#5D6D7E", linestyle="--", linewidth=1)
        ax.set_title(WINDOW_LABELS[window_label])
        ax.set_xlabel("CAR formal simples do S&P 500")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(linestyle=":", alpha=0.35)

    axes[0].set_ylabel("Volatilidade do S&P 500")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
    fig.suptitle("Retorno acumulado formal simples versus volatilidade por janela", y=1.04)
    fig.tight_layout()
    fig.savefig(RETURN_VS_VOLATILITY_OUTPUT, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    window_comparison_by_group, group_sign_summary, return_vs_volatility = load_tables()
    plt.style.use("default")

    plot_window_comparison(window_comparison_by_group)
    plot_event_sign_distribution(group_sign_summary)
    plot_return_vs_volatility(return_vs_volatility)

    print(f"Figura salva em: {WINDOW_COMPARISON_OUTPUT}")
    print(f"Figura salva em: {EVENT_SIGN_DISTRIBUTION_OUTPUT}")
    print(f"Figura salva em: {RETURN_VS_VOLATILITY_OUTPUT}")


if __name__ == "__main__":
    main()
