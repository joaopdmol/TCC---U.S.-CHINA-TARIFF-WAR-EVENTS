from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from academic_plot_style import (
    ESCALATION_COLOR,
    RELIEF_COLOR,
    NEUTRAL_COLOR,
    TITLE_FONTSIZE,
    add_reference_lines,
    apply_axis_style,
    configure_matplotlib,
    place_figure_legend,
    save_figure,
    shared_limits,
)


BASE_DIR = Path(__file__).resolve().parent.parent
ESTIMATION_FILE = BASE_DIR / "DATA" / "sensitivity_estimation_results.csv"
SIGN_FILE = BASE_DIR / "DATA" / "sensitivity_sign_results.csv"
SAMPLE_FILE = BASE_DIR / "DATA" / "sensitivity_sample_results.csv"

ESTIMATION_FIGURE = BASE_DIR / "FIGURES" / "sensitivity_estimation_window.png"
SIGN_FIGURE = BASE_DIR / "FIGURES" / "sensitivity_sign_threshold.png"
SAMPLE_FIGURE = BASE_DIR / "FIGURES" / "sensitivity_sample_selection.png"

WINDOW_ORDER = ["m1_p1", "m3_p3", "m5_p5"]
WINDOW_LABELS = {
    "m1_p1": "[-1,+1]",
    "m3_p3": "[-3,+3]",
    "m5_p5": "[-5,+5]",
}
CONFIG_COLORS = {
    "baseline_m30_m6": "#1f4e79",
    "alt_m20_m6": "#ff7f0e",
    "alt_m40_m6": "#7f7f7f",
}
CONFIG_LABELS = {
    "baseline_m30_m6": "Estimacao [-30,-6]",
    "alt_m20_m6": "Estimacao [-20,-6]",
    "alt_m40_m6": "Estimacao [-40,-6]",
}
GROUP_COLORS = {
    "escalation": ESCALATION_COLOR,
    "relief": RELIEF_COLOR,
}
SAMPLE_COLORS = {
    "core_only": "#1f4e79",
    "core_plus_pandemic_covered": "#ff7f0e",
    "full_market_covered": "#7f7f7f",
}
SAMPLE_LINESTYLES = {
    "core_only": "-",
    "core_plus_pandemic_covered": "--",
    "full_market_covered": ":",
}
SAMPLE_LABELS = {
    "core_only": "Core only",
    "core_plus_pandemic_covered": "Core + pandemic covered",
    "full_market_covered": "Full market covered",
}
SAMPLE_PLOT_ORDER = [
    "full_market_covered",
    "core_plus_pandemic_covered",
    "core_only",
]
SIGN_COLORS = {
    "negative": ESCALATION_COLOR,
    "neutral": NEUTRAL_COLOR,
    "positive": RELIEF_COLOR,
}
SIGN_LABELS = {
    "negative": "Negativo",
    "neutral": "Neutro",
    "positive": "Positivo",
}


def load_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    dataframe = pd.read_csv(path)
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes em {path.name}: {missing_columns}")
    return dataframe


def plot_estimation_sensitivity(estimation: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    x_positions = list(range(len(WINDOW_ORDER)))
    y_limits = shared_limits(estimation["mean_formal_car_sp500"].dropna().tolist(), pad_ratio=0.18)

    for config_label in estimation["estimation_config"].unique():
        subset = estimation.loc[estimation["estimation_config"] == config_label].copy()
        subset["window_label"] = pd.Categorical(subset["window_label"], categories=WINDOW_ORDER, ordered=True)
        subset = subset.sort_values("window_label")
        ax.plot(
            x_positions,
            subset["mean_formal_car_sp500"],
            color=CONFIG_COLORS[config_label],
            marker="o",
            linewidth=2.6,
            markersize=8,
            label=CONFIG_LABELS[config_label],
        )

    add_reference_lines(ax, horizontal_zero=True)
    ax.set_ylim(*y_limits)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([WINDOW_LABELS[label] for label in WINDOW_ORDER])
    apply_axis_style(
        ax,
        xlabel="Janela de evento",
        ylabel="CAR formal medio do S&P 500",
        percent_y=True,
        grid_axis="y",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=False)
    fig.suptitle("Sensibilidade do CAR medio a janela de estimacao", fontsize=TITLE_FONTSIZE, y=0.98)
    save_figure(fig, ESTIMATION_FIGURE, top_rect=0.88)


def plot_sign_sensitivity(sign_results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True, sharey=True)
    thresholds = sorted(sign_results["neutral_threshold"].unique().tolist())

    for row_index, event_group in enumerate(["escalation", "relief"]):
        for col_index, window_label in enumerate(WINDOW_ORDER):
            ax = axes[row_index, col_index]
            subset = sign_results.loc[
                (sign_results["event_group"] == event_group)
                & (sign_results["window_label"] == window_label)
            ].copy()
            for sign_label in ["negative", "neutral", "positive"]:
                sign_subset = (
                    subset.loc[subset["sign_label"] == sign_label]
                    .set_index("neutral_threshold")
                    .reindex(thresholds, fill_value=0)
                    .reset_index()
                    .sort_values("neutral_threshold")
                )
                ax.plot(
                    thresholds,
                    sign_subset["event_share"],
                    color=SIGN_COLORS[sign_label],
                    marker="o",
                    linewidth=2.4,
                    markersize=6,
                    label=SIGN_LABELS[sign_label],
                )

            ax.set_title(f"{event_group.capitalize()} | {WINDOW_LABELS[window_label]}")
            ax.set_xticks(thresholds)
            ax.set_xticklabels([f"{value * 100:.2f}%" for value in thresholds])
            ax.set_ylim(0, 1)
            apply_axis_style(
                ax,
                xlabel="Limiar de neutralidade",
                percent_y=True,
                grid_axis="y",
            )

    apply_axis_style(axes[0, 0], ylabel="Proporcao de eventos", percent_y=True, grid_axis="y")
    apply_axis_style(axes[1, 0], ylabel="Proporcao de eventos", percent_y=True, grid_axis="y")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    place_figure_legend(fig, handles, labels, ncol=3, anchor_y=1.02)
    fig.suptitle("Sensibilidade da distribuicao de sinais ao limiar", fontsize=TITLE_FONTSIZE, y=0.98)
    save_figure(fig, SIGN_FIGURE, top_rect=0.90)


def plot_sample_sensitivity(sample_results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
    x_positions = list(range(len(WINDOW_ORDER)))

    metric_specs = [
        ("mean_formal_car_sp500", "CAR formal medio", True),
        ("mean_volatility_sp500", "Volatilidade media", True),
        ("positive_share", "Participacao de sinais positivos", True),
    ]

    for ax, (metric, ylabel, percent_y) in zip(axes, metric_specs, strict=True):
        metric_values = sample_results[metric].dropna().tolist()
        y_limits = shared_limits(metric_values, pad_ratio=0.18)
        if metric == "positive_share":
            y_limits = (0, 1)

        for sample_label in SAMPLE_PLOT_ORDER:
            subset = sample_results.loc[sample_results["sample_label"] == sample_label].copy()
            subset["window_label"] = pd.Categorical(subset["window_label"], categories=WINDOW_ORDER, ordered=True)
            subset = subset.sort_values("window_label")
            ax.plot(
                x_positions,
                subset[metric],
                color=SAMPLE_COLORS[sample_label],
                linestyle=SAMPLE_LINESTYLES[sample_label],
                marker="o",
                linewidth=2.6,
                markersize=8,
                label=SAMPLE_LABELS[sample_label],
            )

        if metric == "mean_formal_car_sp500":
            add_reference_lines(ax, horizontal_zero=True)
        ax.set_ylim(*y_limits)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([WINDOW_LABELS[label] for label in WINDOW_ORDER])
        apply_axis_style(
            ax,
            xlabel="Janela de evento",
            ylabel=ylabel,
            percent_y=percent_y,
            grid_axis="y",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    place_figure_legend(fig, handles, labels, ncol=3, anchor_y=1.03)
    fig.suptitle("Sensibilidade a composicao da amostra", fontsize=TITLE_FONTSIZE, y=0.98)
    save_figure(fig, SAMPLE_FIGURE, top_rect=0.86)


def main() -> None:
    configure_matplotlib()
    estimation = load_csv(
        ESTIMATION_FILE,
        ["estimation_config", "window_label", "mean_formal_car_sp500"],
    )
    sign_results = load_csv(
        SIGN_FILE,
        ["neutral_threshold", "event_group", "window_label", "sign_label", "event_share"],
    )
    sample_results = load_csv(
        SAMPLE_FILE,
        ["sample_label", "window_label", "mean_formal_car_sp500", "mean_volatility_sp500", "positive_share"],
    )

    plot_estimation_sensitivity(estimation)
    plot_sign_sensitivity(sign_results)
    plot_sample_sensitivity(sample_results)

    print(f"Figura salva em: {ESTIMATION_FIGURE}")
    print(f"Figura salva em: {SIGN_FIGURE}")
    print(f"Figura salva em: {SAMPLE_FIGURE}")


if __name__ == "__main__":
    main()
