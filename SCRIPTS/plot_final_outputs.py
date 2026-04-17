from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from academic_plot_style import (
    AXIS_FONTSIZE,
    RELIEF_COLOR,
    SUBTITLE_FONTSIZE,
    TITLE_FONTSIZE,
    add_reference_lines,
    apply_axis_style,
    configure_matplotlib,
    save_figure,
    shared_limits,
)


BASE_DIR = Path(__file__).resolve().parent.parent
CORE_RESULTS_FILE = BASE_DIR / "DATA" / "final_results_core.csv"
EXPANDED_RESULTS_FILE = BASE_DIR / "DATA" / "final_results_expanded.csv"
COMPARISON_FILE = BASE_DIR / "DATA" / "final_comparison_core_vs_expanded.csv"

CORE_FIGURE = BASE_DIR / "FIGURES" / "final_car_by_window_core.png"
EXPANDED_FIGURE = BASE_DIR / "FIGURES" / "final_car_by_window_expanded.png"
COMPARISON_FIGURE = BASE_DIR / "FIGURES" / "final_core_vs_expanded.png"

WINDOW_LABELS = ["m1_p1", "m3_p3", "m5_p5"]
WINDOW_TITLES = {
    "m1_p1": "[-1,+1]",
    "m3_p3": "[-3,+3]",
    "m5_p5": "[-5,+5]",
}
CORE_COLOR = "#1f4e79"
EXPANDED_COLOR = "#ff7f0e"


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    dataframe = pd.read_csv(path)
    dataframe["window_label"] = pd.Categorical(
        dataframe["window_label"],
        categories=WINDOW_LABELS,
        ordered=True,
    )
    return dataframe.sort_values("window_label").reset_index(drop=True)


def format_window_labels(dataframe: pd.DataFrame) -> list[str]:
    return [WINDOW_TITLES[label] for label in dataframe["window_label"].astype(str)]


def annotate_points(ax, x_positions: list[int], values: pd.Series) -> None:
    for x_position, value in zip(x_positions, values.tolist(), strict=True):
        y_offset = 0.0015 if value >= 0 else -0.0018
        va = "bottom" if value >= 0 else "top"
        ax.text(
            x_position,
            value + y_offset,
            f"{value * 100:.2f}%",
            ha="center",
            va=va,
            fontsize=10,
        )


def plot_single_series_figure(
    dataframe: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
    color: str,
    y_limits: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    x_positions = list(range(len(dataframe)))
    values = dataframe["mean_formal_car_sp500"]

    ax.plot(
        x_positions,
        values,
        color=color,
        marker="o",
        linewidth=2.6,
        markersize=8,
    )
    ax.fill_between(x_positions, values, 0, color=color, alpha=0.10)
    add_reference_lines(ax, horizontal_zero=True)
    ax.set_ylim(*y_limits)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(format_window_labels(dataframe))
    apply_axis_style(
        ax,
        xlabel="Janela de evento",
        ylabel="CAR formal medio do S&P 500",
        percent_y=True,
        grid_axis="y",
    )
    annotate_points(ax, x_positions, values)
    fig.suptitle(title, fontsize=TITLE_FONTSIZE, y=0.98)
    save_figure(fig, output_path, top_rect=0.92)


def plot_comparison_figure(
    comparison: pd.DataFrame,
    *,
    output_path: Path,
    y_limits: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    x_positions = list(range(len(comparison)))
    core_values = comparison["mean_formal_car_sp500_core"]
    expanded_values = comparison["mean_formal_car_sp500_expanded"]

    ax.plot(
        x_positions,
        core_values,
        color=CORE_COLOR,
        marker="o",
        linewidth=2.6,
        markersize=8,
        label="Core",
    )
    ax.plot(
        x_positions,
        expanded_values,
        color=EXPANDED_COLOR,
        marker="o",
        linewidth=2.6,
        markersize=8,
        label="Expanded com cobertura de mercado",
    )
    add_reference_lines(ax, horizontal_zero=True)
    ax.set_ylim(*y_limits)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(format_window_labels(comparison))
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
    fig.suptitle(
        "Comparacao do CAR formal medio por janela: core vs expanded",
        fontsize=TITLE_FONTSIZE,
        y=0.98,
    )
    save_figure(fig, output_path, top_rect=0.88)


def main() -> None:
    configure_matplotlib()
    core_results = load_results(CORE_RESULTS_FILE)
    expanded_results = load_results(EXPANDED_RESULTS_FILE)
    comparison = load_results(COMPARISON_FILE)

    combined_values = (
        core_results["mean_formal_car_sp500"].tolist()
        + expanded_results["mean_formal_car_sp500"].tolist()
    )
    y_limits = shared_limits(combined_values, pad_ratio=0.18)

    plot_single_series_figure(
        core_results,
        output_path=CORE_FIGURE,
        title="CAR formal medio por janela - amostra principal (core)",
        color=CORE_COLOR,
        y_limits=y_limits,
    )
    plot_single_series_figure(
        expanded_results,
        output_path=EXPANDED_FIGURE,
        title="CAR formal medio por janela - amostra expandida com cobertura de mercado",
        color=EXPANDED_COLOR,
        y_limits=y_limits,
    )
    plot_comparison_figure(
        comparison,
        output_path=COMPARISON_FIGURE,
        y_limits=y_limits,
    )

    print(f"Figura salva em: {CORE_FIGURE}")
    print(f"Figura salva em: {EXPANDED_FIGURE}")
    print(f"Figura salva em: {COMPARISON_FIGURE}")


if __name__ == "__main__":
    main()
