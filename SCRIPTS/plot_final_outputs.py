from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
CORE_RESULTS_FILE = BASE_DIR / "DATA" / "final_results_core.csv"
EXPANDED_RESULTS_FILE = BASE_DIR / "DATA" / "final_results_expanded.csv"
COMPARISON_FILE = BASE_DIR / "DATA" / "final_comparison_core_vs_expanded.csv"

CORE_FIGURE = BASE_DIR / "FIGURES" / "final_car_by_window_core.png"
EXPANDED_FIGURE = BASE_DIR / "FIGURES" / "final_car_by_window_expanded.png"
COMPARISON_FIGURE = BASE_DIR / "FIGURES" / "final_core_vs_expanded.png"

WINDOW_LABELS = ["m1_p1", "m3_p3", "m5_p5"]
WINDOW_TITLES = {
    "m1_p1": "[-1, +1]",
    "m3_p3": "[-3, +3]",
    "m5_p5": "[-5, +5]",
}


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


def format_xlabels(dataframe: pd.DataFrame) -> list[str]:
    return [WINDOW_TITLES[label] for label in dataframe["window_label"].astype(str)]


def save_bar_figure(
    dataframe: pd.DataFrame,
    output_path: Path,
    title: str,
    color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    xlabels = format_xlabels(dataframe)
    values = dataframe["mean_formal_car_sp500"] * 100

    ax.bar(xlabels, values, color=color, edgecolor="black", alpha=0.85)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Janela de evento")
    ax.set_ylabel("CAR formal medio do S&P 500 (%)")

    for index, value in enumerate(values):
        ax.text(index, value + (0.10 if value >= 0 else -0.20), f"{value:.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_comparison_figure(dataframe: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = range(len(dataframe))
    width = 0.35

    core_values = dataframe["mean_formal_car_sp500_core"] * 100
    expanded_values = dataframe["mean_formal_car_sp500_expanded"] * 100
    xlabels = [WINDOW_TITLES[label] for label in dataframe["window_label"].astype(str)]

    ax.bar([position - width / 2 for position in x], core_values, width=width, label="Core", color="#1f77b4", edgecolor="black")
    ax.bar([position + width / 2 for position in x], expanded_values, width=width, label="Expanded (cobertura de mercado)", color="#ff7f0e", edgecolor="black")

    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Comparacao do CAR formal medio por janela: core vs expanded")
    ax.set_xlabel("Janela de evento")
    ax.set_ylabel("CAR formal medio do S&P 500 (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(xlabels)
    ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    core_results = load_results(CORE_RESULTS_FILE)
    expanded_results = load_results(EXPANDED_RESULTS_FILE)
    comparison = load_results(COMPARISON_FILE)

    save_bar_figure(
        core_results,
        CORE_FIGURE,
        "CAR formal medio por janela - amostra principal (core)",
        "#1f77b4",
    )
    save_bar_figure(
        expanded_results,
        EXPANDED_FIGURE,
        "CAR formal medio por janela - amostra expandida com cobertura de mercado",
        "#ff7f0e",
    )
    save_comparison_figure(comparison, COMPARISON_FIGURE)

    print(f"Figura salva em: {CORE_FIGURE}")
    print(f"Figura salva em: {EXPANDED_FIGURE}")
    print(f"Figura salva em: {COMPARISON_FIGURE}")


if __name__ == "__main__":
    main()
