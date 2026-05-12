from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from academic_plot_style import (
    AXIS_FONTSIZE,
    TITLE_FONTSIZE,
    add_reference_lines,
    apply_axis_style,
    configure_matplotlib,
    save_figure,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DATA"
FIGURES_DIR = BASE_DIR / "FIGURES"

MODEL_RESULTS_FILE = DATA_DIR / "ml_model_results.csv"
CONFUSION_MATRICES_FILE = DATA_DIR / "ml_confusion_matrices.csv"
FEATURE_IMPORTANCE_FILE = DATA_DIR / "ml_feature_importance.csv"

MODEL_COMPARISON_FIGURE = FIGURES_DIR / "ml_model_comparison.png"
FEATURE_IMPORTANCE_FIGURE = FIGURES_DIR / "ml_feature_importance.png"
RANDOM_FOREST_CM_FIGURE = FIGURES_DIR / "ml_confusion_matrix_random_forest.png"

MODEL_ORDER = ["baseline_majority", "logistic_regression", "random_forest"]
METRICS = ["accuracy", "precision", "recall", "f1"]
MODEL_COLORS = {
    "baseline_majority": "#7f7f7f",
    "logistic_regression": "#1f4e79",
    "random_forest": "#ff7f0e",
}


def load_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    dataframe = pd.read_csv(path)
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes em {path.name}: {missing_columns}")
    return dataframe


def plot_model_comparison(results: pd.DataFrame) -> None:
    results = results.set_index("model").loc[MODEL_ORDER].reset_index()
    fig, axes = plt.subplots(1, len(METRICS), figsize=(18, 5), sharey=True)

    for ax, metric in zip(axes, METRICS, strict=True):
        colors = [MODEL_COLORS[model] for model in results["model"]]
        ax.bar(results["model_label"], results[metric], color=colors, alpha=0.9)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=25)
        apply_axis_style(
            ax,
            title=metric.upper(),
            ylabel="Valor" if metric == "accuracy" else None,
            percent_y=True,
            grid_axis="y",
        )

    fig.suptitle("Comparacao exploratoria dos modelos de ML", fontsize=TITLE_FONTSIZE, y=0.98)
    save_figure(fig, MODEL_COMPARISON_FIGURE, top_rect=0.86)


def plot_feature_importance(importance: pd.DataFrame) -> None:
    random_forest = (
        importance.loc[importance["model"] == "random_forest"]
        .sort_values("importance", ascending=False)
        .head(12)
        .sort_values("importance")
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.barh(random_forest["feature"], random_forest["importance"], color="#ff7f0e", alpha=0.9)
    apply_axis_style(
        ax,
        xlabel="Importancia relativa",
        ylabel="Feature",
        grid_axis="x",
    )
    fig.suptitle(
        "Principais features no Random Forest exploratorio",
        fontsize=TITLE_FONTSIZE,
        y=0.98,
    )
    save_figure(fig, FEATURE_IMPORTANCE_FIGURE, top_rect=0.90)


def plot_random_forest_confusion_matrix(confusion: pd.DataFrame) -> None:
    rf = confusion.loc[confusion["model"] == "random_forest"].copy()
    matrix = np.zeros((2, 2), dtype=int)
    for row in rf.itertuples(index=False):
        matrix[int(row.actual_label), int(row.predicted_label)] = int(row.count)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Nao negativo", "Negativo"])
    ax.set_yticklabels(["Nao negativo", "Negativo"])
    ax.set_xlabel("Classe prevista", fontsize=AXIS_FONTSIZE)
    ax.set_ylabel("Classe real", fontsize=AXIS_FONTSIZE)

    threshold = matrix.max() / 2 if matrix.max() else 0
    for actual in range(2):
        for predicted in range(2):
            color = "white" if matrix[actual, predicted] > threshold else "#222222"
            ax.text(
                predicted,
                actual,
                str(matrix[actual, predicted]),
                ha="center",
                va="center",
                fontsize=13,
                color=color,
            )

    add_reference_lines(ax)
    ax.grid(False)
    fig.suptitle(
        "Matriz de confusao - Random Forest exploratorio",
        fontsize=TITLE_FONTSIZE,
        y=0.98,
    )
    save_figure(fig, RANDOM_FOREST_CM_FIGURE, top_rect=0.90)


def main() -> None:
    configure_matplotlib()
    results = load_csv(MODEL_RESULTS_FILE, ["model", "model_label", *METRICS])
    confusion = load_csv(
        CONFUSION_MATRICES_FILE,
        ["model", "actual_label", "predicted_label", "count"],
    )
    importance = load_csv(FEATURE_IMPORTANCE_FILE, ["model", "feature", "importance"])

    plot_model_comparison(results)
    plot_feature_importance(importance)
    plot_random_forest_confusion_matrix(confusion)

    print(f"Figura salva em: {MODEL_COMPARISON_FIGURE}")
    print(f"Figura salva em: {FEATURE_IMPORTANCE_FIGURE}")
    print(f"Figura salva em: {RANDOM_FOREST_CM_FIGURE}")


if __name__ == "__main__":
    main()
