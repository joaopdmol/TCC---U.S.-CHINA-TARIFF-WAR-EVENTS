from pathlib import Path

import matplotlib.image as mpimg
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DATA"
FIGURES_DIR = BASE_DIR / "FIGURES"

ML_DATASET_FILE = DATA_DIR / "ml_dataset.csv"
MODEL_RESULTS_FILE = DATA_DIR / "ml_model_results.csv"
CONFUSION_MATRICES_FILE = DATA_DIR / "ml_confusion_matrices.csv"
FEATURE_IMPORTANCE_FILE = DATA_DIR / "ml_feature_importance.csv"
ML_SECTION_FILE = BASE_DIR / "ML_SECTION.md"

FIGURE_FILES = [
    FIGURES_DIR / "ml_model_comparison.png",
    FIGURES_DIR / "ml_feature_importance.png",
    FIGURES_DIR / "ml_confusion_matrix_random_forest.png",
]

DATASET_REQUIRED_COLUMNS = [
    "event_id",
    "window_label",
    "target_negative_impact",
    "target_three_class",
    "feature_window_label",
    "feature_event_group",
    "feature_event_type",
    "feature_event_regime",
    "feature_volatility_sp500",
    "feature_n_obs",
    "feature_estimation_n_obs",
    "feature_estimation_sufficient_data",
    "feature_anchor_shift_calendar_days",
    "feature_window_size",
    "feature_event_year",
    "feature_event_month",
]
RESULTS_REQUIRED_COLUMNS = [
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "baseline_accuracy",
]
CONFUSION_REQUIRED_COLUMNS = [
    "model",
    "actual_label",
    "predicted_label",
    "count",
]
IMPORTANCE_REQUIRED_COLUMNS = [
    "model",
    "feature",
    "importance",
]
WINDOW_LABELS = {"m1_p1", "m3_p3", "m5_p5"}


def ok(message: str) -> None:
    print(f"OK: {message}")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"ERRO: arquivo nao encontrado: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"ERRO: arquivo vazio: {path}")


def require_columns(dataframe: pd.DataFrame, expected_columns: list[str], filename: str) -> None:
    missing_columns = [column for column in expected_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"ERRO: colunas ausentes em {filename}: {missing_columns}")


def validate_figure(path: Path) -> None:
    require_file(path)
    image = mpimg.imread(path)
    if image.shape[0] < 400 or image.shape[1] < 600:
        raise ValueError(f"ERRO: resolucao baixa em {path}")


def main() -> None:
    for path in [
        ML_DATASET_FILE,
        MODEL_RESULTS_FILE,
        CONFUSION_MATRICES_FILE,
        FEATURE_IMPORTANCE_FILE,
        ML_SECTION_FILE,
    ]:
        require_file(path)
    ok("arquivos de ML encontrados")

    dataset = pd.read_csv(ML_DATASET_FILE)
    results = pd.read_csv(MODEL_RESULTS_FILE)
    confusion = pd.read_csv(CONFUSION_MATRICES_FILE)
    importance = pd.read_csv(FEATURE_IMPORTANCE_FILE)

    require_columns(dataset, DATASET_REQUIRED_COLUMNS, "ml_dataset.csv")
    require_columns(results, RESULTS_REQUIRED_COLUMNS, "ml_model_results.csv")
    require_columns(confusion, CONFUSION_REQUIRED_COLUMNS, "ml_confusion_matrices.csv")
    require_columns(importance, IMPORTANCE_REQUIRED_COLUMNS, "ml_feature_importance.csv")
    ok("colunas esperadas presentes")

    if dataset["target_negative_impact"].isna().any():
        raise ValueError("ERRO: target_negative_impact contem valores ausentes.")
    if dataset["target_negative_impact"].nunique() != 2:
        raise ValueError("ERRO: target_negative_impact precisa conter duas classes.")
    ok("target binario valido")

    if set(dataset["window_label"].unique()) != WINDOW_LABELS:
        raise ValueError("ERRO: janelas esperadas nao encontradas em ml_dataset.csv.")
    if dataset["event_id"].nunique() < 17:
        raise ValueError("ERRO: dataset de ML deveria conter ao menos os eventos core.")
    ok("janelas e eventos detectados")

    feature_columns = [column for column in dataset.columns if column.startswith("feature_")]
    if any("formal_car_sp500" in column for column in feature_columns):
        raise ValueError("ERRO: formal_car_sp500 foi usado como feature.")
    if "formal_car_sp500" in dataset.columns:
        raise ValueError("ERRO: formal_car_sp500 nao deve estar no ml_dataset.csv final.")
    ok("sem vazamento direto de formal_car_sp500 nas features")

    if dataset.duplicated(subset=["event_id", "window_label"]).any():
        raise ValueError("ERRO: duplicidade em ml_dataset.csv.")
    ok("sem duplicidade no dataset")

    expected_models = {"baseline_majority", "logistic_regression", "random_forest"}
    if set(results["model"].unique()) != expected_models:
        raise ValueError("ERRO: modelos esperados nao encontrados em ml_model_results.csv.")
    metric_columns = ["accuracy", "precision", "recall", "f1"]
    if results[metric_columns].isna().any().any():
        raise ValueError("ERRO: metricas de ML contem valores ausentes.")
    if not results[metric_columns].apply(lambda series: series.between(0, 1).all()).all():
        raise ValueError("ERRO: metricas fora do intervalo [0, 1].")
    ok("metricas calculadas")

    if confusion["count"].isna().any() or confusion["count"].sum() <= 0:
        raise ValueError("ERRO: matriz de confusao invalida.")
    if importance["importance"].isna().all():
        raise ValueError("ERRO: importancia de features totalmente vazia.")
    ok("matrizes e importancias preenchidas")

    for figure_path in FIGURE_FILES:
        validate_figure(figure_path)
    ok("figuras de ML validas")

    section_text = ML_SECTION_FILE.read_text(encoding="utf-8")
    required_headings = [
        "## 1. Objetivo da camada exploratoria de ML",
        "## 2. Construcao do dataset",
        "## 3. Modelos utilizados",
        "## 4. Metricas de avaliacao",
        "## 5. Resultados",
        "## 6. Limitacoes",
        "## 7. Por que ML e complementar ao event study",
    ]
    for heading in required_headings:
        if heading not in section_text:
            raise ValueError(f"ERRO: secao ausente em ML_SECTION.md: {heading}")
    ok("texto academico de ML presente")

    print("\nVALIDACAO CONCLUIDA")


if __name__ == "__main__":
    main()
