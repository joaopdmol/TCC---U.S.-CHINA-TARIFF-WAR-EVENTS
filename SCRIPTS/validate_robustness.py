from pathlib import Path

import matplotlib.image as mpimg
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
ESTIMATION_FILE = BASE_DIR / "DATA" / "sensitivity_estimation_results.csv"
SIGN_FILE = BASE_DIR / "DATA" / "sensitivity_sign_results.csv"
SAMPLE_FILE = BASE_DIR / "DATA" / "sensitivity_sample_results.csv"
SUMMARY_FILE = BASE_DIR / "DATA" / "robustness_summary.csv"
TEXT_FILE = BASE_DIR / "ROBUSTNESS_SECTION.md"

ESTIMATION_FIGURE = BASE_DIR / "FIGURES" / "sensitivity_estimation_window.png"
SIGN_FIGURE = BASE_DIR / "FIGURES" / "sensitivity_sign_threshold.png"
SAMPLE_FIGURE = BASE_DIR / "FIGURES" / "sensitivity_sample_selection.png"

WINDOW_LABELS = {"m1_p1", "m3_p3", "m5_p5"}
ROBUSTNESS_LABELS = {"robust", "moderately sensitive", "sensitive"}


def ok(message: str) -> None:
    print(f"OK: {message}")


def require_columns(dataframe: pd.DataFrame, expected_columns: list[str], filename: str) -> None:
    missing_columns = [column for column in expected_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"ERRO: colunas ausentes em {filename}: {missing_columns}")


def validate_figure(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"ERRO: figura nao encontrada: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"ERRO: figura vazia: {path}")
    image = mpimg.imread(path)
    if image.shape[0] < 400 or image.shape[1] < 600:
        raise ValueError(f"ERRO: resolucao muito baixa em {path}")


def main() -> None:
    for path in [ESTIMATION_FILE, SIGN_FILE, SAMPLE_FILE, SUMMARY_FILE, TEXT_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"ERRO: arquivo nao encontrado: {path}")
    ok("arquivos de robustez encontrados")

    estimation = pd.read_csv(ESTIMATION_FILE)
    sign = pd.read_csv(SIGN_FILE)
    sample = pd.read_csv(SAMPLE_FILE)
    summary = pd.read_csv(SUMMARY_FILE)

    require_columns(
        estimation,
        [
            "estimation_config",
            "window_label",
            "mean_formal_car_sp500",
            "relative_change_pct_vs_baseline",
        ],
        ESTIMATION_FILE.name,
    )
    require_columns(
        sign,
        [
            "neutral_threshold",
            "event_group",
            "window_label",
            "sign_label",
            "event_share",
        ],
        SIGN_FILE.name,
    )
    require_columns(
        sample,
        [
            "sample_label",
            "window_label",
            "mean_formal_car_sp500",
            "mean_volatility_sp500",
            "positive_share",
        ],
        SAMPLE_FILE.name,
    )
    require_columns(
        summary,
        [
            "conclusion_id",
            "baseline_signal",
            "baseline_magnitude",
            "consistency_across_scenarios",
            "robustness_class",
        ],
        SUMMARY_FILE.name,
    )
    ok("colunas corretas")

    if set(estimation["window_label"].unique()) != WINDOW_LABELS:
        raise ValueError("ERRO: janelas esperadas nao encontradas em sensitivity_estimation_results.csv.")
    if set(sign["window_label"].unique()) != WINDOW_LABELS:
        raise ValueError("ERRO: janelas esperadas nao encontradas em sensitivity_sign_results.csv.")
    if set(sample["window_label"].unique()) != WINDOW_LABELS:
        raise ValueError("ERRO: janelas esperadas nao encontradas em sensitivity_sample_results.csv.")
    ok("janelas detectadas")

    if estimation["mean_formal_car_sp500"].isna().all():
        raise ValueError("ERRO: mean_formal_car_sp500 vazio em sensitivity_estimation_results.csv.")
    if sign["event_share"].isna().all():
        raise ValueError("ERRO: event_share vazio em sensitivity_sign_results.csv.")
    if sample["positive_share"].isna().all():
        raise ValueError("ERRO: positive_share vazio em sensitivity_sample_results.csv.")
    if summary["baseline_magnitude"].isna().all():
        raise ValueError("ERRO: baseline_magnitude vazio em robustness_summary.csv.")
    ok("valores principais preenchidos")

    if estimation.duplicated(subset=["estimation_config", "window_label"]).any():
        raise ValueError("ERRO: duplicidade em sensitivity_estimation_results.csv.")
    if sign.duplicated(subset=["neutral_threshold", "event_group", "window_label", "sign_label"]).any():
        raise ValueError("ERRO: duplicidade em sensitivity_sign_results.csv.")
    if sample.duplicated(subset=["sample_label", "window_label"]).any():
        raise ValueError("ERRO: duplicidade em sensitivity_sample_results.csv.")
    if summary.duplicated(subset=["conclusion_id"]).any():
        raise ValueError("ERRO: duplicidade em robustness_summary.csv.")
    ok("sem duplicidade")

    if not set(summary["robustness_class"].unique()).issubset(ROBUSTNESS_LABELS):
        raise ValueError("ERRO: robustness_class inesperado em robustness_summary.csv.")
    ok("classificacoes de robustez coerentes")

    text = TEXT_FILE.read_text(encoding="utf-8")
    for heading in [
        "## 8.1 Introducao a analise de robustez",
        "## 8.2 Sensibilidade a janela de estimacao",
        "## 8.3 Sensibilidade ao criterio de classificacao",
        "## 8.4 Sensibilidade a composicao da amostra",
        "## 8.5 Sintese geral de robustez",
        "## 8.6 Conclusao metodologica",
    ]:
        if heading not in text:
            raise ValueError(f"ERRO: heading ausente em ROBUSTNESS_SECTION.md: {heading}")
    ok("texto academico de robustez presente")

    for figure_path in [ESTIMATION_FIGURE, SIGN_FIGURE, SAMPLE_FIGURE]:
        validate_figure(figure_path)
    ok("figuras de robustez validas")

    print("\nVALIDACAO CONCLUIDA")


if __name__ == "__main__":
    main()
