from pathlib import Path

import matplotlib.image as mpimg
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
CORE_RESULTS_FILE = BASE_DIR / "DATA" / "final_results_core.csv"
EXPANDED_RESULTS_FILE = BASE_DIR / "DATA" / "final_results_expanded.csv"
COMPARISON_FILE = BASE_DIR / "DATA" / "final_comparison_core_vs_expanded.csv"
WINDOW_ANALYSIS_FILE = BASE_DIR / "DATA" / "final_window_analysis.csv"
RESULTS_SECTION_FILE = BASE_DIR / "RESULTS_SECTION.md"

CORE_FIGURE = BASE_DIR / "FIGURES" / "final_car_by_window_core.png"
EXPANDED_FIGURE = BASE_DIR / "FIGURES" / "final_car_by_window_expanded.png"
COMPARISON_FIGURE = BASE_DIR / "FIGURES" / "final_core_vs_expanded.png"

WINDOW_LABELS = {"m1_p1", "m3_p3", "m5_p5"}
CORE_REQUIRED_COLUMNS = [
    "sample_label",
    "window_label",
    "mean_formal_car_sp500",
    "median_formal_car_sp500",
    "std_formal_car_sp500",
    "mean_volatility_sp500",
    "positive_share",
    "negative_share",
    "n_events",
]
COMPARISON_REQUIRED_COLUMNS = [
    "window_label",
    "mean_formal_car_sp500_core",
    "mean_formal_car_sp500_expanded",
    "diff_mean_formal_car_sp500_expanded_minus_core",
    "mean_volatility_sp500_core",
    "mean_volatility_sp500_expanded",
]
WINDOW_ANALYSIS_REQUIRED_COLUMNS = [
    "sample_label",
    "mean_formal_car_m1_p1",
    "mean_formal_car_m3_p3",
    "mean_formal_car_m5_p5",
    "diff_m3_minus_m1",
    "diff_m5_minus_m3",
    "tendency_label",
]
RESULTS_HEADINGS = [
    "## 6.1 Introducao da secao de resultados",
    "## 6.2 Resultados do event study (core)",
    "## 6.3 Robustez com amostra expandida",
    "## 6.4 Comparacao core vs expanded",
    "## 6.5 Relacao retorno vs volatilidade",
    "## 6.6 Interpretacao dos testes estatisticos",
    "## 6.7 Limitacoes metodologicas",
    "## 6.8 Sintese dos achados",
]


def ok(message: str) -> None:
    print(f"OK: {message}")


def require_columns(dataframe: pd.DataFrame, expected_columns: list[str], filename: str) -> None:
    missing_columns = [column for column in expected_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"ERRO: colunas ausentes em {filename}: {missing_columns}")


def validate_figure(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"ERRO: figura nao encontrada: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"ERRO: figura vazia: {path}")
    image = mpimg.imread(path)
    if image.shape[0] < 400 or image.shape[1] < 600:
        raise ValueError(f"ERRO: resolucao muito baixa em {path}")


def main() -> None:
    for path in [
        CORE_RESULTS_FILE,
        EXPANDED_RESULTS_FILE,
        COMPARISON_FILE,
        WINDOW_ANALYSIS_FILE,
        RESULTS_SECTION_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"ERRO: arquivo nao encontrado: {path}")
    ok("arquivos finais encontrados")

    core = pd.read_csv(CORE_RESULTS_FILE)
    expanded = pd.read_csv(EXPANDED_RESULTS_FILE)
    comparison = pd.read_csv(COMPARISON_FILE)
    window_analysis = pd.read_csv(WINDOW_ANALYSIS_FILE)

    require_columns(core, CORE_REQUIRED_COLUMNS, "final_results_core.csv")
    require_columns(expanded, CORE_REQUIRED_COLUMNS, "final_results_expanded.csv")
    require_columns(comparison, COMPARISON_REQUIRED_COLUMNS, "final_comparison_core_vs_expanded.csv")
    require_columns(window_analysis, WINDOW_ANALYSIS_REQUIRED_COLUMNS, "final_window_analysis.csv")
    ok("colunas corretas")

    if set(core["window_label"].unique()) != WINDOW_LABELS:
        raise ValueError("ERRO: janelas esperadas nao encontradas em final_results_core.csv.")
    if set(expanded["window_label"].unique()) != WINDOW_LABELS:
        raise ValueError("ERRO: janelas esperadas nao encontradas em final_results_expanded.csv.")
    if set(comparison["window_label"].unique()) != WINDOW_LABELS:
        raise ValueError("ERRO: janelas esperadas nao encontradas em final_comparison_core_vs_expanded.csv.")
    ok("janelas detectadas")

    if core["n_events"].nunique() != 1 or int(core["n_events"].iloc[0]) != 17:
        raise ValueError("ERRO: amostra core deveria conter 17 eventos em cada janela.")
    if expanded["n_events"].nunique() != 1 or int(expanded["n_events"].iloc[0]) != 24:
        raise ValueError("ERRO: amostra expandida deveria conter 24 eventos cobertos em cada janela.")
    ok("contagens de eventos coerentes")

    for dataframe, filename in [
        (core, "final_results_core.csv"),
        (expanded, "final_results_expanded.csv"),
        (comparison, "final_comparison_core_vs_expanded.csv"),
        (window_analysis, "final_window_analysis.csv"),
    ]:
        if dataframe.isna().all().any():
            raise ValueError(f"ERRO: existe coluna totalmente vazia em {filename}.")
    ok("valores principais preenchidos")

    if comparison.duplicated(subset=["window_label"]).any():
        raise ValueError("ERRO: duplicidade em final_comparison_core_vs_expanded.csv.")
    if window_analysis.duplicated(subset=["sample_label"]).any():
        raise ValueError("ERRO: duplicidade em final_window_analysis.csv.")
    ok("sem duplicidade")

    if not set(window_analysis["sample_label"].unique()) == {"core", "expanded_market_covered"}:
        raise ValueError("ERRO: sample_label inesperado em final_window_analysis.csv.")
    ok("comparacao core e expanded coerente")

    results_text = RESULTS_SECTION_FILE.read_text(encoding="utf-8")
    for heading in RESULTS_HEADINGS:
        if heading not in results_text:
            raise ValueError(f"ERRO: heading ausente em RESULTS_SECTION.md: {heading}")
    ok("texto academico com secoes obrigatorias")

    for figure_path in [CORE_FIGURE, EXPANDED_FIGURE, COMPARISON_FIGURE]:
        validate_figure(figure_path)
    ok("figuras existem e sao validas")

    print("\nVALIDACAO CONCLUIDA")


if __name__ == "__main__":
    main()
