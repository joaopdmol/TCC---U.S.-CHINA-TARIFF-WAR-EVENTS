from pathlib import Path

from PIL import Image
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DATA"
FIGURES_DIR = BASE_DIR / "FIGURES"
WINDOW_ORDER = {"m1_p1", "m3_p3", "m5_p5"}


def ok(message: str) -> None:
    print(f"OK: {message}")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"ERRO: arquivo nao encontrado: {path}")


def validate_window_tests() -> None:
    path = DATA_DIR / "window_stat_tests.csv"
    require_file(path)
    table = pd.read_csv(path)

    required_columns = [
        "scope_type",
        "scope_value",
        "comparison_label",
        "test_method",
        "n_pairs",
        "statistic",
        "p_value",
    ]
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(f"ERRO: colunas ausentes em window_stat_tests.csv: {missing_columns}")
    if table.duplicated(["scope_type", "scope_value", "comparison_label", "test_method"]).any():
        raise ValueError("ERRO: ha duplicidade em window_stat_tests.csv.")
    if table["p_value"].dropna().empty:
        raise ValueError("ERRO: p_value esta totalmente vazio em window_stat_tests.csv.")
    ok("testes entre janelas consistentes")


def validate_group_tests() -> None:
    path = DATA_DIR / "group_stat_tests.csv"
    require_file(path)
    table = pd.read_csv(path)

    required_columns = [
        "window_label",
        "group_a",
        "group_b",
        "test_method",
        "statistic",
        "p_value",
    ]
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(f"ERRO: colunas ausentes em group_stat_tests.csv: {missing_columns}")
    if set(table["window_label"].unique().tolist()) != WINDOW_ORDER:
        raise ValueError("ERRO: janelas esperadas ausentes em group_stat_tests.csv.")
    if table.duplicated(["window_label", "test_method"]).any():
        raise ValueError("ERRO: ha duplicidade em group_stat_tests.csv.")
    ok("testes entre grupos consistentes")


def validate_correlation_tests() -> None:
    path = DATA_DIR / "correlation_tests.csv"
    require_file(path)
    table = pd.read_csv(path)

    required_columns = ["scope_type", "scope_value", "test_method", "correlation", "p_value"]
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(f"ERRO: colunas ausentes em correlation_tests.csv: {missing_columns}")
    if table.duplicated(["scope_type", "scope_value", "test_method"]).any():
        raise ValueError("ERRO: ha duplicidade em correlation_tests.csv.")
    if table["correlation"].dropna().empty:
        raise ValueError("ERRO: correlation esta totalmente vazio em correlation_tests.csv.")
    ok("testes de correlacao consistentes")


def validate_figures() -> None:
    figure_paths = [
        FIGURES_DIR / "car_distribution_by_group.png",
        FIGURES_DIR / "scatter_return_vs_volatility.png",
    ]

    for figure_path in figure_paths:
        require_file(figure_path)
        if figure_path.stat().st_size == 0:
            raise ValueError(f"ERRO: figura vazia: {figure_path}")
        with Image.open(figure_path) as image:
            width, height = image.size
        if width < 1000 or height < 600:
            raise ValueError(
                f"ERRO: resolucao insuficiente em {figure_path.name}: {width}x{height}"
            )
    ok("figuras validas")


def main() -> None:
    validate_window_tests()
    validate_group_tests()
    validate_correlation_tests()
    validate_figures()
    print("\nVALIDACAO CONCLUIDA")


if __name__ == "__main__":
    main()
