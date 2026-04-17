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


def validate_window_comparison() -> None:
    by_event_path = DATA_DIR / "window_comparison_by_event.csv"
    by_group_path = DATA_DIR / "window_comparison_by_group.csv"
    require_file(by_event_path)
    require_file(by_group_path)

    by_event = pd.read_csv(by_event_path)
    by_group = pd.read_csv(by_group_path)

    required_by_event = [
        "event_id",
        "event_group",
        "formal_car_m1_p1",
        "formal_car_m3_p3",
        "formal_car_m5_p5",
        "formal_car_diff_m3_minus_m1",
        "formal_car_diff_m5_minus_m3",
        "formal_car_diff_m5_minus_m1",
    ]
    missing_by_event = [column for column in required_by_event if column not in by_event.columns]
    if missing_by_event:
        raise ValueError(f"ERRO: colunas ausentes em window_comparison_by_event.csv: {missing_by_event}")
    if by_event["event_id"].nunique() != 17:
        raise ValueError("ERRO: window_comparison_by_event.csv nao contem 17 eventos.")
    if by_event.duplicated(["event_id"]).any():
        raise ValueError("ERRO: ha duplicidade por event_id em window_comparison_by_event.csv.")

    required_by_group = [
        "event_group",
        "mean_formal_car_m1_p1",
        "mean_formal_car_m3_p3",
        "mean_formal_car_m5_p5",
    ]
    missing_by_group = [column for column in required_by_group if column not in by_group.columns]
    if missing_by_group:
        raise ValueError(f"ERRO: colunas ausentes em window_comparison_by_group.csv: {missing_by_group}")
    if by_group.duplicated(["event_group"]).any():
        raise ValueError("ERRO: ha duplicidade por event_group em window_comparison_by_group.csv.")
    ok("comparacao entre janelas consistente")


def validate_sign_analysis() -> None:
    by_event_path = DATA_DIR / "event_sign_analysis.csv"
    by_group_path = DATA_DIR / "group_sign_summary.csv"
    require_file(by_event_path)
    require_file(by_group_path)

    by_event = pd.read_csv(by_event_path)
    by_group = pd.read_csv(by_group_path)

    required_by_event = [
        "event_id",
        "window_label",
        "formal_car_sp500",
        "sign_label",
        "neutral_threshold",
    ]
    missing_by_event = [column for column in required_by_event if column not in by_event.columns]
    if missing_by_event:
        raise ValueError(f"ERRO: colunas ausentes em event_sign_analysis.csv: {missing_by_event}")
    if set(by_event["window_label"].unique().tolist()) != WINDOW_ORDER:
        raise ValueError("ERRO: janelas esperadas ausentes em event_sign_analysis.csv.")
    if by_event["sign_label"].dropna().empty:
        raise ValueError("ERRO: sign_label esta totalmente vazio.")

    required_by_group = [
        "event_group",
        "window_label",
        "sign_label",
        "event_count",
        "event_share",
    ]
    missing_by_group = [column for column in required_by_group if column not in by_group.columns]
    if missing_by_group:
        raise ValueError(f"ERRO: colunas ausentes em group_sign_summary.csv: {missing_by_group}")
    if by_group.duplicated(["event_group", "window_label", "sign_label"]).any():
        raise ValueError("ERRO: ha duplicidade em group_sign_summary.csv.")
    ok("analise de sinal consistente")


def validate_return_vs_volatility() -> None:
    path = DATA_DIR / "return_vs_volatility.csv"
    require_file(path)
    table = pd.read_csv(path)

    required_columns = [
        "event_id",
        "window_label",
        "formal_car_sp500",
        "volatility_sp500",
        "car_sign",
        "abs_formal_car_sp500",
    ]
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(f"ERRO: colunas ausentes em return_vs_volatility.csv: {missing_columns}")
    if table.duplicated(["event_id", "window_label"]).any():
        raise ValueError("ERRO: ha duplicidade em return_vs_volatility.csv.")
    if table[["formal_car_sp500", "volatility_sp500"]].isna().all().any():
        raise ValueError("ERRO: campos principais totalmente vazios em return_vs_volatility.csv.")
    ok("retorno versus volatilidade consistente")


def validate_figures() -> None:
    figure_paths = [
        FIGURES_DIR / "window_comparison.png",
        FIGURES_DIR / "event_sign_distribution.png",
        FIGURES_DIR / "return_vs_volatility.png",
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
    validate_window_comparison()
    validate_sign_analysis()
    validate_return_vs_volatility()
    validate_figures()
    print("\nVALIDACAO CONCLUIDA")


if __name__ == "__main__":
    main()
