from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DATA"
FIGURES_DIR = BASE_DIR / "FIGURES"

WINDOW_ORDER = ["m1_p1", "m3_p3", "m5_p5"]
PERIOD_ORDER = ["before", "event_day", "after"]
GROUP_ORDER = ["escalation", "relief"]


def ok(message: str) -> None:
    print(f"OK: {message}")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"ERRO: arquivo nao encontrado: {path}")


def validate_summary_by_event() -> None:
    path = DATA_DIR / "event_summary_by_event.csv"
    require_file(path)
    table = pd.read_csv(path)

    required_columns = [
        "event_id",
        "event_group",
        "event_type",
        "description",
        "window_label",
        "period",
        "sp500_return_mean",
        "nasdaq_return_mean",
        "shanghai_return_mean",
    ]
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(f"ERRO: colunas ausentes em event_summary_by_event.csv: {missing_columns}")
    ok("event_summary_by_event.csv encontrado com colunas esperadas")

    if table["event_id"].nunique() != 17:
        raise ValueError("ERRO: event_summary_by_event.csv nao contem 17 eventos.")
    ok("17 eventos detectados no resumo por evento")

    if set(table["window_label"].unique().tolist()) != set(WINDOW_ORDER):
        raise ValueError("ERRO: janelas esperadas ausentes no resumo por evento.")
    ok("janelas detectadas no resumo por evento")

    if set(table["period"].unique().tolist()) != set(PERIOD_ORDER):
        raise ValueError("ERRO: periodos esperados ausentes no resumo por evento.")
    ok("periodos before/event_day/after detectados")

    if table.duplicated(["event_id", "window_label", "period"]).any():
        raise ValueError("ERRO: ha duplicidade no resumo por evento.")
    ok("sem duplicidade no resumo por evento")

    if table[["sp500_return_mean", "nasdaq_return_mean", "shanghai_return_mean"]].isna().all().any():
        raise ValueError("ERRO: existe metrica principal totalmente vazia no resumo por evento.")


def validate_summary_by_group() -> None:
    path = DATA_DIR / "event_summary_by_group.csv"
    require_file(path)
    table = pd.read_csv(path)

    required_columns = [
        "event_group",
        "window_label",
        "period",
        "event_count",
        "sp500_return_mean",
    ]
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(f"ERRO: colunas ausentes em event_summary_by_group.csv: {missing_columns}")
    ok("event_summary_by_group.csv encontrado com colunas esperadas")

    if set(table["event_group"].unique().tolist()) != set(GROUP_ORDER):
        raise ValueError("ERRO: grupos esperados ausentes no resumo por grupo.")
    if set(table["window_label"].unique().tolist()) != set(WINDOW_ORDER):
        raise ValueError("ERRO: janelas esperadas ausentes no resumo por grupo.")
    if set(table["period"].unique().tolist()) != set(PERIOD_ORDER):
        raise ValueError("ERRO: periodos esperados ausentes no resumo por grupo.")

    if table.duplicated(["event_group", "window_label", "period"]).any():
        raise ValueError("ERRO: ha duplicidade no resumo por grupo.")
    ok("sem duplicidade no resumo por grupo")


def validate_summary_before_after() -> None:
    path = DATA_DIR / "event_summary_before_after.csv"
    require_file(path)
    table = pd.read_csv(path)

    required_columns = ["window_label", "period", "event_count", "sp500_return_mean"]
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(f"ERRO: colunas ausentes em event_summary_before_after.csv: {missing_columns}")

    if table.duplicated(["window_label", "period"]).any():
        raise ValueError("ERRO: ha duplicidade no resumo before/after.")
    ok("event_summary_before_after.csv consistente")


def validate_car_tables() -> None:
    by_event_path = DATA_DIR / "car_by_event.csv"
    by_group_path = DATA_DIR / "car_by_group.csv"
    require_file(by_event_path)
    require_file(by_group_path)

    by_event = pd.read_csv(by_event_path)
    by_group = pd.read_csv(by_group_path)

    required_by_event = [
        "event_id",
        "event_group",
        "event_type",
        "window_label",
        "car_simple_sp500",
        "n_obs_sp500_return",
    ]
    missing_by_event = [column for column in required_by_event if column not in by_event.columns]
    if missing_by_event:
        raise ValueError(f"ERRO: colunas ausentes em car_by_event.csv: {missing_by_event}")

    if by_event["event_id"].nunique() != 17:
        raise ValueError("ERRO: car_by_event.csv nao contem 17 eventos.")
    if set(by_event["window_label"].unique().tolist()) != set(WINDOW_ORDER):
        raise ValueError("ERRO: janelas esperadas ausentes em car_by_event.csv.")
    if by_event.duplicated(["event_id", "window_label"]).any():
        raise ValueError("ERRO: ha duplicidade em car_by_event.csv.")
    if by_event["car_simple_sp500"].dropna().empty:
        raise ValueError("ERRO: car_simple_sp500 esta totalmente vazio.")
    ok("car_by_event.csv consistente")

    required_by_group = [
        "event_group",
        "window_label",
        "car_simple_sp500_mean",
        "n_events",
    ]
    missing_by_group = [column for column in required_by_group if column not in by_group.columns]
    if missing_by_group:
        raise ValueError(f"ERRO: colunas ausentes em car_by_group.csv: {missing_by_group}")

    if by_group.duplicated(["event_group", "window_label"]).any():
        raise ValueError("ERRO: ha duplicidade em car_by_group.csv.")
    ok("car_by_group.csv consistente")


def validate_volatility_tables() -> None:
    by_event_path = DATA_DIR / "volatility_by_event.csv"
    by_group_path = DATA_DIR / "volatility_by_group.csv"
    require_file(by_event_path)
    require_file(by_group_path)

    by_event = pd.read_csv(by_event_path)
    by_group = pd.read_csv(by_group_path)

    required_by_event = [
        "event_id",
        "event_group",
        "event_type",
        "window_label",
        "volatility_sp500",
        "volatility_nasdaq",
        "volatility_shanghai",
    ]
    missing_by_event = [column for column in required_by_event if column not in by_event.columns]
    if missing_by_event:
        raise ValueError(f"ERRO: colunas ausentes em volatility_by_event.csv: {missing_by_event}")

    if by_event["event_id"].nunique() != 17:
        raise ValueError("ERRO: volatility_by_event.csv nao contem 17 eventos.")
    if set(by_event["window_label"].unique().tolist()) != set(WINDOW_ORDER):
        raise ValueError("ERRO: janelas esperadas ausentes em volatility_by_event.csv.")
    if by_event.duplicated(["event_id", "window_label"]).any():
        raise ValueError("ERRO: ha duplicidade em volatility_by_event.csv.")
    if by_event[["volatility_sp500", "volatility_nasdaq", "volatility_shanghai"]].isna().all().any():
        raise ValueError("ERRO: existe coluna principal totalmente vazia em volatility_by_event.csv.")
    ok("volatility_by_event.csv consistente")

    required_by_group = [
        "event_group",
        "window_label",
        "volatility_sp500_mean",
        "n_events",
    ]
    missing_by_group = [column for column in required_by_group if column not in by_group.columns]
    if missing_by_group:
        raise ValueError(f"ERRO: colunas ausentes em volatility_by_group.csv: {missing_by_group}")

    if by_group.duplicated(["event_group", "window_label"]).any():
        raise ValueError("ERRO: ha duplicidade em volatility_by_group.csv.")
    ok("volatility_by_group.csv consistente")


def validate_figures() -> None:
    figure_names = [
        "car_by_event.png",
        "returns_by_group.png",
        "volatility_by_group.png",
    ]
    for figure_name in figure_names:
        figure_path = FIGURES_DIR / figure_name
        require_file(figure_path)
        if figure_path.stat().st_size == 0:
            raise ValueError(f"ERRO: figura vazia: {figure_path}")
    ok("figuras principais encontradas")


def main() -> None:
    validate_summary_by_event()
    validate_summary_by_group()
    validate_summary_before_after()
    validate_car_tables()
    validate_volatility_tables()
    validate_figures()
    print("\nVALIDACAO CONCLUIDA")


if __name__ == "__main__":
    main()
