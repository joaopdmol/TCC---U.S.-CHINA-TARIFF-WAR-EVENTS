from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
ABNORMAL_RETURNS_FILE = BASE_DIR / "DATA" / "abnormal_returns_long.csv"
FORMAL_CAR_BY_EVENT_FILE = BASE_DIR / "DATA" / "formal_car_by_event.csv"
FORMAL_CAR_BY_GROUP_FILE = BASE_DIR / "DATA" / "formal_car_by_group.csv"

WINDOW_ORDER = {"m1_p1", "m3_p3", "m5_p5"}


def ok(message: str) -> None:
    print(f"OK: {message}")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"ERRO: arquivo nao encontrado: {path}")


def validate_abnormal_returns() -> None:
    require_file(ABNORMAL_RETURNS_FILE)
    abnormal_returns = pd.read_csv(ABNORMAL_RETURNS_FILE)

    required_columns = [
        "event_id",
        "event_date",
        "anchor_date",
        "window_label",
        "relative_day",
        "market_date",
        "sp500_return",
        "expected_return",
        "abnormal_return",
        "event_group",
        "event_type",
        "description",
    ]
    missing_columns = [column for column in required_columns if column not in abnormal_returns.columns]
    if missing_columns:
        raise ValueError(f"ERRO: colunas ausentes em abnormal_returns_long.csv: {missing_columns}")
    ok("abnormal_returns_long.csv encontrado com colunas esperadas")

    if abnormal_returns["event_id"].nunique() != 17:
        raise ValueError("ERRO: abnormal_returns_long.csv nao contem 17 eventos.")
    ok("17 eventos detectados")

    if set(abnormal_returns["window_label"].unique().tolist()) != WINDOW_ORDER:
        raise ValueError("ERRO: janelas esperadas ausentes em abnormal_returns_long.csv.")
    ok("janelas detectadas")

    for column in ["event_date", "anchor_date", "market_date"]:
        abnormal_returns[column] = pd.to_datetime(abnormal_returns[column], errors="raise")
    ok("datas parseaveis")

    if abnormal_returns.duplicated(["event_id", "window_label", "relative_day"]).any():
        raise ValueError("ERRO: ha duplicidade em abnormal_returns_long.csv.")
    ok("sem duplicidade")

    if abnormal_returns["expected_return"].dropna().empty:
        raise ValueError("ERRO: expected_return esta totalmente vazio.")
    if abnormal_returns["abnormal_return"].dropna().empty:
        raise ValueError("ERRO: abnormal_return esta totalmente vazio.")
    ok("abnormal returns calculados")


def validate_formal_car_by_event() -> None:
    require_file(FORMAL_CAR_BY_EVENT_FILE)
    formal_car_by_event = pd.read_csv(FORMAL_CAR_BY_EVENT_FILE)

    required_columns = [
        "event_id",
        "event_group",
        "event_type",
        "window_label",
        "formal_car_sp500",
        "n_obs",
    ]
    missing_columns = [column for column in required_columns if column not in formal_car_by_event.columns]
    if missing_columns:
        raise ValueError(f"ERRO: colunas ausentes em formal_car_by_event.csv: {missing_columns}")

    if formal_car_by_event["event_id"].nunique() != 17:
        raise ValueError("ERRO: formal_car_by_event.csv nao contem 17 eventos.")
    if set(formal_car_by_event["window_label"].unique().tolist()) != WINDOW_ORDER:
        raise ValueError("ERRO: janelas esperadas ausentes em formal_car_by_event.csv.")
    if formal_car_by_event.duplicated(["event_id", "window_label"]).any():
        raise ValueError("ERRO: ha duplicidade em formal_car_by_event.csv.")
    if formal_car_by_event["formal_car_sp500"].dropna().empty:
        raise ValueError("ERRO: formal_car_sp500 esta totalmente vazio em formal_car_by_event.csv.")
    ok("formal_car_by_event.csv consistente")


def validate_formal_car_by_group() -> None:
    require_file(FORMAL_CAR_BY_GROUP_FILE)
    formal_car_by_group = pd.read_csv(FORMAL_CAR_BY_GROUP_FILE)

    required_columns = [
        "event_group",
        "window_label",
        "mean_formal_car_sp500",
        "median_formal_car_sp500",
        "std_formal_car_sp500",
        "n_events",
    ]
    missing_columns = [column for column in required_columns if column not in formal_car_by_group.columns]
    if missing_columns:
        raise ValueError(f"ERRO: colunas ausentes em formal_car_by_group.csv: {missing_columns}")

    if formal_car_by_group.duplicated(["event_group", "window_label"]).any():
        raise ValueError("ERRO: ha duplicidade em formal_car_by_group.csv.")
    if formal_car_by_group["mean_formal_car_sp500"].dropna().empty:
        raise ValueError("ERRO: mean_formal_car_sp500 esta totalmente vazio.")
    ok("formal CAR calculado")


def main() -> None:
    validate_abnormal_returns()
    validate_formal_car_by_event()
    validate_formal_car_by_group()
    print("\nVALIDACAO CONCLUIDA")


if __name__ == "__main__":
    main()
