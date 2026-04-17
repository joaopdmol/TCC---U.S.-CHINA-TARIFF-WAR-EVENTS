from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
EVENTS_MASTER_FILE = BASE_DIR / "DATA" / "events_master.csv"
WINDOWS_FILE = BASE_DIR / "DATA" / "event_windows_master.csv"
COVERAGE_FILE = BASE_DIR / "DATA" / "event_windows_master_coverage.csv"
ABNORMAL_FILE = BASE_DIR / "DATA" / "abnormal_returns_master.csv"
FORMAL_CAR_FILE = BASE_DIR / "DATA" / "formal_car_master.csv"
COMPARISON_FILE = BASE_DIR / "DATA" / "core_vs_full_comparison.csv"
MARKET_FEATURES_FILE = BASE_DIR / "DATA" / "market_features.csv"

WINDOW_LABELS = {"m1_p1", "m3_p3", "m5_p5"}
EXPECTED_EVENT_COLUMNS = [
    "event_id",
    "date",
    "event_type",
    "event_group",
    "description",
    "event_regime",
    "is_core_sample",
    "is_pandemic_period",
    "is_post_pandemic",
    "include_in_primary_analysis",
]
EXPECTED_WINDOWS_COLUMNS = [
    "event_id",
    "event_date",
    "anchor_date",
    "window_label",
    "relative_day",
    "market_date",
    "event_group",
    "event_type",
    "description",
    "event_regime",
    "is_core_sample",
    "is_pandemic_period",
    "is_post_pandemic",
    "include_in_primary_analysis",
    "sp500",
    "nasdaq",
    "shanghai",
    "vix",
    "sp500_return",
    "nasdaq_return",
    "shanghai_return",
]
EXPECTED_ABNORMAL_COLUMNS = [
    "event_id",
    "window_label",
    "relative_day",
    "market_date",
    "sp500_return",
    "expected_return",
    "abnormal_return",
]
EXPECTED_FORMAL_CAR_COLUMNS = [
    "event_id",
    "window_label",
    "formal_car_sp500",
    "volatility_sp500",
    "car_sign",
]
EXPECTED_COMPARISON_COLUMNS = [
    "sample_label",
    "window_label",
    "mean_formal_car_sp500",
    "mean_volatility_sp500",
    "positive_share",
    "negative_share",
    "neutral_share",
]


def ok(message: str) -> None:
    print(f"OK: {message}")


def require_columns(dataframe: pd.DataFrame, expected_columns: list[str], filename: str) -> None:
    missing_columns = [column for column in expected_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"ERRO: colunas ausentes em {filename}: {missing_columns}")


def main() -> None:
    for path in [
        EVENTS_MASTER_FILE,
        WINDOWS_FILE,
        COVERAGE_FILE,
        ABNORMAL_FILE,
        FORMAL_CAR_FILE,
        COMPARISON_FILE,
        MARKET_FEATURES_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"ERRO: arquivo nao encontrado: {path}")
    ok("arquivos mestre encontrados")

    events_master = pd.read_csv(EVENTS_MASTER_FILE)
    market_features = pd.read_csv(MARKET_FEATURES_FILE)
    event_windows = pd.read_csv(WINDOWS_FILE)
    coverage = pd.read_csv(COVERAGE_FILE)
    abnormal_returns = pd.read_csv(ABNORMAL_FILE)
    formal_car_master = pd.read_csv(FORMAL_CAR_FILE)
    core_vs_full = pd.read_csv(COMPARISON_FILE)

    require_columns(events_master, EXPECTED_EVENT_COLUMNS, "events_master.csv")
    require_columns(event_windows, EXPECTED_WINDOWS_COLUMNS, "event_windows_master.csv")
    require_columns(abnormal_returns, EXPECTED_ABNORMAL_COLUMNS, "abnormal_returns_master.csv")
    require_columns(formal_car_master, EXPECTED_FORMAL_CAR_COLUMNS, "formal_car_master.csv")
    require_columns(core_vs_full, EXPECTED_COMPARISON_COLUMNS, "core_vs_full_comparison.csv")
    ok("colunas esperadas presentes")

    if len(events_master) != 67:
        raise ValueError(f"ERRO: events_master.csv deve ter 67 eventos, mas encontrou {len(events_master)}")
    ok("67 eventos detectados em events_master.csv")

    for dataframe, columns in [
        (events_master, ["date"]),
        (event_windows, ["event_date", "anchor_date", "market_date"]),
        (coverage, ["event_date", "anchor_date"]),
        (abnormal_returns, ["event_date", "anchor_date", "market_date"]),
    ]:
        for column in columns:
            dataframe[column] = pd.to_datetime(dataframe[column], errors="raise")
    ok("datas parseaveis")

    if coverage["event_id"].nunique() != 67:
        raise ValueError("ERRO: event_windows_master_coverage.csv deve cobrir os 67 eventos.")

    market_features["date"] = pd.to_datetime(market_features["date"], errors="raise")
    last_market_date = market_features["date"].max()
    expected_covered_events = (
        coverage.loc[coverage["is_complete"], "event_id"].drop_duplicates().shape[0]
    )
    if event_windows["event_id"].nunique() != expected_covered_events:
        raise ValueError("ERRO: quantidade de eventos em event_windows_master.csv nao bate com a cobertura completa.")
    if formal_car_master["event_id"].nunique() != expected_covered_events:
        raise ValueError("ERRO: quantidade de eventos em formal_car_master.csv nao bate com a cobertura completa.")
    ok("cobertura coerente entre janelas e event study")

    if set(event_windows["window_label"].unique()) != WINDOW_LABELS:
        raise ValueError("ERRO: janelas esperadas nao encontradas em event_windows_master.csv.")
    if set(formal_car_master["window_label"].unique()) != WINDOW_LABELS:
        raise ValueError("ERRO: janelas esperadas nao encontradas em formal_car_master.csv.")
    if set(core_vs_full["window_label"].unique()) != WINDOW_LABELS:
        raise ValueError("ERRO: janelas esperadas nao encontradas em core_vs_full_comparison.csv.")
    ok("janelas detectadas")

    if event_windows.duplicated(subset=["event_id", "window_label", "relative_day"]).any():
        raise ValueError("ERRO: duplicidade em (event_id, window_label, relative_day) em event_windows_master.csv.")
    if abnormal_returns.duplicated(subset=["event_id", "window_label", "relative_day"]).any():
        raise ValueError("ERRO: duplicidade em abnormal_returns_master.csv.")
    if formal_car_master.duplicated(subset=["event_id", "window_label"]).any():
        raise ValueError("ERRO: duplicidade em formal_car_master.csv.")
    ok("sem duplicidade")

    if abnormal_returns["expected_return"].isna().all():
        raise ValueError("ERRO: expected_return esta totalmente vazio.")
    if abnormal_returns["abnormal_return"].isna().all():
        raise ValueError("ERRO: abnormal_return esta totalmente vazio.")
    if formal_car_master["formal_car_sp500"].isna().all():
        raise ValueError("ERRO: formal_car_sp500 esta totalmente vazio.")
    if formal_car_master["volatility_sp500"].isna().all():
        raise ValueError("ERRO: volatility_sp500 esta totalmente vazio.")
    ok("metricas principais calculadas")

    if set(core_vs_full["sample_label"].unique()) != {"core_sample", "full_market_covered_sample"}:
        raise ValueError("ERRO: sample_label inesperado em core_vs_full_comparison.csv.")
    ok("comparacao core vs expandido encontrada")

    if (events_master.loc[events_master["date"] > last_market_date, "include_in_primary_analysis"]).any():
        raise ValueError("ERRO: eventos fora da cobertura de mercado nao deveriam entrar na analise primaria.")
    ok("flags de amostra principal preservadas")

    print("\nVALIDACAO CONCLUIDA")


if __name__ == "__main__":
    main()
