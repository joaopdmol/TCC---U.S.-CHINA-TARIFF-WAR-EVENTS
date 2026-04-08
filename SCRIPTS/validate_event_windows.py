from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_LONG_FILE = BASE_DIR / "DATA" / "event_windows_long.csv"
INPUT_COVERAGE_FILE = BASE_DIR / "DATA" / "event_windows_coverage.csv"

REQUIRED_COLUMNS = [
    "event_id",
    "event_date",
    "anchor_date",
    "window_label",
    "relative_day",
    "market_date",
    "event_type",
    "event_group",
    "description",
    "sp500",
    "nasdaq",
    "shanghai",
    "vix",
    "sp500_return",
    "nasdaq_return",
    "shanghai_return",
]
WINDOW_EXPECTED_COUNTS = {
    "m1_p1": 3,
    "m3_p3": 7,
    "m5_p5": 11,
}
MARKET_COLUMNS = [
    "sp500",
    "nasdaq",
    "shanghai",
    "vix",
    "sp500_return",
    "nasdaq_return",
    "shanghai_return",
]


def ok(message: str) -> None:
    print(f"OK: {message}")


def validate_coverage_file() -> None:
    if not INPUT_COVERAGE_FILE.exists():
        raise FileNotFoundError(f"ERRO: arquivo nao encontrado: {INPUT_COVERAGE_FILE}")

    coverage = pd.read_csv(INPUT_COVERAGE_FILE)
    required_columns = [
        "event_id",
        "event_date",
        "anchor_date",
        "window_label",
        "expected_count",
        "row_count",
        "is_complete",
    ]
    missing_columns = [column for column in required_columns if column not in coverage.columns]
    if missing_columns:
        raise ValueError(f"ERRO: colunas ausentes em event_windows_coverage.csv: {missing_columns}")

    coverage["event_date"] = pd.to_datetime(coverage["event_date"], errors="raise")
    coverage["anchor_date"] = pd.to_datetime(coverage["anchor_date"], errors="raise")

    if coverage[["event_id", "window_label"]].duplicated().any():
        raise ValueError("ERRO: cobertura contem pares event_id/window_label duplicados.")

    if sorted(coverage["window_label"].unique().tolist()) != sorted(WINDOW_EXPECTED_COUNTS.keys()):
        raise ValueError("ERRO: cobertura nao contem todas as janelas esperadas.")

    mismatched_counts = coverage[
        coverage["row_count"] != coverage["window_label"].map(WINDOW_EXPECTED_COUNTS)
    ]
    if not mismatched_counts.empty:
        raise ValueError(
            "ERRO: quantidade de linhas por janela nao bate com o esperado. "
            f"Linhas problematicas: {mismatched_counts[['event_id', 'window_label', 'row_count']].to_dict('records')}"
        )

    incomplete_windows = coverage.loc[coverage["is_complete"] != True]
    if not incomplete_windows.empty:
        raise ValueError(
            "ERRO: existem janelas incompletas na cobertura. "
            f"Linhas problematicas: {incomplete_windows[['event_id', 'window_label']].to_dict('records')}"
        )

    ok("arquivo de cobertura encontrado e consistente")


def main() -> None:
    if not INPUT_LONG_FILE.exists():
        raise FileNotFoundError(f"ERRO: arquivo nao encontrado: {INPUT_LONG_FILE}")
    ok("arquivo encontrado")

    event_windows = pd.read_csv(INPUT_LONG_FILE)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in event_windows.columns]
    if missing_columns:
        raise ValueError(f"ERRO: colunas esperadas ausentes: {missing_columns}")
    ok("colunas esperadas presentes")

    unique_events = sorted(event_windows["event_id"].unique().tolist())
    if len(unique_events) != 17:
        raise ValueError(f"ERRO: esperado 17 eventos, mas foram encontrados {len(unique_events)}")
    ok("17 eventos detectados")

    windows_found = sorted(event_windows["window_label"].unique().tolist())
    if windows_found != sorted(WINDOW_EXPECTED_COUNTS.keys()):
        raise ValueError(
            f"ERRO: janelas esperadas {sorted(WINDOW_EXPECTED_COUNTS.keys())}, "
            f"mas foram encontradas {windows_found}"
        )
    ok("janelas detectadas")

    for date_column in ["event_date", "anchor_date", "market_date"]:
        event_windows[date_column] = pd.to_datetime(event_windows[date_column], errors="raise")
    ok("datas parseaveis")

    if event_windows.duplicated(["event_id", "window_label", "relative_day"]).any():
        raise ValueError("ERRO: ha duplicidade em (event_id, window_label, relative_day)")
    ok("sem duplicidade")

    anchor_rows = event_windows.loc[event_windows["relative_day"] == 0]
    expected_anchor_pairs = len(unique_events) * len(WINDOW_EXPECTED_COUNTS)
    if len(anchor_rows) != expected_anchor_pairs:
        raise ValueError(
            "ERRO: relative_day = 0 nao foi encontrado para todos os eventos e janelas. "
            f"Quantidade encontrada: {len(anchor_rows)}"
        )
    if (anchor_rows["anchor_date"] != anchor_rows["market_date"]).any():
        raise ValueError("ERRO: ha linhas ancora em que anchor_date nao coincide com market_date.")
    ok("linhas ancora presentes")

    empty_market_columns = [column for column in MARKET_COLUMNS if event_windows[column].dropna().empty]
    if empty_market_columns:
        raise ValueError(f"ERRO: colunas de mercado/retorno totalmente vazias: {empty_market_columns}")
    ok("colunas principais de mercado e retorno preenchidas")

    row_counts = event_windows.groupby(["event_id", "window_label"]).size().reset_index(name="row_count")
    row_counts["expected_count"] = row_counts["window_label"].map(WINDOW_EXPECTED_COUNTS)
    if (row_counts["row_count"] <= 0).any():
        raise ValueError("ERRO: existe ao menos uma janela vazia.")
    if (row_counts["row_count"] != row_counts["expected_count"]).any():
        raise ValueError("ERRO: existe ao menos uma janela com quantidade inesperada de linhas.")
    ok("nenhuma janela ficou vazia")

    validate_coverage_file()

    print("\nResumo rapido:")
    print(
        event_windows.groupby("window_label")
        .agg(linhas=("event_id", "count"), eventos=("event_id", "nunique"))
        .to_string()
    )
    print("\nVALIDACAO CONCLUIDA")


if __name__ == "__main__":
    main()
