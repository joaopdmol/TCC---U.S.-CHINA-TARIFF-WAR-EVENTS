from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
EVENT_WINDOWS_FILE = BASE_DIR / "DATA" / "event_windows_long.csv"
MARKET_FEATURES_FILE = BASE_DIR / "DATA" / "market_features.csv"
OUTPUT_FILE = BASE_DIR / "DATA" / "abnormal_returns_long.csv"

REQUIRED_EVENT_WINDOW_COLUMNS = [
    "event_id",
    "event_date",
    "anchor_date",
    "window_label",
    "relative_day",
    "market_date",
    "event_group",
    "event_type",
    "description",
]
REQUIRED_MARKET_FEATURES_COLUMNS = ["date", "sp500"]

# Janela de estimacao simples e conservadora:
# usa pregões relativos ao evento de -30 ate -6 para evitar contaminar
# o retorno esperado com a propria janela do evento.
ESTIMATION_START_OFFSET = -30
ESTIMATION_END_OFFSET = -6
MIN_ESTIMATION_OBS = 20


def load_event_windows() -> pd.DataFrame:
    if not EVENT_WINDOWS_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {EVENT_WINDOWS_FILE}. "
            "Rode primeiro build_event_windows.py."
        )

    event_windows = pd.read_csv(EVENT_WINDOWS_FILE)
    missing_columns = [
        column for column in REQUIRED_EVENT_WINDOW_COLUMNS if column not in event_windows.columns
    ]
    if missing_columns:
        raise ValueError(f"Colunas ausentes em event_windows_long.csv: {missing_columns}")

    for column in ["event_date", "anchor_date", "market_date"]:
        event_windows[column] = pd.to_datetime(event_windows[column], errors="raise")

    return event_windows.sort_values(["event_id", "window_label", "relative_day"]).reset_index(drop=True)


def load_market_features() -> pd.DataFrame:
    if not MARKET_FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {MARKET_FEATURES_FILE}. "
            "Rode primeiro build_market_features.py."
        )

    market_features = pd.read_csv(MARKET_FEATURES_FILE)
    missing_columns = [
        column for column in REQUIRED_MARKET_FEATURES_COLUMNS if column not in market_features.columns
    ]
    if missing_columns:
        raise ValueError(f"Colunas ausentes em market_features.csv: {missing_columns}")

    market_features["date"] = pd.to_datetime(market_features["date"], errors="raise")
    return market_features.sort_values("date").reset_index(drop=True)


def build_sp500_trading_history(market_features: pd.DataFrame) -> pd.DataFrame:
    trading_history = market_features.loc[market_features["sp500"].notna(), ["date", "sp500"]].copy()
    trading_history = trading_history.sort_values("date").reset_index(drop=True)

    # Para o event study formal simples, recalculamos o retorno na propria
    # linha de pregão do S&P 500, evitando efeitos do calendario combinado.
    trading_history["sp500_return"] = trading_history["sp500"].pct_change(fill_method=None)
    return trading_history


def build_expected_return_table(
    event_windows: pd.DataFrame,
    sp500_history: pd.DataFrame,
) -> pd.DataFrame:
    event_metadata = (
        event_windows[
            ["event_id", "event_date", "anchor_date", "event_group", "event_type", "description"]
        ]
        .drop_duplicates()
        .sort_values("event_id")
        .reset_index(drop=True)
    )

    trading_calendar = pd.Index(sp500_history["date"])
    expected_rows = []

    for event in event_metadata.itertuples(index=False):
        if event.anchor_date not in trading_calendar:
            raise ValueError(
                f"anchor_date do evento {event.event_id} nao foi encontrada no calendario do S&P 500."
            )

        anchor_position = trading_calendar.get_loc(event.anchor_date)
        start_position = anchor_position + ESTIMATION_START_OFFSET
        end_position = anchor_position + ESTIMATION_END_OFFSET

        if start_position < 0 or end_position < 0 or end_position <= start_position:
            estimation_slice = sp500_history.iloc[0:0].copy()
        else:
            estimation_slice = sp500_history.iloc[start_position : end_position + 1].copy()

        valid_returns = estimation_slice["sp500_return"].dropna()
        has_sufficient_data = len(valid_returns) >= MIN_ESTIMATION_OBS
        expected_return = valid_returns.mean() if has_sufficient_data else float("nan")

        expected_rows.append(
            {
                "event_id": event.event_id,
                "event_date": event.event_date,
                "anchor_date": event.anchor_date,
                "event_group": event.event_group,
                "event_type": event.event_type,
                "description": event.description,
                "estimation_window_start": (
                    estimation_slice["date"].min() if not estimation_slice.empty else pd.NaT
                ),
                "estimation_window_end": (
                    estimation_slice["date"].max() if not estimation_slice.empty else pd.NaT
                ),
                "estimation_n_obs": int(len(valid_returns)),
                "estimation_sufficient_data": bool(has_sufficient_data),
                "expected_return": expected_return,
            }
        )

    return pd.DataFrame(expected_rows)


def main() -> None:
    event_windows = load_event_windows()
    market_features = load_market_features()
    sp500_history = build_sp500_trading_history(market_features)
    expected_return_table = build_expected_return_table(event_windows, sp500_history)

    abnormal_returns = event_windows.drop(columns=["sp500_return"], errors="ignore").merge(
        sp500_history[["date", "sp500_return"]].rename(columns={"date": "market_date"}),
        on="market_date",
        how="left",
    )
    abnormal_returns = abnormal_returns.merge(
        expected_return_table,
        on=["event_id", "event_date", "anchor_date", "event_group", "event_type", "description"],
        how="left",
    )

    abnormal_returns["abnormal_return"] = (
        abnormal_returns["sp500_return"] - abnormal_returns["expected_return"]
    )

    ordered_columns = [
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
        "estimation_window_start",
        "estimation_window_end",
        "estimation_n_obs",
        "estimation_sufficient_data",
    ]
    abnormal_returns = abnormal_returns[ordered_columns].sort_values(
        ["event_id", "window_label", "relative_day"]
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    abnormal_returns.to_csv(OUTPUT_FILE, index=False)

    print(f"Arquivo salvo em: {OUTPUT_FILE}")
    print(f"Linhas geradas: {len(abnormal_returns)}")
    print("\nResumo de cobertura da estimacao:")
    print(
        expected_return_table[
            [
                "event_id",
                "estimation_window_start",
                "estimation_window_end",
                "estimation_n_obs",
                "expected_return",
                "estimation_sufficient_data",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
