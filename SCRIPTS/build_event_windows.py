from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
EVENTS_FILE = BASE_DIR / "DATA" / "events.csv"
MARKET_FEATURES_FILE = BASE_DIR / "DATA" / "market_features.csv"
OUTPUT_LONG_FILE = BASE_DIR / "DATA" / "event_windows_long.csv"
OUTPUT_COVERAGE_FILE = BASE_DIR / "DATA" / "event_windows_coverage.csv"

REQUIRED_EVENT_COLUMNS = ["event_id", "date", "event_type", "event_group", "description"]
REQUIRED_MARKET_COLUMNS = [
    "date",
    "sp500",
    "nasdaq",
    "shanghai",
    "vix",
    "sp500_return",
    "nasdaq_return",
    "shanghai_return",
]
OPTIONAL_MARKET_COLUMNS = [
    "sp500_log_return",
    "nasdaq_log_return",
    "shanghai_log_return",
]
WINDOW_SPECS = [
    ("m1_p1", 1),
    ("m3_p3", 3),
    ("m5_p5", 5),
]


def load_events() -> pd.DataFrame:
    if not EVENTS_FILE.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {EVENTS_FILE}")

    events = pd.read_csv(EVENTS_FILE)
    missing_columns = [column for column in REQUIRED_EVENT_COLUMNS if column not in events.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes em events.csv: {missing_columns}")

    events["date"] = pd.to_datetime(events["date"], errors="raise")
    events = events.sort_values("date").reset_index(drop=True)

    if events["event_id"].duplicated().any():
        raise ValueError("events.csv contem event_id duplicado.")

    return events


def load_market_features() -> pd.DataFrame:
    if not MARKET_FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {MARKET_FEATURES_FILE}. "
            "Rode primeiro build_market_features.py."
        )

    market_features = pd.read_csv(MARKET_FEATURES_FILE)
    missing_columns = [column for column in REQUIRED_MARKET_COLUMNS if column not in market_features.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes em market_features.csv: {missing_columns}")

    market_features["date"] = pd.to_datetime(market_features["date"], errors="raise")
    market_features = market_features.sort_values("date").reset_index(drop=True)

    if market_features["date"].duplicated().any():
        raise ValueError("market_features.csv contem datas duplicadas.")

    return market_features


def build_anchor_map(events: pd.DataFrame, market_features: pd.DataFrame) -> pd.DataFrame:
    trading_calendar = pd.Index(
        market_features.loc[market_features["sp500"].notna(), "date"]
        .drop_duplicates()
        .sort_values()
    )

    anchor_rows = []
    for event in events.itertuples(index=False):
        position = trading_calendar.searchsorted(event.date, side="left")
        if position >= len(trading_calendar):
            raise ValueError(
                f"Nao existe pregao do S&P 500 posterior ao evento {event.event_id} em {event.date.date()}."
            )

        anchor_date = trading_calendar[position]
        anchor_rows.append(
            {
                "event_id": event.event_id,
                "event_date": event.date,
                "anchor_date": anchor_date,
                "event_type": event.event_type,
                "event_group": event.event_group,
                "description": event.description,
            }
        )

    return pd.DataFrame(anchor_rows)


def build_event_windows(
    anchor_map: pd.DataFrame,
    market_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trading_calendar = pd.Index(
        market_features.loc[market_features["sp500"].notna(), "date"]
        .drop_duplicates()
        .sort_values()
    )
    market_by_date = market_features.set_index("date")
    available_optional_columns = [
        column for column in OPTIONAL_MARKET_COLUMNS if column in market_features.columns
    ]

    long_rows = []
    coverage_rows = []

    for event in anchor_map.itertuples(index=False):
        anchor_position = trading_calendar.get_loc(event.anchor_date)

        for window_label, window_size in WINDOW_SPECS:
            expected_count = (window_size * 2) + 1

            for relative_day in range(-window_size, window_size + 1):
                market_position = anchor_position + relative_day
                if market_position < 0 or market_position >= len(trading_calendar):
                    continue

                market_date = trading_calendar[market_position]
                market_row = market_by_date.loc[market_date]

                row = {
                    "event_id": event.event_id,
                    "event_date": event.event_date,
                    "anchor_date": event.anchor_date,
                    "window_label": window_label,
                    "relative_day": relative_day,
                    "market_date": market_date,
                    "event_type": event.event_type,
                    "event_group": event.event_group,
                    "description": event.description,
                    "sp500": market_row["sp500"],
                    "nasdaq": market_row["nasdaq"],
                    "shanghai": market_row["shanghai"],
                    "vix": market_row["vix"],
                    "sp500_return": market_row["sp500_return"],
                    "nasdaq_return": market_row["nasdaq_return"],
                    "shanghai_return": market_row["shanghai_return"],
                }

                for column in available_optional_columns:
                    row[column] = market_row[column]

                long_rows.append(row)

            coverage_rows.append(
                {
                    "event_id": event.event_id,
                    "event_date": event.event_date,
                    "anchor_date": event.anchor_date,
                    "event_type": event.event_type,
                    "event_group": event.event_group,
                    "window_label": window_label,
                    "expected_count": expected_count,
                }
            )

    event_windows_long = pd.DataFrame(long_rows)
    if event_windows_long.empty:
        raise ValueError("Nenhuma linha foi gerada para as janelas de evento.")

    event_windows_long = event_windows_long.sort_values(
        ["event_id", "window_label", "relative_day"]
    ).reset_index(drop=True)

    event_windows_coverage = pd.DataFrame(coverage_rows).merge(
        event_windows_long.groupby(["event_id", "window_label"])
        .agg(
            row_count=("market_date", "count"),
            min_relative_day=("relative_day", "min"),
            max_relative_day=("relative_day", "max"),
        )
        .reset_index(),
        on=["event_id", "window_label"],
        how="left",
    )

    event_windows_coverage["anchor_shift_calendar_days"] = (
        event_windows_coverage["anchor_date"] - event_windows_coverage["event_date"]
    ).dt.days
    event_windows_coverage["is_complete"] = (
        event_windows_coverage["row_count"] == event_windows_coverage["expected_count"]
    )

    return event_windows_long, event_windows_coverage


def main() -> None:
    events = load_events()
    market_features = load_market_features()
    anchor_map = build_anchor_map(events, market_features)
    event_windows_long, event_windows_coverage = build_event_windows(anchor_map, market_features)

    OUTPUT_LONG_FILE.parent.mkdir(parents=True, exist_ok=True)
    event_windows_long.to_csv(OUTPUT_LONG_FILE, index=False)
    event_windows_coverage.to_csv(OUTPUT_COVERAGE_FILE, index=False)

    print(f"Arquivo salvo em: {OUTPUT_LONG_FILE}")
    print(f"Linhas geradas: {len(event_windows_long)}")
    print(f"Eventos cobertos: {event_windows_long['event_id'].nunique()}")
    print(f"Janelas geradas: {sorted(event_windows_long['window_label'].unique().tolist())}")
    print("\nResumo de cobertura:")
    print(event_windows_coverage.to_string(index=False))


if __name__ == "__main__":
    main()
