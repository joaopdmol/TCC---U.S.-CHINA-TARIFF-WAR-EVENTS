from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
EVENTS_FILE = BASE_DIR / "DATA" / "events_master.csv"
MARKET_FEATURES_FILE = BASE_DIR / "DATA" / "market_features.csv"
OUTPUT_LONG_FILE = BASE_DIR / "DATA" / "event_windows_master.csv"
OUTPUT_COVERAGE_FILE = BASE_DIR / "DATA" / "event_windows_master_coverage.csv"

REQUIRED_EVENT_COLUMNS = [
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


def load_events_master() -> pd.DataFrame:
    if not EVENTS_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {EVENTS_FILE}. "
            "Rode primeiro build_events_master.py."
        )

    events = pd.read_csv(EVENTS_FILE)
    missing_columns = [column for column in REQUIRED_EVENT_COLUMNS if column not in events.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes em events_master.csv: {missing_columns}")

    events["date"] = pd.to_datetime(events["date"], errors="raise")
    events = events.sort_values("date").reset_index(drop=True)

    if events["event_id"].duplicated().any():
        raise ValueError("events_master.csv contem event_id duplicado.")

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
        has_anchor = position < len(trading_calendar)
        anchor_date = trading_calendar[position] if has_anchor else pd.NaT

        anchor_rows.append(
            {
                "event_id": event.event_id,
                "event_date": event.date,
                "anchor_date": anchor_date,
                "event_type": event.event_type,
                "event_group": event.event_group,
                "description": event.description,
                "event_regime": event.event_regime,
                "is_core_sample": event.is_core_sample,
                "is_pandemic_period": event.is_pandemic_period,
                "is_post_pandemic": event.is_post_pandemic,
                "include_in_primary_analysis": event.include_in_primary_analysis,
                "has_anchor": has_anchor,
                "anchor_shift_calendar_days": (
                    int((anchor_date - event.date).days) if has_anchor else pd.NA
                ),
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

    raw_rows = []
    coverage_rows = []

    for event in anchor_map.itertuples(index=False):
        if event.has_anchor:
            anchor_position = trading_calendar.get_loc(event.anchor_date)

        for window_label, window_size in WINDOW_SPECS:
            expected_count = (window_size * 2) + 1
            row_count = 0
            min_relative_day = pd.NA
            max_relative_day = pd.NA

            if not event.has_anchor:
                coverage_rows.append(
                    {
                        "event_id": event.event_id,
                        "event_date": event.event_date,
                        "anchor_date": pd.NaT,
                        "event_type": event.event_type,
                        "event_group": event.event_group,
                        "description": event.description,
                        "event_regime": event.event_regime,
                        "is_core_sample": event.is_core_sample,
                        "is_pandemic_period": event.is_pandemic_period,
                        "is_post_pandemic": event.is_post_pandemic,
                        "include_in_primary_analysis": event.include_in_primary_analysis,
                        "window_label": window_label,
                        "expected_count": expected_count,
                        "row_count": 0,
                        "min_relative_day": pd.NA,
                        "max_relative_day": pd.NA,
                        "anchor_shift_calendar_days": pd.NA,
                        "has_anchor": False,
                        "coverage_status": "outside_market_coverage",
                        "is_complete": False,
                    }
                )
                continue

            for relative_day in range(-window_size, window_size + 1):
                market_position = anchor_position + relative_day
                if market_position < 0 or market_position >= len(trading_calendar):
                    continue

                market_date = trading_calendar[market_position]
                market_row = market_by_date.loc[market_date]
                row_count += 1
                min_relative_day = relative_day if pd.isna(min_relative_day) else min(min_relative_day, relative_day)
                max_relative_day = relative_day if pd.isna(max_relative_day) else max(max_relative_day, relative_day)

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
                    "event_regime": event.event_regime,
                    "is_core_sample": event.is_core_sample,
                    "is_pandemic_period": event.is_pandemic_period,
                    "is_post_pandemic": event.is_post_pandemic,
                    "include_in_primary_analysis": event.include_in_primary_analysis,
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

                raw_rows.append(row)

            is_complete = row_count == expected_count
            coverage_rows.append(
                {
                    "event_id": event.event_id,
                    "event_date": event.event_date,
                    "anchor_date": event.anchor_date,
                    "event_type": event.event_type,
                    "event_group": event.event_group,
                    "description": event.description,
                    "event_regime": event.event_regime,
                    "is_core_sample": event.is_core_sample,
                    "is_pandemic_period": event.is_pandemic_period,
                    "is_post_pandemic": event.is_post_pandemic,
                    "include_in_primary_analysis": event.include_in_primary_analysis,
                    "window_label": window_label,
                    "expected_count": expected_count,
                    "row_count": row_count,
                    "min_relative_day": min_relative_day,
                    "max_relative_day": max_relative_day,
                    "anchor_shift_calendar_days": event.anchor_shift_calendar_days,
                    "has_anchor": True,
                    "coverage_status": "complete" if is_complete else "incomplete_window",
                    "is_complete": is_complete,
                }
            )

    coverage = pd.DataFrame(coverage_rows).sort_values(
        ["event_id", "window_label"]
    ).reset_index(drop=True)

    raw_long = pd.DataFrame(raw_rows)
    if raw_long.empty:
        raise ValueError("Nenhuma linha foi gerada para event_windows_master.csv.")

    complete_pairs = coverage.loc[
        coverage["is_complete"], ["event_id", "window_label"]
    ].drop_duplicates()

    event_windows_master = raw_long.merge(
        complete_pairs.assign(_keep=True),
        on=["event_id", "window_label"],
        how="inner",
    ).drop(columns="_keep")

    event_windows_master = event_windows_master.sort_values(
        ["event_id", "window_label", "relative_day"]
    ).reset_index(drop=True)

    return event_windows_master, coverage


def main() -> None:
    events_master = load_events_master()
    market_features = load_market_features()
    anchor_map = build_anchor_map(events_master, market_features)
    event_windows_master, coverage = build_event_windows(anchor_map, market_features)

    OUTPUT_LONG_FILE.parent.mkdir(parents=True, exist_ok=True)
    event_windows_master.to_csv(OUTPUT_LONG_FILE, index=False)
    coverage.to_csv(OUTPUT_COVERAGE_FILE, index=False)

    covered_events = coverage.loc[coverage["is_complete"], "event_id"].nunique()
    total_events = coverage["event_id"].nunique()

    print(f"Arquivo salvo em: {OUTPUT_LONG_FILE}")
    print(f"Arquivo salvo em: {OUTPUT_COVERAGE_FILE}")
    print(f"Eventos no mestre: {total_events}")
    print(f"Eventos com cobertura completa de janela: {covered_events}")
    print(f"Linhas finais em event_windows_master.csv: {len(event_windows_master)}")
    print("\nResumo de cobertura por status:")
    print(coverage["coverage_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
