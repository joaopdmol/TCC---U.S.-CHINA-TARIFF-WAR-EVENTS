from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
EVENT_WINDOWS_FILE = BASE_DIR / "DATA" / "event_windows_master.csv"
MARKET_FEATURES_FILE = BASE_DIR / "DATA" / "market_features.csv"
OUTPUT_ABNORMAL = BASE_DIR / "DATA" / "abnormal_returns_master.csv"
OUTPUT_FORMAL_CAR = BASE_DIR / "DATA" / "formal_car_master.csv"
OUTPUT_COMPARISON = BASE_DIR / "DATA" / "core_vs_full_comparison.csv"

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
    "event_regime",
    "is_core_sample",
    "is_pandemic_period",
    "is_post_pandemic",
    "include_in_primary_analysis",
]
REQUIRED_MARKET_FEATURES_COLUMNS = ["date", "sp500"]

# Event study formal simples:
# estima o retorno esperado como a media do retorno diario do S&P 500
# entre -30 e -6 pregões relativos ao evento.
ESTIMATION_START_OFFSET = -30
ESTIMATION_END_OFFSET = -6
MIN_ESTIMATION_OBS = 20
NEUTRAL_THRESHOLD = 0.005


def load_event_windows() -> pd.DataFrame:
    if not EVENT_WINDOWS_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {EVENT_WINDOWS_FILE}. "
            "Rode primeiro build_event_windows_master.py."
        )

    event_windows = pd.read_csv(EVENT_WINDOWS_FILE)
    missing_columns = [
        column for column in REQUIRED_EVENT_WINDOW_COLUMNS if column not in event_windows.columns
    ]
    if missing_columns:
        raise ValueError(f"Colunas ausentes em event_windows_master.csv: {missing_columns}")

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
    trading_history["sp500_return"] = trading_history["sp500"].pct_change(fill_method=None)
    return trading_history


def classify_sign(value: float) -> str:
    if pd.isna(value):
        return "missing"
    if value > NEUTRAL_THRESHOLD:
        return "positive"
    if value < -NEUTRAL_THRESHOLD:
        return "negative"
    return "neutral"


def compute_formal_car(series: pd.Series) -> float:
    valid_returns = series.dropna()
    if valid_returns.empty:
        return float("nan")
    return valid_returns.sum()


def build_expected_return_table(
    event_windows: pd.DataFrame,
    sp500_history: pd.DataFrame,
) -> pd.DataFrame:
    event_metadata = (
        event_windows[
            [
                "event_id",
                "event_date",
                "anchor_date",
                "event_group",
                "event_type",
                "description",
                "event_regime",
                "is_core_sample",
                "is_pandemic_period",
                "is_post_pandemic",
                "include_in_primary_analysis",
            ]
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
                "event_regime": event.event_regime,
                "is_core_sample": event.is_core_sample,
                "is_pandemic_period": event.is_pandemic_period,
                "is_post_pandemic": event.is_post_pandemic,
                "include_in_primary_analysis": event.include_in_primary_analysis,
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


def build_core_vs_full_comparison(formal_car_master: pd.DataFrame) -> pd.DataFrame:
    comparison_rows = []
    sample_filters = {
        "core_sample": formal_car_master["is_core_sample"],
        "full_market_covered_sample": pd.Series(True, index=formal_car_master.index),
    }

    for sample_label, sample_mask in sample_filters.items():
        sample_df = formal_car_master.loc[sample_mask].copy()
        for window_label, window_df in sample_df.groupby("window_label", observed=True):
            comparison_rows.append(
                {
                    "sample_label": sample_label,
                    "window_label": window_label,
                    "n_events": int(window_df["event_id"].nunique()),
                    "mean_formal_car_sp500": window_df["formal_car_sp500"].mean(),
                    "median_formal_car_sp500": window_df["formal_car_sp500"].median(),
                    "mean_volatility_sp500": window_df["volatility_sp500"].mean(),
                    "median_volatility_sp500": window_df["volatility_sp500"].median(),
                    "positive_share": window_df["car_sign"].eq("positive").mean(),
                    "negative_share": window_df["car_sign"].eq("negative").mean(),
                    "neutral_share": window_df["car_sign"].eq("neutral").mean(),
                    "neutral_threshold": NEUTRAL_THRESHOLD,
                }
            )

    return pd.DataFrame(comparison_rows).sort_values(
        ["sample_label", "window_label"]
    ).reset_index(drop=True)


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
        on=[
            "event_id",
            "event_date",
            "anchor_date",
            "event_group",
            "event_type",
            "description",
            "event_regime",
            "is_core_sample",
            "is_pandemic_period",
            "is_post_pandemic",
            "include_in_primary_analysis",
        ],
        how="left",
    )
    abnormal_returns["abnormal_return"] = (
        abnormal_returns["sp500_return"] - abnormal_returns["expected_return"]
    )

    abnormal_columns = [
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
        "event_regime",
        "is_core_sample",
        "is_pandemic_period",
        "is_post_pandemic",
        "include_in_primary_analysis",
        "estimation_window_start",
        "estimation_window_end",
        "estimation_n_obs",
        "estimation_sufficient_data",
    ]
    abnormal_returns = abnormal_returns[abnormal_columns].sort_values(
        ["event_id", "window_label", "relative_day"]
    ).reset_index(drop=True)

    formal_car_master = (
        abnormal_returns.groupby(
            [
                "event_id",
                "event_group",
                "event_type",
                "description",
                "event_regime",
                "is_core_sample",
                "is_pandemic_period",
                "is_post_pandemic",
                "include_in_primary_analysis",
                "window_label",
            ],
            dropna=False,
            observed=True,
        )
        .agg(
            formal_car_sp500=("abnormal_return", compute_formal_car),
            n_obs=("abnormal_return", "count"),
            expected_return=("expected_return", "first"),
            estimation_n_obs=("estimation_n_obs", "first"),
            estimation_sufficient_data=("estimation_sufficient_data", "first"),
            volatility_sp500=("sp500_return", "std"),
        )
        .reset_index()
        .sort_values(["event_id", "window_label"])
        .reset_index(drop=True)
    )
    formal_car_master["car_sign"] = formal_car_master["formal_car_sp500"].apply(classify_sign)
    formal_car_master["neutral_threshold"] = NEUTRAL_THRESHOLD

    core_vs_full_comparison = build_core_vs_full_comparison(formal_car_master)

    OUTPUT_ABNORMAL.parent.mkdir(parents=True, exist_ok=True)
    abnormal_returns.to_csv(OUTPUT_ABNORMAL, index=False)
    formal_car_master.to_csv(OUTPUT_FORMAL_CAR, index=False)
    core_vs_full_comparison.to_csv(OUTPUT_COMPARISON, index=False)

    print(f"Arquivo salvo em: {OUTPUT_ABNORMAL}")
    print(f"Arquivo salvo em: {OUTPUT_FORMAL_CAR}")
    print(f"Arquivo salvo em: {OUTPUT_COMPARISON}")
    print(f"Eventos cobertos no event study mestre: {formal_car_master['event_id'].nunique()}")
    print("\nResumo core vs expandido:")
    print(core_vs_full_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
