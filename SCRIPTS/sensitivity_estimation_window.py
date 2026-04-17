from pathlib import Path

import pandas as pd

from build_abnormal_returns import (
    build_sp500_trading_history,
    load_event_windows,
    load_market_features,
)


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "DATA" / "sensitivity_estimation_results.csv"

ESTIMATION_CONFIGS = [
    ("baseline_m30_m6", -30, -6),
    ("alt_m20_m6", -20, -6),
    ("alt_m40_m6", -40, -6),
]
WINDOW_ORDER = ["m1_p1", "m3_p3", "m5_p5"]


def compute_formal_car(series: pd.Series) -> float:
    valid_values = series.dropna()
    if valid_values.empty:
        return float("nan")
    return valid_values.sum()


def min_required_obs(start_offset: int, end_offset: int) -> int:
    expected_obs = abs(end_offset - start_offset) + 1
    return min(20, expected_obs)


def build_expected_return_table(
    event_windows: pd.DataFrame,
    sp500_history: pd.DataFrame,
    start_offset: int,
    end_offset: int,
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
    required_obs = min_required_obs(start_offset, end_offset)
    rows = []

    for event in event_metadata.itertuples(index=False):
        anchor_position = trading_calendar.get_loc(event.anchor_date)
        start_position = anchor_position + start_offset
        end_position = anchor_position + end_offset

        if start_position < 0 or end_position < 0 or end_position <= start_position:
            estimation_slice = sp500_history.iloc[0:0].copy()
        else:
            estimation_slice = sp500_history.iloc[start_position : end_position + 1].copy()

        valid_returns = estimation_slice["sp500_return"].dropna()
        sufficient_data = len(valid_returns) >= required_obs
        rows.append(
            {
                "event_id": event.event_id,
                "expected_return": valid_returns.mean() if sufficient_data else float("nan"),
                "estimation_n_obs": int(len(valid_returns)),
                "required_obs": int(required_obs),
                "estimation_sufficient_data": bool(sufficient_data),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    event_windows = load_event_windows()
    market_features = load_market_features()
    sp500_history = build_sp500_trading_history(market_features)

    summary_rows = []

    for config_label, start_offset, end_offset in ESTIMATION_CONFIGS:
        expected_returns = build_expected_return_table(
            event_windows,
            sp500_history,
            start_offset,
            end_offset,
        )

        abnormal_returns = event_windows[["event_id", "window_label", "market_date"]].merge(
            sp500_history[["date", "sp500_return"]].rename(columns={"date": "market_date"}),
            on="market_date",
            how="left",
        )
        abnormal_returns = abnormal_returns.merge(
            expected_returns,
            on="event_id",
            how="left",
        )
        abnormal_returns["abnormal_return"] = (
            abnormal_returns["sp500_return"] - abnormal_returns["expected_return"]
        )

        formal_car = (
            abnormal_returns.groupby(["event_id", "window_label"], observed=True)
            .agg(
                formal_car_sp500=("abnormal_return", compute_formal_car),
                estimation_n_obs=("estimation_n_obs", "first"),
                required_obs=("required_obs", "first"),
                estimation_sufficient_data=("estimation_sufficient_data", "first"),
            )
            .reset_index()
        )

        aggregated = (
            formal_car.groupby("window_label", observed=True)
            .agg(
                n_events=("event_id", "nunique"),
                mean_formal_car_sp500=("formal_car_sp500", "mean"),
                median_formal_car_sp500=("formal_car_sp500", "median"),
                std_formal_car_sp500=("formal_car_sp500", "std"),
                sufficient_events=("estimation_sufficient_data", "sum"),
            )
            .reset_index()
        )

        for row in aggregated.itertuples(index=False):
            summary_rows.append(
                {
                    "analysis_scope": "core",
                    "estimation_config": config_label,
                    "estimation_start_offset": start_offset,
                    "estimation_end_offset": end_offset,
                    "required_obs": int(min_required_obs(start_offset, end_offset)),
                    "window_label": row.window_label,
                    "n_events": int(row.n_events),
                    "sufficient_events": int(row.sufficient_events),
                    "mean_formal_car_sp500": row.mean_formal_car_sp500,
                    "median_formal_car_sp500": row.median_formal_car_sp500,
                    "std_formal_car_sp500": row.std_formal_car_sp500,
                }
            )

    results = pd.DataFrame(summary_rows)
    baseline = (
        results.loc[results["estimation_config"] == "baseline_m30_m6", ["window_label", "mean_formal_car_sp500"]]
        .rename(columns={"mean_formal_car_sp500": "baseline_mean_formal_car_sp500"})
    )
    results = results.merge(baseline, on="window_label", how="left")
    results["abs_diff_vs_baseline"] = (
        results["mean_formal_car_sp500"] - results["baseline_mean_formal_car_sp500"]
    )
    results["relative_change_pct_vs_baseline"] = (
        results["abs_diff_vs_baseline"] / results["baseline_mean_formal_car_sp500"].abs()
    ) * 100
    results["window_label"] = pd.Categorical(results["window_label"], categories=WINDOW_ORDER, ordered=True)
    results = results.sort_values(["estimation_config", "window_label"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"Arquivo salvo em: {OUTPUT_FILE}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
