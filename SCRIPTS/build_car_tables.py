from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "DATA" / "event_windows_long.csv"
OUTPUT_BY_EVENT = BASE_DIR / "DATA" / "car_by_event.csv"
OUTPUT_BY_GROUP = BASE_DIR / "DATA" / "car_by_group.csv"

REQUIRED_COLUMNS = [
    "event_id",
    "event_group",
    "event_type",
    "description",
    "window_label",
    "sp500_return",
]
WINDOW_ORDER = ["m1_p1", "m3_p3", "m5_p5"]


def load_event_windows() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {INPUT_FILE}. "
            "Rode primeiro build_event_windows.py."
        )

    event_windows = pd.read_csv(INPUT_FILE)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in event_windows.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes em event_windows_long.csv: {missing_columns}")

    event_windows["window_label"] = pd.Categorical(
        event_windows["window_label"],
        categories=WINDOW_ORDER,
        ordered=True,
    )
    return event_windows


def compute_cumulative_return(series: pd.Series) -> float:
    valid_returns = series.dropna()
    if valid_returns.empty:
        return float("nan")
    return (1.0 + valid_returns).prod() - 1.0


def main() -> None:
    event_windows = load_event_windows()

    car_by_event = (
        event_windows.groupby(
            ["event_id", "event_group", "event_type", "description", "window_label"],
            dropna=False,
            observed=True,
        )
        .agg(
            car_simple_sp500=("sp500_return", compute_cumulative_return),
            n_obs_sp500_return=("sp500_return", "count"),
        )
        .reset_index()
        .sort_values(["event_id", "window_label"])
        .reset_index(drop=True)
    )

    car_by_group = (
        car_by_event.groupby(["event_group", "window_label"], dropna=False, observed=True)
        .agg(
            car_simple_sp500_mean=("car_simple_sp500", "mean"),
            car_simple_sp500_median=("car_simple_sp500", "median"),
            car_simple_sp500_std=("car_simple_sp500", "std"),
            n_events=("event_id", "nunique"),
            n_obs_sp500_return_total=("n_obs_sp500_return", "sum"),
        )
        .reset_index()
        .sort_values(["event_group", "window_label"])
        .reset_index(drop=True)
    )

    OUTPUT_BY_EVENT.parent.mkdir(parents=True, exist_ok=True)
    car_by_event.to_csv(OUTPUT_BY_EVENT, index=False)
    car_by_group.to_csv(OUTPUT_BY_GROUP, index=False)

    print(f"Arquivo salvo em: {OUTPUT_BY_EVENT}")
    print(f"Arquivo salvo em: {OUTPUT_BY_GROUP}")
    print("\nResumo por grupo:")
    print(car_by_group.to_string(index=False))


if __name__ == "__main__":
    main()
