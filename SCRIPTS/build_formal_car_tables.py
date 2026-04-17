from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "DATA" / "abnormal_returns_long.csv"
OUTPUT_BY_EVENT = BASE_DIR / "DATA" / "formal_car_by_event.csv"
OUTPUT_BY_GROUP = BASE_DIR / "DATA" / "formal_car_by_group.csv"

REQUIRED_COLUMNS = [
    "event_id",
    "event_group",
    "event_type",
    "description",
    "window_label",
    "abnormal_return",
    "expected_return",
    "estimation_n_obs",
    "estimation_sufficient_data",
]


def load_abnormal_returns() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {INPUT_FILE}. "
            "Rode primeiro build_abnormal_returns.py."
        )

    abnormal_returns = pd.read_csv(INPUT_FILE)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in abnormal_returns.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes em abnormal_returns_long.csv: {missing_columns}")

    return abnormal_returns


def compute_formal_car(series: pd.Series) -> float:
    valid_abnormal_returns = series.dropna()
    if valid_abnormal_returns.empty:
        return float("nan")
    return valid_abnormal_returns.sum()


def main() -> None:
    abnormal_returns = load_abnormal_returns()

    formal_car_by_event = (
        abnormal_returns.groupby(
            ["event_id", "event_group", "event_type", "description", "window_label"],
            dropna=False,
            observed=True,
        )
        .agg(
            formal_car_sp500=("abnormal_return", compute_formal_car),
            n_obs=("abnormal_return", "count"),
            expected_return=("expected_return", "first"),
            estimation_n_obs=("estimation_n_obs", "first"),
            estimation_sufficient_data=("estimation_sufficient_data", "first"),
        )
        .reset_index()
        .sort_values(["event_id", "window_label"])
        .reset_index(drop=True)
    )

    formal_car_by_group = (
        formal_car_by_event.groupby(["event_group", "window_label"], dropna=False, observed=True)
        .agg(
            mean_formal_car_sp500=("formal_car_sp500", "mean"),
            median_formal_car_sp500=("formal_car_sp500", "median"),
            std_formal_car_sp500=("formal_car_sp500", "std"),
            n_events=("event_id", "nunique"),
        )
        .reset_index()
        .sort_values(["event_group", "window_label"])
        .reset_index(drop=True)
    )

    OUTPUT_BY_EVENT.parent.mkdir(parents=True, exist_ok=True)
    formal_car_by_event.to_csv(OUTPUT_BY_EVENT, index=False)
    formal_car_by_group.to_csv(OUTPUT_BY_GROUP, index=False)

    print(f"Arquivo salvo em: {OUTPUT_BY_EVENT}")
    print(f"Arquivo salvo em: {OUTPUT_BY_GROUP}")
    print("\nResumo por grupo:")
    print(formal_car_by_group.to_string(index=False))


if __name__ == "__main__":
    main()
