from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "DATA" / "event_windows_long.csv"
OUTPUT_BY_EVENT = BASE_DIR / "DATA" / "volatility_by_event.csv"
OUTPUT_BY_GROUP = BASE_DIR / "DATA" / "volatility_by_group.csv"

RETURN_COLUMNS = ["sp500_return", "nasdaq_return", "shanghai_return"]
REQUIRED_COLUMNS = [
    "event_id",
    "event_group",
    "event_type",
    "description",
    "window_label",
    *RETURN_COLUMNS,
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


def event_level_aggregations() -> dict[str, tuple[str, str]]:
    aggregations: dict[str, tuple[str, str]] = {}
    for column in RETURN_COLUMNS:
        label = column.replace("_return", "")
        aggregations[f"volatility_{label}"] = (column, "std")
        aggregations[f"n_obs_{label}_return"] = (column, "count")
    return aggregations


def group_level_aggregations() -> dict[str, tuple[str, str]]:
    aggregations: dict[str, tuple[str, str]] = {
        "n_events": ("event_id", "nunique"),
    }
    for label in ["sp500", "nasdaq", "shanghai"]:
        aggregations[f"volatility_{label}_mean"] = (f"volatility_{label}", "mean")
        aggregations[f"volatility_{label}_median"] = (f"volatility_{label}", "median")
        aggregations[f"volatility_{label}_std"] = (f"volatility_{label}", "std")
    return aggregations


def main() -> None:
    event_windows = load_event_windows()

    volatility_by_event = (
        event_windows.groupby(
            ["event_id", "event_group", "event_type", "description", "window_label"],
            dropna=False,
            observed=True,
        )
        .agg(**event_level_aggregations())
        .reset_index()
        .sort_values(["event_id", "window_label"])
        .reset_index(drop=True)
    )

    volatility_by_group = (
        volatility_by_event.groupby(["event_group", "window_label"], dropna=False, observed=True)
        .agg(**group_level_aggregations())
        .reset_index()
        .sort_values(["event_group", "window_label"])
        .reset_index(drop=True)
    )

    OUTPUT_BY_EVENT.parent.mkdir(parents=True, exist_ok=True)
    volatility_by_event.to_csv(OUTPUT_BY_EVENT, index=False)
    volatility_by_group.to_csv(OUTPUT_BY_GROUP, index=False)

    print(f"Arquivo salvo em: {OUTPUT_BY_EVENT}")
    print(f"Arquivo salvo em: {OUTPUT_BY_GROUP}")
    print("\nResumo por grupo:")
    print(volatility_by_group.to_string(index=False))


if __name__ == "__main__":
    main()
