from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "DATA" / "event_windows_long.csv"
OUTPUT_BY_EVENT = BASE_DIR / "DATA" / "event_summary_by_event.csv"
OUTPUT_BY_GROUP = BASE_DIR / "DATA" / "event_summary_by_group.csv"
OUTPUT_BEFORE_AFTER = BASE_DIR / "DATA" / "event_summary_before_after.csv"

RETURN_COLUMNS = ["sp500_return", "nasdaq_return", "shanghai_return"]
REQUIRED_COLUMNS = [
    "event_id",
    "event_group",
    "event_type",
    "description",
    "window_label",
    "relative_day",
    *RETURN_COLUMNS,
]
WINDOW_ORDER = ["m1_p1", "m3_p3", "m5_p5"]
PERIOD_ORDER = ["before", "event_day", "after"]


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

    event_windows["period"] = pd.Categorical(
        event_windows["relative_day"].map(classify_period),
        categories=PERIOD_ORDER,
        ordered=True,
    )
    event_windows["window_label"] = pd.Categorical(
        event_windows["window_label"],
        categories=WINDOW_ORDER,
        ordered=True,
    )
    return event_windows


def classify_period(relative_day: int) -> str:
    if relative_day < 0:
        return "before"
    if relative_day == 0:
        return "event_day"
    return "after"


def summary_aggregations() -> dict[str, tuple[str, str]]:
    aggregations: dict[str, tuple[str, str]] = {
        "row_count": ("event_id", "size"),
    }
    for column in RETURN_COLUMNS:
        aggregations[f"{column}_mean"] = (column, "mean")
        aggregations[f"{column}_median"] = (column, "median")
        aggregations[f"{column}_std"] = (column, "std")
        aggregations[f"{column}_n"] = (column, "count")
    return aggregations


def build_summary_table(
    event_windows: pd.DataFrame,
    group_columns: list[str],
    include_event_count: bool,
) -> pd.DataFrame:
    summary = (
        event_windows.groupby(group_columns, dropna=False, observed=True)
        .agg(**summary_aggregations())
        .reset_index()
    )

    if include_event_count:
        event_counts = (
            event_windows.groupby(group_columns, dropna=False, observed=True)["event_id"]
            .nunique()
            .reset_index(name="event_count")
        )
        summary = summary.merge(event_counts, on=group_columns, how="left")

    return summary.sort_values(group_columns).reset_index(drop=True)


def main() -> None:
    event_windows = load_event_windows()

    summary_by_event = build_summary_table(
        event_windows,
        ["event_id", "event_group", "event_type", "description", "window_label", "period"],
        include_event_count=False,
    )
    summary_by_group = build_summary_table(
        event_windows,
        ["event_group", "window_label", "period"],
        include_event_count=True,
    )
    summary_before_after = build_summary_table(
        event_windows,
        ["window_label", "period"],
        include_event_count=True,
    )

    OUTPUT_BY_EVENT.parent.mkdir(parents=True, exist_ok=True)
    summary_by_event.to_csv(OUTPUT_BY_EVENT, index=False)
    summary_by_group.to_csv(OUTPUT_BY_GROUP, index=False)
    summary_before_after.to_csv(OUTPUT_BEFORE_AFTER, index=False)

    print(f"Arquivo salvo em: {OUTPUT_BY_EVENT}")
    print(f"Arquivo salvo em: {OUTPUT_BY_GROUP}")
    print(f"Arquivo salvo em: {OUTPUT_BEFORE_AFTER}")
    print("\nResumo por grupo:")
    print(summary_by_group.to_string(index=False))


if __name__ == "__main__":
    main()
