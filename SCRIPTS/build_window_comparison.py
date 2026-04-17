from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_BY_EVENT = BASE_DIR / "DATA" / "formal_car_by_event.csv"
INPUT_BY_GROUP = BASE_DIR / "DATA" / "formal_car_by_group.csv"
OUTPUT_BY_EVENT = BASE_DIR / "DATA" / "window_comparison_by_event.csv"
OUTPUT_BY_GROUP = BASE_DIR / "DATA" / "window_comparison_by_group.csv"

WINDOW_ORDER = ["m1_p1", "m3_p3", "m5_p5"]


def load_formal_car_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not INPUT_BY_EVENT.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {INPUT_BY_EVENT}. "
            "Rode primeiro build_formal_car_tables.py."
        )
    if not INPUT_BY_GROUP.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {INPUT_BY_GROUP}. "
            "Rode primeiro build_formal_car_tables.py."
        )

    by_event = pd.read_csv(INPUT_BY_EVENT)
    by_group = pd.read_csv(INPUT_BY_GROUP)
    return by_event, by_group


def add_comparison_columns(table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    table[f"{prefix}_diff_m3_minus_m1"] = table[f"{prefix}_m3_p3"] - table[f"{prefix}_m1_p1"]
    table[f"{prefix}_diff_m5_minus_m3"] = table[f"{prefix}_m5_p5"] - table[f"{prefix}_m3_p3"]
    table[f"{prefix}_diff_m5_minus_m1"] = table[f"{prefix}_m5_p5"] - table[f"{prefix}_m1_p1"]
    return table


def build_by_event(by_event: pd.DataFrame) -> pd.DataFrame:
    pivoted = (
        by_event.pivot_table(
            index=["event_id", "event_group", "event_type", "description"],
            columns="window_label",
            values="formal_car_sp500",
            aggfunc="first",
        )
        .reindex(columns=WINDOW_ORDER)
        .reset_index()
    )

    pivoted = pivoted.rename(
        columns={
            "m1_p1": "formal_car_m1_p1",
            "m3_p3": "formal_car_m3_p3",
            "m5_p5": "formal_car_m5_p5",
        }
    )
    pivoted = add_comparison_columns(pivoted, "formal_car")
    return pivoted.sort_values("event_id").reset_index(drop=True)


def build_by_group(by_group: pd.DataFrame) -> pd.DataFrame:
    pivoted = (
        by_group.pivot_table(
            index="event_group",
            columns="window_label",
            values="mean_formal_car_sp500",
            aggfunc="first",
        )
        .reindex(columns=WINDOW_ORDER)
        .reset_index()
    )

    pivoted = pivoted.rename(
        columns={
            "m1_p1": "mean_formal_car_m1_p1",
            "m3_p3": "mean_formal_car_m3_p3",
            "m5_p5": "mean_formal_car_m5_p5",
        }
    )
    pivoted = add_comparison_columns(pivoted, "mean_formal_car")
    return pivoted.sort_values("event_group").reset_index(drop=True)


def main() -> None:
    by_event, by_group = load_formal_car_tables()
    comparison_by_event = build_by_event(by_event)
    comparison_by_group = build_by_group(by_group)

    OUTPUT_BY_EVENT.parent.mkdir(parents=True, exist_ok=True)
    comparison_by_event.to_csv(OUTPUT_BY_EVENT, index=False)
    comparison_by_group.to_csv(OUTPUT_BY_GROUP, index=False)

    print(f"Arquivo salvo em: {OUTPUT_BY_EVENT}")
    print(f"Arquivo salvo em: {OUTPUT_BY_GROUP}")
    print("\nResumo por grupo:")
    print(comparison_by_group.to_string(index=False))


if __name__ == "__main__":
    main()
