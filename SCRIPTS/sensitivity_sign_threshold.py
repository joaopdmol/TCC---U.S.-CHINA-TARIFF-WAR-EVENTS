from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "DATA" / "formal_car_by_event.csv"
OUTPUT_FILE = BASE_DIR / "DATA" / "sensitivity_sign_results.csv"

THRESHOLDS = [0.0025, 0.0050, 0.0075]
WINDOW_ORDER = ["m1_p1", "m3_p3", "m5_p5"]
GROUP_ORDER = ["escalation", "relief"]
SIGN_ORDER = ["negative", "neutral", "positive"]


def load_formal_car() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {INPUT_FILE}")
    dataframe = pd.read_csv(INPUT_FILE)
    required_columns = ["event_id", "event_group", "window_label", "formal_car_sp500"]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes em formal_car_by_event.csv: {missing_columns}")
    return dataframe


def classify_sign(value: float, threshold: float) -> str:
    if pd.isna(value):
        return "missing"
    if value > threshold:
        return "positive"
    if value < -threshold:
        return "negative"
    return "neutral"


def main() -> None:
    formal_car = load_formal_car()
    rows = []
    full_index = pd.MultiIndex.from_product(
        [GROUP_ORDER, WINDOW_ORDER, SIGN_ORDER],
        names=["event_group", "window_label", "sign_label"],
    )

    for threshold in THRESHOLDS:
        classified = formal_car.copy()
        classified["sign_label"] = classified["formal_car_sp500"].apply(lambda value: classify_sign(value, threshold))

        grouped = (
            classified.groupby(["event_group", "window_label", "sign_label"], observed=True)
            .agg(event_count=("event_id", "count"))
            .reset_index()
        )
        totals = (
            classified.groupby(["event_group", "window_label"], observed=True)
            .agg(total_events=("event_id", "count"))
            .reset_index()
        )
        grouped = (
            grouped.set_index(["event_group", "window_label", "sign_label"])
            .reindex(full_index, fill_value=0)
            .reset_index()
        )
        grouped = grouped.merge(totals, on=["event_group", "window_label"], how="left")
        grouped["event_share"] = grouped["event_count"] / grouped["total_events"]
        grouped["neutral_threshold"] = threshold
        rows.append(grouped)

    results = pd.concat(rows, ignore_index=True)
    results["window_label"] = pd.Categorical(results["window_label"], categories=WINDOW_ORDER, ordered=True)
    results["event_group"] = pd.Categorical(results["event_group"], categories=GROUP_ORDER, ordered=True)
    results["sign_label"] = pd.Categorical(results["sign_label"], categories=SIGN_ORDER, ordered=True)
    results = results.sort_values(["neutral_threshold", "event_group", "window_label", "sign_label"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"Arquivo salvo em: {OUTPUT_FILE}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
