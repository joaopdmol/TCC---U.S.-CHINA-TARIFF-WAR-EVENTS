from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "DATA" / "formal_car_by_event.csv"
OUTPUT_BY_EVENT = BASE_DIR / "DATA" / "event_sign_analysis.csv"
OUTPUT_BY_GROUP = BASE_DIR / "DATA" / "group_sign_summary.csv"

# Faixa conservadora para classificar um CAR como aproximadamente neutro.
# A ideia aqui e operacional, nao estatistica: valores dentro de +/-0,5 p.p.
# sao tratados como economicamente proximos de zero nesta etapa.
NEUTRAL_THRESHOLD = 0.005


def load_formal_car_by_event() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {INPUT_FILE}. "
            "Rode primeiro build_formal_car_tables.py."
        )

    return pd.read_csv(INPUT_FILE)


def classify_sign(value: float) -> str:
    if pd.isna(value):
        return "missing"
    if value > NEUTRAL_THRESHOLD:
        return "positive"
    if value < -NEUTRAL_THRESHOLD:
        return "negative"
    return "neutral"


def main() -> None:
    formal_car_by_event = load_formal_car_by_event()

    event_sign_analysis = formal_car_by_event.copy()
    event_sign_analysis["sign_label"] = event_sign_analysis["formal_car_sp500"].apply(classify_sign)
    event_sign_analysis["neutral_threshold"] = NEUTRAL_THRESHOLD

    group_sign_summary = (
        event_sign_analysis.groupby(["event_group", "window_label", "sign_label"], dropna=False, observed=True)
        .agg(event_count=("event_id", "count"))
        .reset_index()
    )
    totals = (
        event_sign_analysis.groupby(["event_group", "window_label"], dropna=False, observed=True)
        .agg(total_events=("event_id", "count"))
        .reset_index()
    )
    group_sign_summary = group_sign_summary.merge(
        totals,
        on=["event_group", "window_label"],
        how="left",
    )
    group_sign_summary["event_share"] = (
        group_sign_summary["event_count"] / group_sign_summary["total_events"]
    )
    group_sign_summary["neutral_threshold"] = NEUTRAL_THRESHOLD
    group_sign_summary = group_sign_summary.sort_values(
        ["event_group", "window_label", "sign_label"]
    ).reset_index(drop=True)

    OUTPUT_BY_EVENT.parent.mkdir(parents=True, exist_ok=True)
    event_sign_analysis.to_csv(OUTPUT_BY_EVENT, index=False)
    group_sign_summary.to_csv(OUTPUT_BY_GROUP, index=False)

    print(f"Arquivo salvo em: {OUTPUT_BY_EVENT}")
    print(f"Arquivo salvo em: {OUTPUT_BY_GROUP}")
    print("\nResumo por grupo:")
    print(group_sign_summary.to_string(index=False))


if __name__ == "__main__":
    main()
