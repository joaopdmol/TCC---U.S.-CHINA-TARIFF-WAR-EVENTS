from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "DATA" / "formal_car_master.csv"
OUTPUT_FILE = BASE_DIR / "DATA" / "sensitivity_sample_results.csv"

WINDOW_ORDER = ["m1_p1", "m3_p3", "m5_p5"]


def load_formal_car_master() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {INPUT_FILE}")

    dataframe = pd.read_csv(INPUT_FILE)
    required_columns = [
        "event_id",
        "window_label",
        "event_regime",
        "is_core_sample",
        "formal_car_sp500",
        "volatility_sp500",
        "car_sign",
    ]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes em formal_car_master.csv: {missing_columns}")

    dataframe["is_core_sample"] = dataframe["is_core_sample"].astype(str).str.lower().eq("true")
    return dataframe


def main() -> None:
    formal_car_master = load_formal_car_master()

    sample_masks = {
        "core_only": formal_car_master["is_core_sample"],
        "core_plus_pandemic_covered": formal_car_master["event_regime"].isin(["core", "pandemic"]),
        "full_market_covered": pd.Series(True, index=formal_car_master.index),
    }
    sample_notes = {
        "core_only": "Amostra principal com 17 eventos.",
        "core_plus_pandemic_covered": "Eventos core e pandemic com cobertura de mercado.",
        "full_market_covered": "Todos os eventos do master que possuem cobertura de mercado.",
    }

    rows = []
    for sample_label, sample_mask in sample_masks.items():
        sample_df = formal_car_master.loc[sample_mask].copy()
        grouped = (
            sample_df.groupby("window_label", observed=True)
            .agg(
                n_events=("event_id", "nunique"),
                mean_formal_car_sp500=("formal_car_sp500", "mean"),
                mean_volatility_sp500=("volatility_sp500", "mean"),
                positive_share=("car_sign", lambda series: series.eq("positive").mean()),
                negative_share=("car_sign", lambda series: series.eq("negative").mean()),
                neutral_share=("car_sign", lambda series: series.eq("neutral").mean()),
            )
            .reset_index()
        )
        grouped["sample_label"] = sample_label
        grouped["sample_note"] = sample_notes[sample_label]
        rows.append(grouped)

    results = pd.concat(rows, ignore_index=True)
    results["window_label"] = pd.Categorical(results["window_label"], categories=WINDOW_ORDER, ordered=True)
    results = results.sort_values(["sample_label", "window_label"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"Arquivo salvo em: {OUTPUT_FILE}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
