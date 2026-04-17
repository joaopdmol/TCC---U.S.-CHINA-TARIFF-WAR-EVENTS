from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
FORMAL_CAR_FILE = BASE_DIR / "DATA" / "formal_car_by_event.csv"
VOLATILITY_FILE = BASE_DIR / "DATA" / "volatility_by_event.csv"
OUTPUT_FILE = BASE_DIR / "DATA" / "return_vs_volatility.csv"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not FORMAL_CAR_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {FORMAL_CAR_FILE}. "
            "Rode primeiro build_formal_car_tables.py."
        )
    if not VOLATILITY_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {VOLATILITY_FILE}. "
            "Rode primeiro build_window_volatility.py."
        )

    return pd.read_csv(FORMAL_CAR_FILE), pd.read_csv(VOLATILITY_FILE)


def main() -> None:
    formal_car_by_event, volatility_by_event = load_inputs()

    return_vs_volatility = formal_car_by_event.merge(
        volatility_by_event[
            [
                "event_id",
                "window_label",
                "volatility_sp500",
                "volatility_nasdaq",
                "volatility_shanghai",
                "n_obs_sp500_return",
            ]
        ],
        on=["event_id", "window_label"],
        how="left",
        validate="one_to_one",
    )

    return_vs_volatility["car_sign"] = return_vs_volatility["formal_car_sp500"].apply(
        lambda value: "positive" if value > 0 else ("negative" if value < 0 else "neutral")
    )
    return_vs_volatility["abs_formal_car_sp500"] = return_vs_volatility["formal_car_sp500"].abs()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    return_vs_volatility.to_csv(OUTPUT_FILE, index=False)

    print(f"Arquivo salvo em: {OUTPUT_FILE}")
    print("\nPrimeiras linhas:")
    print(return_vs_volatility.head().to_string(index=False))


if __name__ == "__main__":
    main()
