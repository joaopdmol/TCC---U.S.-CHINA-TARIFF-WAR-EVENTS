from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "DATA" / "market_data.csv"
OUTPUT_FILE = BASE_DIR / "DATA" / "market_features.csv"

REQUIRED_COLUMNS = ["date", "sp500", "nasdaq", "shanghai", "vix"]
PRICE_COLUMNS = ["sp500", "nasdaq", "shanghai"]


def require_input_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de entrada nao encontrado: {path}. "
            "Rode primeiro o script de download da base de mercado."
        )


def load_market_data() -> pd.DataFrame:
    require_input_file(INPUT_FILE)

    market_data = pd.read_csv(INPUT_FILE)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in market_data.columns]
    if missing_columns:
        raise ValueError(
            "market_data.csv nao contem todas as colunas esperadas. "
            f"Colunas ausentes: {missing_columns}"
        )

    market_data["date"] = pd.to_datetime(market_data["date"], errors="raise")

    if market_data["date"].duplicated().any():
        duplicated_dates = market_data.loc[market_data["date"].duplicated(), "date"].dt.strftime("%Y-%m-%d")
        raise ValueError(
            "market_data.csv contem datas duplicadas, o que impediria o calculo seguro dos retornos. "
            f"Datas duplicadas: {duplicated_dates.tolist()}"
        )

    market_data = market_data.sort_values("date").reset_index(drop=True)
    return market_data


def compute_simple_return(series: pd.Series) -> pd.Series:
    valid_series = series.dropna()
    if valid_series.empty:
        return pd.Series(index=series.index, dtype="float64")

    returns = valid_series.pct_change(fill_method=None)
    return returns.reindex(series.index)


def compute_log_return(series: pd.Series) -> pd.Series:
    valid_series = series.dropna()
    if valid_series.empty:
        return pd.Series(index=series.index, dtype="float64")

    ratios = valid_series.div(valid_series.shift(1))
    log_returns = np.log(ratios.where(ratios > 0))
    return log_returns.reindex(series.index)


def build_market_features(market_data: pd.DataFrame) -> pd.DataFrame:
    features = market_data.copy()

    for column in PRICE_COLUMNS:
        features[f"{column}_return"] = compute_simple_return(features[column])
        features[f"{column}_log_return"] = compute_log_return(features[column])

    ordered_columns = REQUIRED_COLUMNS + [
        "sp500_return",
        "nasdaq_return",
        "shanghai_return",
        "sp500_log_return",
        "nasdaq_log_return",
        "shanghai_log_return",
    ]
    return features[ordered_columns]


def main() -> None:
    market_data = load_market_data()
    market_features = build_market_features(market_data)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    market_features.to_csv(OUTPUT_FILE, index=False)

    print(f"Arquivo salvo em: {OUTPUT_FILE}")
    print(f"Linhas: {len(market_features)}")
    print(
        "Periodo: "
        f"{market_features['date'].min().date()} ate {market_features['date'].max().date()}"
    )
    print("\nPrimeiras linhas:")
    print(market_features.head().to_string(index=False))
    print("\nValores ausentes por coluna:")
    print(market_features.isna().sum().to_string())


if __name__ == "__main__":
    main()
