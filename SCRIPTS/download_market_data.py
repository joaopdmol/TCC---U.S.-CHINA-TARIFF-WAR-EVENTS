from pathlib import Path

import pandas as pd
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DATA"
OUTPUT_FILE = DATA_DIR / "market_data.csv"

START_DATE = "2017-01-01"
# yfinance usa o parametro end como limite exclusivo.
END_DATE = "2021-01-01"

TICKERS = {
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "shanghai": "000001.SS",
    "vix": "^VIX",
}


def download_series(ticker: str, column_name: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"Nenhum dado retornado para {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].copy()
    df.columns = [column_name]
    df.index.name = "date"
    return df


def main() -> None:
    frames = []

    for column_name, ticker in TICKERS.items():
        print(f"Baixando {column_name} ({ticker})...")
        series_df = download_series(ticker, column_name)
        frames.append(series_df)

    market_data = pd.concat(frames, axis=1, sort=False).sort_index()
    market_data.reset_index(inplace=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    market_data.to_csv(OUTPUT_FILE, index=False)

    print(f"\nArquivo salvo em: {OUTPUT_FILE}")
    print(market_data.head())
    print("\nColunas:")
    print(list(market_data.columns))


if __name__ == "__main__":
    main()
