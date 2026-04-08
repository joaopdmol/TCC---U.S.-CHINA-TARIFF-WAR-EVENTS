from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MARKET_DATA_PATH = BASE_DIR / "DATA" / "market_data.csv"


def main() -> None:
    expected_columns = ["date", "sp500", "nasdaq", "shanghai", "vix"]

    if not MARKET_DATA_PATH.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {MARKET_DATA_PATH}")

    market_data = pd.read_csv(MARKET_DATA_PATH)

    assert list(market_data.columns) == expected_columns, (
        f"Colunas esperadas: {expected_columns}. "
        f"Colunas encontradas: {list(market_data.columns)}"
    )
    assert len(market_data) > 500, "A base deve ter mais de 500 linhas."
    assert market_data.isna().all().sum() == 0, "Nenhuma coluna pode estar totalmente vazia."

    market_data["date"] = pd.to_datetime(market_data["date"], errors="raise")

    print("market_data.csv validado com sucesso.")
    print(f"Arquivo: {MARKET_DATA_PATH}")
    print(f"Linhas: {len(market_data)}")
    print(f"Periodo: {market_data['date'].min().date()} ate {market_data['date'].max().date()}")
    print("\nValores ausentes por coluna:")
    print(market_data.isna().sum().to_string())


if __name__ == "__main__":
    main()
