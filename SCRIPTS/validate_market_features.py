from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "DATA" / "market_features.csv"

REQUIRED_COLUMNS = [
    "date",
    "sp500",
    "nasdaq",
    "shanghai",
    "vix",
    "sp500_return",
    "nasdaq_return",
    "shanghai_return",
]
RAW_COLUMNS = ["sp500", "nasdaq", "shanghai", "vix"]
RETURN_COLUMNS = ["sp500_return", "nasdaq_return", "shanghai_return"]


def ok(message: str) -> None:
    print(f"OK: {message}")


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"ERRO: arquivo nao encontrado: {INPUT_FILE}")
    ok("arquivo encontrado")

    market_features = pd.read_csv(INPUT_FILE)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in market_features.columns]
    if missing_columns:
        raise ValueError(f"ERRO: colunas esperadas ausentes: {missing_columns}")
    ok("colunas esperadas presentes")

    if len(market_features) <= 500:
        raise ValueError(
            "ERRO: a base analitica precisa ter mais de 500 linhas. "
            f"Linhas encontradas: {len(market_features)}"
        )
    ok(f"numero de linhas valido ({len(market_features)})")

    market_features["date"] = pd.to_datetime(market_features["date"], errors="raise")
    ok("coluna date interpretada como data")

    if market_features["date"].duplicated().any():
        duplicated_dates = (
            market_features.loc[market_features["date"].duplicated(), "date"]
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )
        raise ValueError(f"ERRO: datas duplicadas encontradas: {duplicated_dates}")
    ok("datas sem duplicidade")

    empty_raw_columns = [column for column in RAW_COLUMNS if market_features[column].dropna().empty]
    if empty_raw_columns:
        raise ValueError(f"ERRO: colunas brutas totalmente vazias: {empty_raw_columns}")
    ok("colunas brutas principais preenchidas")

    empty_return_columns = [column for column in RETURN_COLUMNS if market_features[column].dropna().empty]
    if empty_return_columns:
        raise ValueError(f"ERRO: retornos nao calculados corretamente: {empty_return_columns}")
    ok("retornos calculados")

    print("\nResumo rapido:")
    print(
        f"Periodo: {market_features['date'].min().date()} ate "
        f"{market_features['date'].max().date()}"
    )
    print("Valores ausentes por coluna:")
    print(market_features[REQUIRED_COLUMNS].isna().sum().to_string())
    print("\nVALIDACAO CONCLUIDA")


if __name__ == "__main__":
    main()
