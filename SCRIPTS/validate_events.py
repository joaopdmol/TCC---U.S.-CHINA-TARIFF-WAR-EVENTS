from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
EVENTS_PATH = BASE_DIR / "DATA" / "events.csv"
EXPECTED_COLUMNS = [
    "event_id",
    "date",
    "event_type",
    "event_group",
    "description",
]
EXPECTED_GROUPS = {"escalation", "relief"}
EXPECTED_EVENT_COUNT = 17


def main() -> None:
    events = pd.read_csv(EVENTS_PATH)

    if list(events.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Colunas esperadas: {EXPECTED_COLUMNS}. "
            f"Colunas encontradas: {list(events.columns)}"
        )
    print("OK: colunas corretas")

    if len(events) != EXPECTED_EVENT_COUNT:
        raise ValueError(
            f"events.csv deve conter exatamente {EXPECTED_EVENT_COUNT} eventos. "
            f"Quantidade encontrada: {len(events)}"
        )
    print("OK: 17 eventos detectados")

    if events["date"].isna().sum() != 0:
        raise ValueError("A coluna date nao pode ter valores vazios.")
    print("OK: datas sem valores vazios")

    if events["event_group"].isna().sum() != 0:
        raise ValueError("A coluna event_group nao pode ter valores vazios.")

    dates = pd.to_datetime(events["date"], errors="raise")
    print("OK: datas parseaveis")

    if dates.duplicated().sum() != 0:
        raise ValueError("Nao deve haver datas duplicadas em events.csv.")
    print("OK: sem datas duplicadas")

    unique_groups = set(events["event_group"].unique().tolist())
    if unique_groups != EXPECTED_GROUPS:
        raise ValueError(
            f"event_group deve conter apenas {sorted(EXPECTED_GROUPS)}. "
            f"Valores encontrados: {sorted(unique_groups)}"
        )
    print("OK: event_group restrito a escalation e relief")

    print(f"Arquivo: {EVENTS_PATH}")
    print(f"Numero de eventos: {len(events)}")
    print(f"Grupos encontrados: {sorted(unique_groups)}")
    print("\nCronologia:")
    print(events.assign(date=dates).sort_values("date").to_string(index=False))
    print("\nVALIDACAO CONCLUIDA")


if __name__ == "__main__":
    main()
