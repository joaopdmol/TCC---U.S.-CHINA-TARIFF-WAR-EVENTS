from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "DATA" / "events_expanded.csv"
MASTER_OUTPUT = BASE_DIR / "DATA" / "events_master.csv"
CORE_OUTPUT = BASE_DIR / "DATA" / "events_core.csv"
PANDEMIC_OUTPUT = BASE_DIR / "DATA" / "events_pandemic.csv"
POST_OUTPUT = BASE_DIR / "DATA" / "events_post.csv"

REQUIRED_COLUMNS = ["event_id", "date", "event_type", "event_group", "description"]
CORE_END_DATE = pd.Timestamp("2020-02-14")
PANDEMIC_END_DATE = pd.Timestamp("2023-12-31")


def classify_regime(event_date: pd.Timestamp) -> str:
    if event_date <= CORE_END_DATE:
        return "core"
    if event_date <= PANDEMIC_END_DATE:
        return "pandemic"
    return "post"


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {INPUT_FILE}")

    events = pd.read_csv(INPUT_FILE)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in events.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes em events_expanded.csv: {missing_columns}")

    events["date"] = pd.to_datetime(events["date"], errors="raise")
    events = events.sort_values("date").reset_index(drop=True)

    events["event_regime"] = events["date"].apply(classify_regime)
    events["is_core_sample"] = events["event_regime"].eq("core")
    events["is_pandemic_period"] = events["event_regime"].eq("pandemic")
    events["is_post_pandemic"] = events["event_regime"].eq("post")
    events["include_in_primary_analysis"] = events["is_core_sample"]

    ordered_columns = REQUIRED_COLUMNS + [
        "event_regime",
        "is_core_sample",
        "is_pandemic_period",
        "is_post_pandemic",
        "include_in_primary_analysis",
    ]
    events = events[ordered_columns]

    master_events = events.copy()
    core_events = master_events.loc[master_events["event_regime"] == "core"].copy()
    pandemic_events = master_events.loc[master_events["event_regime"] == "pandemic"].copy()
    post_events = master_events.loc[master_events["event_regime"] == "post"].copy()

    MASTER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    master_events.to_csv(MASTER_OUTPUT, index=False)
    core_events.to_csv(CORE_OUTPUT, index=False)
    pandemic_events.to_csv(PANDEMIC_OUTPUT, index=False)
    post_events.to_csv(POST_OUTPUT, index=False)

    print(f"Arquivo salvo em: {MASTER_OUTPUT}")
    print(f"Arquivo salvo em: {CORE_OUTPUT}")
    print(f"Arquivo salvo em: {PANDEMIC_OUTPUT}")
    print(f"Arquivo salvo em: {POST_OUTPUT}")
    print("\nContagem por regime:")
    print(master_events["event_regime"].value_counts().to_string())


if __name__ == "__main__":
    main()
