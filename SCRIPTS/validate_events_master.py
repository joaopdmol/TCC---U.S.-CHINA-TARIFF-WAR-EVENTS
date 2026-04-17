from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_FILE = BASE_DIR / "DATA" / "events_master.csv"
CORE_FILE = BASE_DIR / "DATA" / "events_core.csv"
PANDEMIC_FILE = BASE_DIR / "DATA" / "events_pandemic.csv"
POST_FILE = BASE_DIR / "DATA" / "events_post.csv"

EXPECTED_COLUMNS = [
    "event_id",
    "date",
    "event_type",
    "event_group",
    "description",
    "event_regime",
    "is_core_sample",
    "is_pandemic_period",
    "is_post_pandemic",
    "include_in_primary_analysis",
]
EXPECTED_COUNTS = {
    "all": 67,
    "core": 17,
    "pandemic": 19,
    "post": 31,
}
CORE_END_DATE = pd.Timestamp("2020-02-14")
PANDEMIC_END_DATE = pd.Timestamp("2023-12-31")


def ok(message: str) -> None:
    print(f"OK: {message}")


def main() -> None:
    for path in [MASTER_FILE, CORE_FILE, PANDEMIC_FILE, POST_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"ERRO: arquivo nao encontrado: {path}")
    ok("arquivos mestre e subconjuntos encontrados")

    master_events = pd.read_csv(MASTER_FILE)
    if list(master_events.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"ERRO: colunas esperadas {EXPECTED_COLUMNS}, "
            f"mas encontradas {list(master_events.columns)}"
        )
    ok("colunas corretas")

    if len(master_events) != EXPECTED_COUNTS["all"]:
        raise ValueError(
            f"ERRO: events_master.csv deve ter {EXPECTED_COUNTS['all']} eventos, "
            f"mas encontrou {len(master_events)}"
        )
    ok("67 eventos detectados")

    master_events["date"] = pd.to_datetime(master_events["date"], errors="raise")
    ok("datas parseaveis")

    if master_events["event_id"].duplicated().any():
        raise ValueError("ERRO: event_id duplicado em events_master.csv.")
    if master_events["date"].duplicated().any():
        raise ValueError("ERRO: datas duplicadas em events_master.csv.")
    ok("sem duplicidade")

    expected_regime = master_events["date"].apply(
        lambda date: "core" if date <= CORE_END_DATE else ("pandemic" if date <= PANDEMIC_END_DATE else "post")
    )
    if not expected_regime.equals(master_events["event_regime"]):
        raise ValueError("ERRO: event_regime nao esta coerente com as datas.")
    ok("regimes corretos")

    if not master_events["is_core_sample"].equals(master_events["event_regime"].eq("core")):
        raise ValueError("ERRO: is_core_sample incoerente.")
    if not master_events["is_pandemic_period"].equals(master_events["event_regime"].eq("pandemic")):
        raise ValueError("ERRO: is_pandemic_period incoerente.")
    if not master_events["is_post_pandemic"].equals(master_events["event_regime"].eq("post")):
        raise ValueError("ERRO: is_post_pandemic incoerente.")
    if not master_events["include_in_primary_analysis"].equals(master_events["is_core_sample"]):
        raise ValueError("ERRO: include_in_primary_analysis incoerente.")
    ok("flags coerentes")

    subsets = {
        "core": pd.read_csv(CORE_FILE),
        "pandemic": pd.read_csv(PANDEMIC_FILE),
        "post": pd.read_csv(POST_FILE),
    }
    for regime_name, subset in subsets.items():
        if len(subset) != EXPECTED_COUNTS[regime_name]:
            raise ValueError(
                f"ERRO: subconjunto {regime_name} deveria ter {EXPECTED_COUNTS[regime_name]} eventos, "
                f"mas encontrou {len(subset)}"
            )
        if not subset["event_regime"].eq(regime_name).all():
            raise ValueError(f"ERRO: subconjunto {regime_name} contem regime incorreto.")
    ok("subconjuntos coerentes")

    print("\nVALIDACAO CONCLUIDA")


if __name__ == "__main__":
    main()
