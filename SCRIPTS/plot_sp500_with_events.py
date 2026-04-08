from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MARKET_FEATURES_FILE = BASE_DIR / "DATA" / "market_features.csv"
EVENTS_FILE = BASE_DIR / "DATA" / "events.csv"
OUTPUT_FILE = BASE_DIR / "FIGURES" / "sp500_with_events.png"

REQUIRED_MARKET_COLUMNS = ["date", "sp500"]
REQUIRED_EVENT_COLUMNS = ["date", "event_group", "event_id"]

EVENT_COLORS = {
    "escalation": "#B03A2E",
    "relief": "#1E8449",
    "implementacao": "#7D3C98",
    "acordo": "#117A65",
    "exclusao": "#B7950B",
}

EVENT_LABELS = {
    "escalation": "Eventos de escalada",
    "relief": "Eventos de alivio",
    "implementacao": "Eventos de implementacao",
    "acordo": "Eventos de acordo",
    "exclusao": "Eventos de exclusao",
}


def load_market_features() -> pd.DataFrame:
    if not MARKET_FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {MARKET_FEATURES_FILE}. "
            "Rode primeiro o script build_market_features.py."
        )

    market_features = pd.read_csv(MARKET_FEATURES_FILE)
    missing_columns = [column for column in REQUIRED_MARKET_COLUMNS if column not in market_features.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes em market_features.csv: {missing_columns}")

    market_features["date"] = pd.to_datetime(market_features["date"], errors="raise")
    return market_features.sort_values("date").reset_index(drop=True)


def load_events() -> pd.DataFrame:
    if not EVENTS_FILE.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {EVENTS_FILE}")

    events = pd.read_csv(EVENTS_FILE)
    missing_columns = [column for column in REQUIRED_EVENT_COLUMNS if column not in events.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes em events.csv: {missing_columns}")

    events["date"] = pd.to_datetime(events["date"], errors="raise")
    return events.sort_values("date").reset_index(drop=True)


def main() -> None:
    market_features = load_market_features()
    events = load_events()

    date_min = market_features["date"].min()
    date_max = market_features["date"].max()
    events_in_range = events[(events["date"] >= date_min) & (events["date"] <= date_max)].copy()

    if events_in_range.empty:
        raise ValueError("Nenhum evento encontrado dentro do intervalo da base de mercado.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(13, 7), dpi=300)

    ax.plot(
        market_features["date"],
        market_features["sp500"],
        color="#1F4E79",
        linewidth=2,
        label="S&P 500",
    )

    seen_groups = set()
    for event in events_in_range.itertuples(index=False):
        color = EVENT_COLORS.get(event.event_group, "#5D6D7E")
        label = EVENT_LABELS.get(event.event_group, event.event_group)
        show_label = label if event.event_group not in seen_groups else None

        ax.axvline(
            event.date,
            color=color,
            linestyle="--",
            linewidth=1.1,
            alpha=0.85,
            label=show_label,
        )
        seen_groups.add(event.event_group)

    ax.set_title(
        "S&P 500 e Eventos-Chave da Guerra Tarifaria EUA-China (2017-2020)",
        fontsize=14,
        pad=14,
    )
    ax.set_xlabel("Data")
    ax.set_ylabel("Fechamento do indice")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.margins(x=0.01)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)

    print(f"Grafico salvo em: {OUTPUT_FILE}")
    print(f"Eventos plotados: {len(events_in_range)}")


if __name__ == "__main__":
    main()
