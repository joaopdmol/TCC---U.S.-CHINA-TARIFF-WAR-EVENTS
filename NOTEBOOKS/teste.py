from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
EVENTS_PATH = BASE_DIR / "DATA" / "events.csv"

events = pd.read_csv(EVENTS_PATH)
events["date"] = pd.to_datetime(events["date"])

print(events)
