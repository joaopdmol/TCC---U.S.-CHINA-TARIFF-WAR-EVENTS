from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DATA"
GENERATED_DIR = BASE_DIR / "partes" / "generated"

MARKET_FEATURES_FILE = DATA_DIR / "market_features.csv"
EVENT_WINDOWS_MASTER_FILE = DATA_DIR / "event_windows_master.csv"
FORMAL_CAR_CANDIDATES = [
    DATA_DIR / "formal_car_master.csv",
    DATA_DIR / "formal_car.csv",
    DATA_DIR / "formal_car_by_event.csv",
]

PLACEBO_SUMMARY_FILE = DATA_DIR / "placebo_summary.csv"
PLACEBO_TABLE_FILE = GENERATED_DIR / "placebo_summary_table.tex"

WINDOW_SPECS = {
    "m1_p1": 1,
    "m3_p3": 3,
    "m5_p5": 5,
}
WINDOW_LABELS = {
    "m1_p1": "[-1,+1]",
    "m3_p3": "[-3,+3]",
    "m5_p5": "[-5,+5]",
}
ESTIMATION_START_OFFSET = -30
ESTIMATION_END_OFFSET = -6
MIN_ESTIMATION_OBS = 20
PLACEBO_TARGET_N = 500
SEED = 42
BANNED_LATEX = ["booktabs", "tabularx", "longtable", "\\toprule", "\\midrule", "\\bottomrule"]


def load_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    dataframe = pd.read_csv(path)
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")
    return dataframe


def detect_formal_car_source() -> tuple[Path, pd.DataFrame, str]:
    for path in FORMAL_CAR_CANDIDATES:
        if not path.exists():
            continue
        dataframe = pd.read_csv(path)
        if {"event_id", "window_label"}.issubset(dataframe.columns) and "formal_car_sp500" in dataframe.columns:
            return path, dataframe, "formal_car_sp500"

        fallback_columns = [
            column
            for column in dataframe.columns
            if "car" in column.lower() and pd.api.types.is_numeric_dtype(dataframe[column])
        ]
        if {"event_id", "window_label"}.issubset(dataframe.columns) and fallback_columns:
            return path, dataframe, fallback_columns[0]

    raise FileNotFoundError("No valid formal CAR file was found.")


def build_sp500_history() -> pd.DataFrame:
    market = load_csv(MARKET_FEATURES_FILE, ["date", "sp500"])
    market["date"] = pd.to_datetime(market["date"], errors="raise")
    history = market.loc[market["sp500"].notna(), ["date", "sp500"]].copy()
    history = history.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    history["sp500_return"] = history["sp500"].pct_change(fill_method=None)
    return history


def get_real_event_dates() -> set[pd.Timestamp]:
    event_windows = load_csv(EVENT_WINDOWS_MASTER_FILE, ["event_date", "anchor_date"])
    event_windows["event_date"] = pd.to_datetime(event_windows["event_date"], errors="raise")
    event_windows["anchor_date"] = pd.to_datetime(event_windows["anchor_date"], errors="raise")
    event_dates = set(event_windows["event_date"].dropna().dt.normalize())
    anchor_dates = set(event_windows["anchor_date"].dropna().dt.normalize())
    return event_dates | anchor_dates


def select_placebo_dates(history: pd.DataFrame, real_dates: set[pd.Timestamp]) -> pd.DataFrame:
    trading_calendar = pd.Index(history["date"])
    eligible_rows = []

    for position, date in enumerate(trading_calendar):
        if date.normalize() in real_dates:
            continue
        start_position = position + ESTIMATION_START_OFFSET
        end_position = position + ESTIMATION_END_OFFSET
        max_window_position = position + max(WINDOW_SPECS.values())
        min_window_position = position - max(WINDOW_SPECS.values())

        if start_position < 0 or end_position < 0:
            continue
        if min_window_position < 0 or max_window_position >= len(trading_calendar):
            continue

        estimation_returns = history.iloc[start_position : end_position + 1]["sp500_return"].dropna()
        if len(estimation_returns) < MIN_ESTIMATION_OBS:
            continue

        eligible_rows.append(
            {
                "placebo_id": f"P{len(eligible_rows) + 1:04d}",
                "placebo_date": date,
                "calendar_position": position,
            }
        )

    eligible = pd.DataFrame(eligible_rows)
    if eligible.empty:
        raise ValueError("No eligible placebo dates were found.")

    sample_n = min(PLACEBO_TARGET_N, len(eligible))
    selected = eligible.sample(n=sample_n, random_state=SEED, replace=False).sort_values(
        "placebo_date"
    )
    return selected.reset_index(drop=True)


def compute_placebo_car(history: pd.DataFrame, placebo_dates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    trading_calendar = pd.Index(history["date"])
    returns = history.set_index("date")["sp500_return"]

    for placebo in placebo_dates.itertuples(index=False):
        anchor_position = int(placebo.calendar_position)
        estimation_slice = history.iloc[
            anchor_position + ESTIMATION_START_OFFSET : anchor_position + ESTIMATION_END_OFFSET + 1
        ]
        expected_return = estimation_slice["sp500_return"].dropna().mean()

        for window_label, window_size in WINDOW_SPECS.items():
            window_dates = [
                trading_calendar[anchor_position + offset]
                for offset in range(-window_size, window_size + 1)
            ]
            window_returns = returns.loc[window_dates].dropna()
            abnormal_returns = window_returns - expected_return
            placebo_car = abnormal_returns.sum()
            rows.append(
                {
                    "placebo_id": placebo.placebo_id,
                    "placebo_date": placebo.placebo_date,
                    "window_label": window_label,
                    "placebo_car_sp500": placebo_car,
                    "expected_return": expected_return,
                    "n_obs": int(len(abnormal_returns)),
                }
            )

    return pd.DataFrame(rows)


def build_summary(
    real_car: pd.DataFrame,
    car_column: str,
    placebo_car: pd.DataFrame,
) -> pd.DataFrame:
    observed = (
        real_car.loc[real_car["window_label"].isin(WINDOW_SPECS.keys())]
        .groupby("window_label", observed=True)
        .agg(
            observed_n=("event_id", "nunique"),
            observed_mean_car=(car_column, "mean"),
        )
        .reset_index()
    )

    placebo_summary = (
        placebo_car.groupby("window_label", observed=True)
        .agg(
            placebo_n=("placebo_id", "nunique"),
            placebo_mean_car=("placebo_car_sp500", "mean"),
            placebo_sd=("placebo_car_sp500", "std"),
        )
        .reset_index()
    )
    summary = observed.merge(placebo_summary, on="window_label", how="inner")

    p_values = []
    for row in summary.itertuples(index=False):
        placebo_values = placebo_car.loc[
            placebo_car["window_label"].eq(row.window_label),
            "placebo_car_sp500",
        ].dropna()
        empirical_pvalue = (
            (placebo_values.abs() >= abs(row.observed_mean_car)).mean()
            if len(placebo_values) > 0
            else np.nan
        )
        p_values.append(empirical_pvalue)

    summary["empirical_placebo_p_value"] = p_values
    summary["interpretation"] = summary["empirical_placebo_p_value"].apply(classify_interpretation)
    summary["event_window"] = summary["window_label"].map(WINDOW_LABELS)
    summary["window_label"] = pd.Categorical(
        summary["window_label"],
        categories=list(WINDOW_SPECS.keys()),
        ordered=True,
    )
    return summary.sort_values("window_label").reset_index(drop=True)


def classify_interpretation(p_value: float) -> str:
    if pd.isna(p_value):
        return "Not available"
    if p_value < 0.05:
        return "Unusual relative to placebo"
    if p_value < 0.10:
        return "Suggestive placebo evidence"
    return "Consistent with placebo variation"


def percent(value: float) -> str:
    return f"{value * 100:.2f}\\%"


def pvalue(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def write_latex_table(summary: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Placebo Robustness Check}",
        r"\label{tab:placebo_summary}",
        r"\begin{tabular}{p{2.0cm} p{2.0cm} p{2.0cm} p{1.7cm} p{2.1cm} p{3.2cm}}",
        r"\hline",
        r"Event window & Observed mean CAR & Placebo mean CAR & Placebo SD & Empirical placebo p-value & Interpretation \\",
        r"\hline",
    ]

    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.event_window} & {percent(row.observed_mean_car)} & "
            f"{percent(row.placebo_mean_car)} & {percent(row.placebo_sd)} & "
            f"{pvalue(row.empirical_placebo_p_value)} & {row.interpretation} \\\\"
        )

    lines.extend(
        [
            r"\hline",
            r"\multicolumn{6}{p{13.0cm}}{\footnotesize Notes: Placebo dates were sampled from the real S\&P 500 trading calendar using seed 42 and excluding real event dates and anchor dates. The empirical placebo p-value is the share of placebo CARs whose absolute value is at least as large as the absolute observed mean CAR.} \\",
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    PLACEBO_TABLE_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_tdd_checks(
    summary: pd.DataFrame,
    placebo_dates: pd.DataFrame,
    real_dates: set[pd.Timestamp],
) -> None:
    if not PLACEBO_SUMMARY_FILE.exists():
        raise AssertionError("DATA/placebo_summary.csv was not created.")
    if not PLACEBO_TABLE_FILE.exists():
        raise AssertionError("partes/generated/placebo_summary_table.tex was not created.")
    if len(summary) != 3:
        raise AssertionError(f"Expected three event-window rows, found {len(summary)}.")
    if placebo_dates["placebo_date"].dt.normalize().isin(real_dates).any():
        raise AssertionError("At least one placebo date coincides with a real event_date or anchor_date.")
    if summary.isna().any().any():
        raise AssertionError("placebo_summary.csv contains NaN.")

    latex = PLACEBO_TABLE_FILE.read_text(encoding="utf-8")
    for token in BANNED_LATEX:
        if token in latex:
            raise AssertionError(f"Banned LaTeX token found: {token}")
    if r"\label{tab:placebo_summary}" not in latex:
        raise AssertionError("Missing table label tab:placebo_summary.")
    for window_label in WINDOW_LABELS.values():
        if window_label not in latex:
            raise AssertionError(f"Missing event window in LaTeX table: {window_label}")


def main() -> None:
    source_path, real_car, car_column = detect_formal_car_source()
    print(f"Real CAR source used: {source_path.relative_to(BASE_DIR)}")
    print(f"Formal CAR column used: {car_column}")

    history = build_sp500_history()
    real_dates = get_real_event_dates()
    placebo_dates = select_placebo_dates(history, real_dates)
    placebo_car = compute_placebo_car(history, placebo_dates)
    summary = build_summary(real_car, car_column, placebo_car)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(PLACEBO_SUMMARY_FILE, index=False)
    write_latex_table(summary)
    run_tdd_checks(summary, placebo_dates, real_dates)

    print(f"Placebo dates selected: {placebo_dates['placebo_id'].nunique()}")
    print(f"Output CSV: {PLACEBO_SUMMARY_FILE.relative_to(BASE_DIR)}")
    print(f"Output LaTeX: {PLACEBO_TABLE_FILE.relative_to(BASE_DIR)}")
    print("TDD: OK DATA/placebo_summary.csv created")
    print("TDD: OK partes/generated/placebo_summary_table.tex created")
    print("TDD: OK three event-window rows")
    print("TDD: OK placebo dates do not coincide with real event dates or anchor dates")
    print("TDD: OK no NaN")
    print("TDD: OK LaTeX does not use tabularx/booktabs/toprule/midrule/bottomrule")
    print("\nPlacebo test results:")
    for row in summary.itertuples(index=False):
        print(
            f"{row.event_window}: Observed mean CAR={percent(row.observed_mean_car)}, "
            f"Placebo mean CAR={percent(row.placebo_mean_car)}, "
            f"Placebo SD={percent(row.placebo_sd)}, "
            f"Empirical p-value={pvalue(row.empirical_placebo_p_value)}, "
            f"Interpretation={row.interpretation}"
        )


if __name__ == "__main__":
    main()
