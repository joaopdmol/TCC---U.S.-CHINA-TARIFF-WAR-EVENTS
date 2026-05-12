from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DATA"

FORMAL_CAR_MASTER_FILE = DATA_DIR / "formal_car_master.csv"
ABNORMAL_RETURNS_MASTER_FILE = DATA_DIR / "abnormal_returns_master.csv"
EVENT_WINDOWS_MASTER_FILE = DATA_DIR / "event_windows_master.csv"
EVENTS_MASTER_FILE = DATA_DIR / "events_master.csv"
MASTER_COVERAGE_FILE = DATA_DIR / "event_windows_master_coverage.csv"
OUTPUT_FILE = DATA_DIR / "ml_dataset.csv"

WINDOW_SIZE_MAP = {
    "m1_p1": 1,
    "m3_p3": 3,
    "m5_p5": 5,
}
NEUTRAL_THRESHOLD = 0.005


def load_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    dataframe = pd.read_csv(path)
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes em {path.name}: {missing_columns}")
    return dataframe


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def classify_three_class(value: float) -> str:
    if pd.isna(value):
        return "missing"
    if value < -NEUTRAL_THRESHOLD:
        return "negative"
    if value > NEUTRAL_THRESHOLD:
        return "positive"
    return "neutral"


def main() -> None:
    formal_car = load_csv(
        FORMAL_CAR_MASTER_FILE,
        [
            "event_id",
            "event_group",
            "event_type",
            "description",
            "event_regime",
            "is_core_sample",
            "is_pandemic_period",
            "is_post_pandemic",
            "include_in_primary_analysis",
            "window_label",
            "formal_car_sp500",
            "n_obs",
            "estimation_n_obs",
            "estimation_sufficient_data",
            "volatility_sp500",
        ],
    )
    events = load_csv(EVENTS_MASTER_FILE, ["event_id", "date"])
    coverage = load_csv(
        MASTER_COVERAGE_FILE,
        [
            "event_id",
            "window_label",
            "anchor_date",
            "anchor_shift_calendar_days",
            "is_complete",
        ],
    )
    abnormal_returns = load_csv(
        ABNORMAL_RETURNS_MASTER_FILE,
        ["event_id", "window_label", "abnormal_return"],
    )
    event_windows = load_csv(
        EVENT_WINDOWS_MASTER_FILE,
        ["event_id", "window_label", "relative_day", "market_date"],
    )

    if formal_car.duplicated(subset=["event_id", "window_label"]).any():
        raise ValueError("Duplicidade inesperada em formal_car_master.csv.")

    complete_coverage = coverage.loc[as_bool(coverage["is_complete"])].copy()
    complete_coverage = complete_coverage.drop_duplicates(subset=["event_id", "window_label"])

    available_windows = event_windows[["event_id", "window_label"]].drop_duplicates()
    formal_car = formal_car.merge(
        available_windows.assign(has_event_window=True),
        on=["event_id", "window_label"],
        how="left",
    )
    if not formal_car["has_event_window"].fillna(False).all():
        missing = formal_car.loc[formal_car["has_event_window"].isna(), ["event_id", "window_label"]]
        raise ValueError(f"Eventos sem janela de mercado correspondente: {missing.to_dict('records')}")

    dataset = formal_car.merge(
        events[["event_id", "date"]].rename(columns={"date": "event_date"}),
        on="event_id",
        how="left",
    ).merge(
        complete_coverage[
            [
                "event_id",
                "window_label",
                "anchor_date",
                "anchor_shift_calendar_days",
            ]
        ],
        on=["event_id", "window_label"],
        how="left",
    )

    dataset["event_date"] = pd.to_datetime(dataset["event_date"], errors="coerce")
    dataset["anchor_date"] = pd.to_datetime(dataset["anchor_date"], errors="coerce")
    if dataset["event_date"].isna().any():
        raise ValueError("Ha event_date invalida no dataset de ML.")
    if dataset["anchor_date"].isna().any():
        raise ValueError("Ha anchor_date invalida no dataset de ML.")

    abnormal_counts = (
        abnormal_returns.groupby(["event_id", "window_label"], observed=True)
        .agg(abnormal_return_rows=("abnormal_return", "count"))
        .reset_index()
    )
    dataset = dataset.merge(abnormal_counts, on=["event_id", "window_label"], how="left")

    dataset["target_negative_impact"] = (dataset["formal_car_sp500"] < 0).astype(int)
    dataset["target_three_class"] = dataset["formal_car_sp500"].apply(classify_three_class)

    dataset["feature_window_size"] = dataset["window_label"].map(WINDOW_SIZE_MAP)
    dataset["feature_event_year"] = dataset["event_date"].dt.year
    dataset["feature_event_month"] = dataset["event_date"].dt.month
    dataset["feature_window_label"] = dataset["window_label"]
    dataset["feature_event_group"] = dataset["event_group"]
    dataset["feature_event_type"] = dataset["event_type"]
    dataset["feature_event_regime"] = dataset["event_regime"]
    dataset["feature_volatility_sp500"] = dataset["volatility_sp500"]
    dataset["feature_n_obs"] = dataset["n_obs"]
    dataset["feature_estimation_n_obs"] = dataset["estimation_n_obs"]
    dataset["feature_estimation_sufficient_data"] = as_bool(
        dataset["estimation_sufficient_data"]
    ).astype(int)
    dataset["feature_anchor_shift_calendar_days"] = dataset["anchor_shift_calendar_days"]
    dataset["feature_is_core_sample"] = as_bool(dataset["is_core_sample"]).astype(int)
    dataset["feature_is_pandemic_period"] = as_bool(dataset["is_pandemic_period"]).astype(int)
    dataset["feature_is_post_pandemic"] = as_bool(dataset["is_post_pandemic"]).astype(int)
    dataset["feature_include_in_primary_analysis"] = as_bool(
        dataset["include_in_primary_analysis"]
    ).astype(int)

    # O CAR formal e usado apenas para construir o alvo; ele nao e salvo como feature.
    output_columns = [
        "event_id",
        "event_date",
        "anchor_date",
        "window_label",
        "event_group",
        "event_type",
        "event_regime",
        "description",
        "target_negative_impact",
        "target_three_class",
        "feature_window_label",
        "feature_event_group",
        "feature_event_type",
        "feature_event_regime",
        "feature_volatility_sp500",
        "feature_n_obs",
        "feature_estimation_n_obs",
        "feature_estimation_sufficient_data",
        "feature_anchor_shift_calendar_days",
        "feature_window_size",
        "feature_event_year",
        "feature_event_month",
        "feature_is_core_sample",
        "feature_is_pandemic_period",
        "feature_is_post_pandemic",
        "feature_include_in_primary_analysis",
        "abnormal_return_rows",
    ]
    dataset = dataset[output_columns].sort_values(["event_id", "window_label"]).reset_index(drop=True)

    if dataset["target_negative_impact"].isna().any():
        raise ValueError("Target binario com valores ausentes.")
    if dataset["target_negative_impact"].nunique() != 2:
        raise ValueError("Target binario precisa conter duas classes.")
    if dataset.filter(regex=r"^feature_").isna().all().any():
        empty_features = dataset.filter(regex=r"^feature_").columns[
            dataset.filter(regex=r"^feature_").isna().all()
        ].tolist()
        raise ValueError(f"Features totalmente vazias: {empty_features}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_FILE, index=False)

    print(f"Arquivo salvo em: {OUTPUT_FILE}")
    print(f"Linhas: {len(dataset)}")
    print(f"Eventos cobertos: {dataset['event_id'].nunique()}")
    print("Distribuicao do alvo:")
    print(dataset["target_negative_impact"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
