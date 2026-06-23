from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import (
    binomtest,
    iqr,
    median_abs_deviation,
    ttest_1samp,
    trim_mean,
    wilcoxon,
)
from scipy.stats.mstats import winsorize as scipy_winsorize


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DATA"

INPUT_FILE = DATA_DIR / "formal_car_master.csv"
OUTPUT_SUMMARY = DATA_DIR / "robust_car_summary.csv"
OUTPUT_BY_GROUP = DATA_DIR / "robust_car_by_group.csv"
OUTPUT_LOO = DATA_DIR / "leave_one_out_car_influence.csv"

WINDOW_ORDER = ["m1_p1", "m3_p3", "m5_p5"]
NEUTRAL_THRESHOLD = 0.005
TRIM_PROPORTION = 0.10
WINSOR_LIMITS = [0.10, 0.10]
BOOTSTRAP_N_RESAMPLES = 5000
BOOTSTRAP_SEED = 42
MIN_N_INFERENCE = 5
GROUP_ORDER = ["escalation", "relief", "structural"]


def load_formal_car_master() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {INPUT_FILE}. "
            "Rode primeiro build_event_study_master.py."
        )
    df = pd.read_csv(INPUT_FILE)
    required = ["event_id", "event_group", "window_label", "formal_car_sp500"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes em formal_car_master.csv: {missing}")
    return df


def _bootstrap_ci_mean(
    values: np.ndarray,
    n_boot: int = BOOTSTRAP_N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boots = np.array(
        [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    )
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _bootstrap_ci_median(
    values: np.ndarray,
    n_boot: int = BOOTSTRAP_N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    if len(values) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array(
        [np.median(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    )
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _ttest_pvalue(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    result = ttest_1samp(values, 0.0, nan_policy="omit")
    return float(result.pvalue)


def _wilcoxon_pvalue(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    nonzero = values[values != 0.0]
    if len(nonzero) < 2:
        return float("nan")
    try:
        result = wilcoxon(values, zero_method="wilcox", correction=False, alternative="two-sided")
        return float(result.pvalue)
    except Exception:
        return float("nan")


def _sign_test_pvalue(values: np.ndarray) -> float:
    n_pos = int((values > 0).sum())
    n_neg = int((values < 0).sum())
    n_nonzero = n_pos + n_neg
    if n_nonzero == 0:
        return float("nan")
    result = binomtest(n_pos, n_nonzero, 0.5, alternative="two-sided")
    return float(result.pvalue)


def compute_robust_stats_for_window(
    values: np.ndarray,
    n_boot: int = BOOTSTRAP_N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)

    if n == 0:
        return {"n": 0}

    mean_car = float(np.mean(values))
    median_car = float(np.median(values))
    std_car = float(np.std(values, ddof=1)) if n >= 2 else float("nan")
    iqr_car = float(iqr(values))
    mad_car = float(median_abs_deviation(values))

    trimmed_mean_car = float(trim_mean(values, TRIM_PROPORTION)) if n >= MIN_N_INFERENCE else float("nan")
    winsor_arr = np.array(scipy_winsorize(values, limits=WINSOR_LIMITS))
    winsorized_mean_car = float(np.mean(winsor_arr))

    min_car = float(np.min(values))
    max_car = float(np.max(values))

    ci_mean_lo, ci_mean_hi = _bootstrap_ci_mean(values, n_boot, seed)
    ci_med_lo, ci_med_hi = _bootstrap_ci_median(values, n_boot, seed)

    run_tests = n >= MIN_N_INFERENCE
    ttest_pvalue = _ttest_pvalue(values) if run_tests else float("nan")
    wilcoxon_pvalue = _wilcoxon_pvalue(values) if run_tests else float("nan")
    sign_test_pvalue = _sign_test_pvalue(values) if run_tests else float("nan")

    n_positive = int((values > NEUTRAL_THRESHOLD).sum())
    n_negative = int((values < -NEUTRAL_THRESHOLD).sum())
    n_neutral = n - n_positive - n_negative

    return {
        "n": n,
        "mean_car": mean_car,
        "median_car": median_car,
        "std_car": std_car,
        "iqr_car": iqr_car,
        "mad_car": mad_car,
        "trimmed_mean_car": trimmed_mean_car,
        "winsorized_mean_car": winsorized_mean_car,
        "min_car": min_car,
        "max_car": max_car,
        "bootstrap_ci_mean_lower": ci_mean_lo,
        "bootstrap_ci_mean_upper": ci_mean_hi,
        "bootstrap_ci_median_lower": ci_med_lo,
        "bootstrap_ci_median_upper": ci_med_hi,
        "ttest_pvalue": ttest_pvalue,
        "wilcoxon_pvalue": wilcoxon_pvalue,
        "sign_test_pvalue": sign_test_pvalue,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_neutral": n_neutral,
        "share_positive": n_positive / n,
        "share_negative": n_negative / n,
    }


def build_robust_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window in WINDOW_ORDER:
        values = df.loc[df["window_label"] == window, "formal_car_sp500"].dropna().values
        stats = compute_robust_stats_for_window(values)
        stats["window_label"] = window
        rows.append(stats)

    summary = pd.DataFrame(rows)
    col_order = [
        "window_label", "n", "mean_car", "median_car", "std_car", "iqr_car", "mad_car",
        "trimmed_mean_car", "winsorized_mean_car", "min_car", "max_car",
        "bootstrap_ci_mean_lower", "bootstrap_ci_mean_upper",
        "bootstrap_ci_median_lower", "bootstrap_ci_median_upper",
        "ttest_pvalue", "wilcoxon_pvalue", "sign_test_pvalue",
        "n_positive", "n_negative", "n_neutral", "share_positive", "share_negative",
    ]
    return summary[col_order]


def build_robust_by_group(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    present_groups = df["event_group"].unique()
    for group in GROUP_ORDER:
        if group not in present_groups:
            continue
        for window in WINDOW_ORDER:
            mask = (df["event_group"] == group) & (df["window_label"] == window)
            values = df.loc[mask, "formal_car_sp500"].dropna().values
            stats = compute_robust_stats_for_window(values)
            stats["event_group"] = group
            stats["window_label"] = window
            rows.append(stats)

    by_group = pd.DataFrame(rows)
    col_order = [
        "event_group", "window_label", "n", "mean_car", "median_car",
        "trimmed_mean_car", "winsorized_mean_car", "iqr_car", "mad_car",
        "bootstrap_ci_mean_lower", "bootstrap_ci_mean_upper",
        "bootstrap_ci_median_lower", "bootstrap_ci_median_upper",
        "ttest_pvalue", "wilcoxon_pvalue", "sign_test_pvalue",
        "n_positive", "n_negative", "n_neutral", "share_positive", "share_negative",
    ]
    return by_group[col_order]


def build_leave_one_out(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    event_ids = sorted(df["event_id"].unique())
    for window in WINDOW_ORDER:
        subset = df.loc[df["window_label"] == window, ["event_id", "formal_car_sp500"]].dropna()
        full_values = subset["formal_car_sp500"].values
        full_mean = float(np.mean(full_values))
        full_median = float(np.median(full_values))
        for event_id in event_ids:
            loo_values = subset.loc[subset["event_id"] != event_id, "formal_car_sp500"].values
            if len(loo_values) == 0:
                continue
            loo_mean = float(np.mean(loo_values))
            loo_median = float(np.median(loo_values))
            rows.append(
                {
                    "window_label": window,
                    "excluded_event_id": event_id,
                    "n_remaining": len(loo_values),
                    "full_mean_car": full_mean,
                    "loo_mean_car": loo_mean,
                    "delta_mean_car": loo_mean - full_mean,
                    "full_median_car": full_median,
                    "loo_median_car": loo_median,
                    "delta_median_car": loo_median - full_median,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    df = load_formal_car_master()

    summary = build_robust_summary(df)
    by_group = build_robust_by_group(df)
    loo = build_leave_one_out(df)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_SUMMARY, index=False)
    by_group.to_csv(OUTPUT_BY_GROUP, index=False)
    loo.to_csv(OUTPUT_LOO, index=False)

    print(f"Arquivo salvo em: {OUTPUT_SUMMARY}")
    print(f"Arquivo salvo em: {OUTPUT_BY_GROUP}")
    print(f"Arquivo salvo em: {OUTPUT_LOO}")

    display_cols = [
        "window_label", "n", "mean_car", "median_car", "trimmed_mean_car",
        "winsorized_mean_car", "bootstrap_ci_mean_lower", "bootstrap_ci_mean_upper",
        "wilcoxon_pvalue", "sign_test_pvalue",
    ]
    print("\nResumo robusto por janela:")
    print(summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
