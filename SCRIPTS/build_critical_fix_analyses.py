"""
build_critical_fix_analyses.py
Generates supplementary analytical outputs needed to close methodological gaps
identified in the critical review. All inputs are derived from existing repository data.

Outputs:
  DATA/car_volatility_correlation_expanded.csv   -- Pearson/Spearman CAR x volatility (N=24)
  DATA/signal_threshold_sensitivity_full.csv     -- sign counts at 0.25/0.50/0.75% thresholds (N=24)
  DATA/group_comparison_tests_expanded.csv       -- Mann-Whitney / Welch t: escalation vs relief (N=24)
  DATA/core_vs_expanded_summary.csv             -- formatted Core vs Expanded comparison table
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DATA"

FORMAL_CAR_MASTER = DATA_DIR / "formal_car_master.csv"
VOLATILITY_BY_EVENT = DATA_DIR / "volatility_by_event.csv"

WINDOW_LABELS = ["m1_p1", "m3_p3", "m5_p5"]
WINDOW_DISPLAY = {"m1_p1": "[-1,+1]", "m3_p3": "[-3,+3]", "m5_p5": "[-5,+5]"}
THRESHOLDS = [0.0025, 0.0050, 0.0075]
THRESHOLD_LABELS = {0.0025: "0.25%", 0.0050: "0.50%", 0.0075: "0.75%"}


# ---------------------------------------------------------------------------
# 1. CAR x Volatility correlation (expanded sample, N=24 events, all 3 windows)
# ---------------------------------------------------------------------------

def build_car_volatility_correlation() -> pd.DataFrame:
    """Pearson and Spearman correlations between formal_car_sp500 and
    volatility_sp500, computed on the expanded market-covered sample (N=24).
    formal_car_master.csv already contains volatility_sp500 for all 72 obs.
    Reported for the pooled sample (72 obs) and per event window (24 obs each).
    """
    merged = pd.read_csv(FORMAL_CAR_MASTER)
    merged = merged.dropna(subset=["formal_car_sp500", "volatility_sp500"])

    records = []

    # Pooled
    x = merged["formal_car_sp500"].values
    y = merged["volatility_sp500"].values
    r_p, p_p = stats.pearsonr(x, y)
    r_s, p_s = stats.spearmanr(x, y)
    records.append({
        "scope": "pooled (all windows)",
        "window": "all",
        "n_obs": len(x),
        "pearson_r": round(r_p, 6),
        "pearson_p": round(p_p, 6),
        "pearson_sig": p_p < 0.05,
        "spearman_r": round(r_s, 6),
        "spearman_p": round(p_s, 6),
        "spearman_sig": p_s < 0.05,
    })

    # Per window
    for w in WINDOW_LABELS:
        sub = merged[merged["window_label"] == w].dropna(
            subset=["formal_car_sp500", "volatility_sp500"]
        )
        x = sub["formal_car_sp500"].values
        y = sub["volatility_sp500"].values
        if len(x) < 3:
            continue
        r_p, p_p = stats.pearsonr(x, y)
        r_s, p_s = stats.spearmanr(x, y)
        records.append({
            "scope": f"window {WINDOW_DISPLAY[w]}",
            "window": w,
            "n_obs": len(x),
            "pearson_r": round(r_p, 6),
            "pearson_p": round(p_p, 6),
            "pearson_sig": p_p < 0.05,
            "spearman_r": round(r_s, 6),
            "spearman_p": round(p_s, 6),
            "spearman_sig": p_s < 0.05,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. Signal threshold sensitivity (expanded N=24, full sample + by group)
# ---------------------------------------------------------------------------

def _classify(car_series: pd.Series, threshold: float) -> pd.Series:
    return pd.cut(
        car_series,
        bins=[-np.inf, -threshold, threshold, np.inf],
        labels=["negative", "neutral", "positive"],
    )


def build_signal_threshold_sensitivity() -> pd.DataFrame:
    """For each threshold and window, count positive/neutral/negative events
    in the full expanded sample (N=24) and by event group.
    """
    car = pd.read_csv(FORMAL_CAR_MASTER)

    records = []
    for w in WINDOW_LABELS:
        sub = car[car["window_label"] == w]
        for thr in THRESHOLDS:
            # Full sample
            labels = _classify(sub["formal_car_sp500"], thr)
            counts = labels.value_counts()
            n = len(sub)
            records.append({
                "sample": "full (N=24)",
                "window": WINDOW_DISPLAY[w],
                "threshold": THRESHOLD_LABELS[thr],
                "n_positive": int(counts.get("positive", 0)),
                "n_neutral": int(counts.get("neutral", 0)),
                "n_negative": int(counts.get("negative", 0)),
                "n_total": n,
                "share_positive": round(counts.get("positive", 0) / n, 4),
                "share_negative": round(counts.get("negative", 0) / n, 4),
            })
            # By group (escalation / relief only — structural N=1 skipped)
            for grp in ["escalation", "relief"]:
                grp_sub = sub[sub["event_group"] == grp]
                if len(grp_sub) == 0:
                    continue
                g_labels = _classify(grp_sub["formal_car_sp500"], thr)
                g_counts = g_labels.value_counts()
                n_grp = len(grp_sub)
                records.append({
                    "sample": f"{grp} (N={n_grp})",
                    "window": WINDOW_DISPLAY[w],
                    "threshold": THRESHOLD_LABELS[thr],
                    "n_positive": int(g_counts.get("positive", 0)),
                    "n_neutral": int(g_counts.get("neutral", 0)),
                    "n_negative": int(g_counts.get("negative", 0)),
                    "n_total": n_grp,
                    "share_positive": round(g_counts.get("positive", 0) / n_grp, 4),
                    "share_negative": round(g_counts.get("negative", 0) / n_grp, 4),
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 3. Group comparison tests (expanded: escalation N=13 vs relief N=10)
# ---------------------------------------------------------------------------

def build_group_comparison_tests_expanded() -> pd.DataFrame:
    """Per event window, compare escalation vs relief formal CAR using
    Welch t-test and Mann-Whitney U test.  Expanded sample (N=24):
    escalation=13, relief=10.  Structural (N=1) is excluded.
    """
    car = pd.read_csv(FORMAL_CAR_MASTER)

    records = []
    for w in WINDOW_LABELS:
        sub = car[car["window_label"] == w]
        esc = sub[sub["event_group"] == "escalation"]["formal_car_sp500"].dropna().values
        rel = sub[sub["event_group"] == "relief"]["formal_car_sp500"].dropna().values

        # Welch t-test
        t_stat, t_p = stats.ttest_ind(esc, rel, equal_var=False)
        # Mann-Whitney U
        u_stat, u_p = stats.mannwhitneyu(esc, rel, alternative="two-sided")

        for method, stat, pval in [
            ("welch_ttest", t_stat, t_p),
            ("mannwhitney_u", u_stat, u_p),
        ]:
            records.append({
                "window": WINDOW_DISPLAY[w],
                "window_label": w,
                "group_a": "escalation",
                "group_b": "relief",
                "n_escalation": len(esc),
                "n_relief": len(rel),
                "mean_escalation": round(float(np.mean(esc)), 8),
                "mean_relief": round(float(np.mean(rel)), 8),
                "median_escalation": round(float(np.median(esc)), 8),
                "median_relief": round(float(np.median(rel)), 8),
                "mean_diff_esc_minus_rel": round(float(np.mean(esc) - np.mean(rel)), 8),
                "test_method": method,
                "statistic": round(float(stat), 6),
                "p_value": round(float(pval), 6),
                "significant_5pct": bool(pval < 0.05),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. Core vs Expanded summary table
# ---------------------------------------------------------------------------

def build_core_vs_expanded_summary() -> pd.DataFrame:
    """Formatted comparison table: core (N=17) vs expanded (N=24) mean/median CAR."""
    src = DATA_DIR / "core_vs_full_comparison.csv"
    df = pd.read_csv(src)

    records = []
    for w in WINDOW_LABELS:
        core_row = df[(df["sample_label"] == "core_sample") & (df["window_label"] == w)].iloc[0]
        exp_row = df[
            (df["sample_label"] == "full_market_covered_sample") & (df["window_label"] == w)
        ].iloc[0]

        records.append({
            "window": WINDOW_DISPLAY[w],
            "window_label": w,
            "n_core": int(core_row["n_events"]),
            "n_expanded": int(exp_row["n_events"]),
            "mean_car_core": round(float(core_row["mean_formal_car_sp500"]) * 100, 4),
            "mean_car_expanded": round(float(exp_row["mean_formal_car_sp500"]) * 100, 4),
            "median_car_core": round(float(core_row["median_formal_car_sp500"]) * 100, 4),
            "median_car_expanded": round(float(exp_row["median_formal_car_sp500"]) * 100, 4),
            "diff_mean_pct": round(
                (float(exp_row["mean_formal_car_sp500"]) - float(core_row["mean_formal_car_sp500"]))
                * 100,
                4,
            ),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Building CAR x Volatility correlation (expanded N=24)...")
    corr = build_car_volatility_correlation()
    out = DATA_DIR / "car_volatility_correlation_expanded.csv"
    corr.to_csv(out, index=False)
    print(f"  Saved: {out.name}  ({len(corr)} rows)")
    print(corr[["scope", "n_obs", "pearson_r", "pearson_p", "spearman_r", "spearman_p"]].to_string(index=False))

    print("\nBuilding signal threshold sensitivity (expanded N=24)...")
    thr = build_signal_threshold_sensitivity()
    out = DATA_DIR / "signal_threshold_sensitivity_full.csv"
    thr.to_csv(out, index=False)
    print(f"  Saved: {out.name}  ({len(thr)} rows)")

    print("\nBuilding group comparison tests (expanded: escalation=13 vs relief=10)...")
    grp = build_group_comparison_tests_expanded()
    out = DATA_DIR / "group_comparison_tests_expanded.csv"
    grp.to_csv(out, index=False)
    print(f"  Saved: {out.name}  ({len(grp)} rows)")
    print(grp[["window", "test_method", "n_escalation", "n_relief", "p_value", "significant_5pct"]].to_string(index=False))

    print("\nBuilding Core vs Expanded summary table...")
    cve = build_core_vs_expanded_summary()
    out = DATA_DIR / "core_vs_expanded_summary.csv"
    cve.to_csv(out, index=False)
    print(f"  Saved: {out.name}  ({len(cve)} rows)")
    print(cve.to_string(index=False))

    print("\nAll critical-fix analyses completed.")


if __name__ == "__main__":
    main()
