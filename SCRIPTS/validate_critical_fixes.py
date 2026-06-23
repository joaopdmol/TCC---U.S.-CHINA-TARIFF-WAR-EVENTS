"""
validate_critical_fixes.py
TDD validator for the critical-fix analytical layer.
All checks must pass before the thesis is considered production-ready.

Checks:
  1.  New CSVs exist and have expected schemas.
  2.  N=24 coherence in expanded-sample outputs.
  3.  N=17 coherence in core-sample references.
  4.  Correlation CSV has Pearson + Spearman + p-values.
  5.  Threshold sensitivity CSV contains 0.25%, 0.50%, 0.75%.
  6.  Group comparison CSV has Mann-Whitney and Welch t-test per window.
  7.  Core-vs-expanded CSV has 3 rows (one per window).
  8.  Original data NOT modified.
  9.  All previously passing validators still importable.
 10.  Key numerical spot-checks (reproducibility).
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DATA"
SCRIPTS_DIR = BASE_DIR / "SCRIPTS"

# -----------------------------------------------------------------------
# Expected file definitions
# -----------------------------------------------------------------------
EXPECTED_FILES = {
    "car_volatility_correlation_expanded.csv": [
        "scope", "window", "n_obs",
        "pearson_r", "pearson_p", "pearson_sig",
        "spearman_r", "spearman_p", "spearman_sig",
    ],
    "signal_threshold_sensitivity_full.csv": [
        "sample", "window", "threshold",
        "n_positive", "n_neutral", "n_negative", "n_total",
        "share_positive", "share_negative",
    ],
    "group_comparison_tests_expanded.csv": [
        "window", "window_label",
        "group_a", "group_b",
        "n_escalation", "n_relief",
        "mean_escalation", "mean_relief",
        "test_method", "statistic", "p_value", "significant_5pct",
    ],
    "core_vs_expanded_summary.csv": [
        "window", "window_label",
        "n_core", "n_expanded",
        "mean_car_core", "mean_car_expanded",
        "median_car_core", "median_car_expanded",
    ],
}

WINDOW_ORDER = ["m1_p1", "m3_p3", "m5_p5"]
WINDOW_DISPLAY = ["[-1,+1]", "[-3,+3]", "[-5,+5]"]
EXPECTED_THRESHOLDS = {"0.25%", "0.50%", "0.75%"}
EXPECTED_TESTS = {"welch_ttest", "mannwhitney_u"}


def ok(msg: str) -> None:
    print(f"OK  {msg}")


def fail(msg: str) -> None:
    raise AssertionError(f"FAIL  {msg}")


# -----------------------------------------------------------------------
# Check 1: Files exist and have expected columns
# -----------------------------------------------------------------------
def check_files_exist_and_columns() -> None:
    for filename, expected_cols in EXPECTED_FILES.items():
        path = DATA_DIR / filename
        if not path.exists():
            fail(f"{filename} does not exist")
        df = pd.read_csv(path)
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            fail(f"{filename} missing columns: {missing}")
    ok("All new CSVs exist with expected columns")


# -----------------------------------------------------------------------
# Check 2: N=24 coherence in expanded-sample outputs
# -----------------------------------------------------------------------
def check_n24_coherence() -> None:
    # group comparison should have escalation=13, relief=10
    grp = pd.read_csv(DATA_DIR / "group_comparison_tests_expanded.csv")
    esc_vals = grp["n_escalation"].unique()
    rel_vals = grp["n_relief"].unique()
    if not (esc_vals == 13).all():
        fail(f"n_escalation should be 13 everywhere; got {esc_vals}")
    if not (rel_vals == 10).all():
        fail(f"n_relief should be 10 everywhere; got {rel_vals}")

    # core_vs_expanded: n_expanded must be 24
    cve = pd.read_csv(DATA_DIR / "core_vs_expanded_summary.csv")
    if not (cve["n_expanded"] == 24).all():
        fail(f"n_expanded should be 24 everywhere; got {cve['n_expanded'].unique()}")
    ok("N=24 coherence verified (escalation=13, relief=10, expanded=24)")


# -----------------------------------------------------------------------
# Check 3: N=17 coherence for core sample
# -----------------------------------------------------------------------
def check_n17_coherence() -> None:
    cve = pd.read_csv(DATA_DIR / "core_vs_expanded_summary.csv")
    if not (cve["n_core"] == 17).all():
        fail(f"n_core should be 17 everywhere; got {cve['n_core'].unique()}")
    ok("N=17 coherence verified (core sample)")


# -----------------------------------------------------------------------
# Check 4: Correlation CSV has Pearson + Spearman p-values in [0,1]
# -----------------------------------------------------------------------
def check_correlation_structure() -> None:
    corr = pd.read_csv(DATA_DIR / "car_volatility_correlation_expanded.csv")
    for col in ["pearson_p", "spearman_p"]:
        vals = corr[col].dropna()
        if vals.empty:
            fail(f"Correlation CSV: {col} is empty")
        if not ((vals >= 0) & (vals <= 1)).all():
            fail(f"Correlation CSV: {col} has values outside [0,1]")
    # Should have 4 rows: pooled + 3 windows
    if len(corr) != 4:
        fail(f"Correlation CSV should have 4 rows, got {len(corr)}")
    ok("Correlation CSV structure valid (4 rows, p-values in [0,1])")


# -----------------------------------------------------------------------
# Check 5: Threshold sensitivity has 0.25%, 0.50%, 0.75%
# -----------------------------------------------------------------------
def check_threshold_sensitivity() -> None:
    thr = pd.read_csv(DATA_DIR / "signal_threshold_sensitivity_full.csv")
    found = set(thr["threshold"].unique())
    missing = EXPECTED_THRESHOLDS - found
    if missing:
        fail(f"Threshold sensitivity missing thresholds: {missing}")
    # Full-sample rows: 3 windows × 3 thresholds = 9
    full_rows = thr[thr["sample"] == "full (N=24)"]
    if len(full_rows) != 9:
        fail(f"Expected 9 full-sample rows (3w × 3t), got {len(full_rows)}")
    ok(f"Threshold sensitivity has all expected thresholds: {EXPECTED_THRESHOLDS}")


# -----------------------------------------------------------------------
# Check 6: Group comparison has Mann-Whitney + Welch t-test per window
# -----------------------------------------------------------------------
def check_group_comparison() -> None:
    grp = pd.read_csv(DATA_DIR / "group_comparison_tests_expanded.csv")
    found_tests = set(grp["test_method"].unique())
    missing = EXPECTED_TESTS - found_tests
    if missing:
        fail(f"Group comparison CSV missing tests: {missing}")
    # Should have 3 windows × 2 tests = 6 rows
    if len(grp) != 6:
        fail(f"Expected 6 rows (3 windows × 2 tests), got {len(grp)}")
    # All p-values in [0,1]
    pvals = grp["p_value"].dropna()
    if not ((pvals >= 0) & (pvals <= 1)).all():
        fail("Group comparison p_value has values outside [0,1]")
    ok("Group comparison CSV: Mann-Whitney + Welch t-test present for all windows")


# -----------------------------------------------------------------------
# Check 7: Core-vs-expanded has 3 rows
# -----------------------------------------------------------------------
def check_core_vs_expanded() -> None:
    cve = pd.read_csv(DATA_DIR / "core_vs_expanded_summary.csv")
    if len(cve) != 3:
        fail(f"Core-vs-expanded should have 3 rows, got {len(cve)}")
    found_windows = set(cve["window_label"].unique())
    expected = set(WINDOW_ORDER)
    if found_windows != expected:
        fail(f"Core-vs-expanded windows mismatch: expected {expected}, got {found_windows}")
    ok("Core-vs-expanded CSV has 3 rows (one per window)")


# -----------------------------------------------------------------------
# Check 8: Original data not modified
# -----------------------------------------------------------------------
def check_original_data_intact() -> None:
    master = DATA_DIR / "formal_car_master.csv"
    if not master.exists():
        fail("formal_car_master.csv not found")
    df = pd.read_csv(master)
    if len(df) != 72:
        fail(f"formal_car_master.csv should have 72 rows, got {len(df)}")
    e08 = df.loc[(df["event_id"] == "E08") & (df["window_label"] == "m3_p3"), "formal_car_sp500"]
    if e08.empty:
        fail("E08/m3_p3 not found in formal_car_master.csv")
    if abs(float(e08.values[0]) - (-0.1102467244752972)) > 1e-8:
        fail(f"E08/m3_p3 value changed: {float(e08.values[0])}")
    ok("Original data intact (formal_car_master.csv, E08 spot-check)")


# -----------------------------------------------------------------------
# Check 9: Prior validators still importable
# -----------------------------------------------------------------------
def check_prior_validators_importable() -> None:
    import importlib.util, sys

    prior = [
        "validate_robust_car_statistics",
    ]
    sys.path.insert(0, str(SCRIPTS_DIR))
    for mod_name in prior:
        spec = importlib.util.spec_from_file_location(mod_name, SCRIPTS_DIR / f"{mod_name}.py")
        if spec is None:
            fail(f"Cannot locate {mod_name}.py")
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:
            fail(f"{mod_name} raised on import: {exc}")
    ok("Prior validators importable without errors")


# -----------------------------------------------------------------------
# Check 10: Numerical spot-checks
# -----------------------------------------------------------------------
def check_numerical_spot_checks() -> None:
    # Pooled Pearson should be negative (CAR tends negative when vol is high)
    corr = pd.read_csv(DATA_DIR / "car_volatility_correlation_expanded.csv")
    pooled = corr[corr["window"] == "all"].iloc[0]
    if float(pooled["pearson_r"]) >= 0:
        fail(f"Expected pooled Pearson r < 0, got {pooled['pearson_r']}")

    # Group comparison: none should be significant at 5%
    grp = pd.read_csv(DATA_DIR / "group_comparison_tests_expanded.csv")
    sig_rows = grp[grp["significant_5pct"] == True]
    if len(sig_rows) > 0:
        fail(
            f"Expected no significant group comparisons, but found: "
            f"{sig_rows[['window','test_method','p_value']].to_dict('records')}"
        )

    # Core expanded: mean in [-5,+5] should be positive for both samples
    cve = pd.read_csv(DATA_DIR / "core_vs_expanded_summary.csv")
    row55 = cve[cve["window_label"] == "m5_p5"].iloc[0]
    if float(row55["mean_car_core"]) < 0:
        fail(f"Core mean in [-5,+5] should be positive, got {row55['mean_car_core']}")
    if float(row55["mean_car_expanded"]) < 0:
        fail(f"Expanded mean in [-5,+5] should be positive, got {row55['mean_car_expanded']}")

    ok("Numerical spot-checks passed")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main() -> None:
    check_files_exist_and_columns()
    check_n24_coherence()
    check_n17_coherence()
    check_correlation_structure()
    check_threshold_sensitivity()
    check_group_comparison()
    check_core_vs_expanded()
    check_original_data_intact()
    check_prior_validators_importable()
    check_numerical_spot_checks()
    print("\nVALIDACOES CONCLUIDAS — todos os checks passaram.")


if __name__ == "__main__":
    main()
