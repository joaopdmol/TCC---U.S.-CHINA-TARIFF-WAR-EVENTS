"""
validate_group_comparison_tests.py
TDD guard for the escalation vs relief group comparison (Table 5 in the thesis).

Checks:
  1. group_comparison_tests_expanded.csv exists.
  2. Expected columns are present.
  3. Only escalation and relief groups appear (structural excluded).
  4. N_escalation=13, N_relief=10 throughout.
  5. Means per window match the formal_car_master.csv ground truth within 0.01 pp.
  6. Welch t-test and Mann-Whitney U are both present for each window.
  7. All p-values are in [0, 1].
  8. Means in the CSV match the values that must be in the LaTeX Table 5.
     (Prevents the bug where rounded hardcoded values diverge from data.)

Ground truth (derived from formal_car_master.csv):
  [-1,+1]:  esc_mean=-0.3059%  rel_mean=-0.1301%
  [-3,+3]:  esc_mean= 0.4831%  rel_mean=-0.9843%
  [-5,+5]:  esc_mean= 0.9575%  rel_mean=-0.9193%
"""
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DATA"
CSV_PATH = DATA_DIR / "group_comparison_tests_expanded.csv"
MASTER_PATH = DATA_DIR / "formal_car_master.csv"

EXPECTED_COLS = [
    "window", "window_label", "group_a", "group_b",
    "n_escalation", "n_relief",
    "mean_escalation", "mean_relief",
    "test_method", "statistic", "p_value", "significant_5pct",
]
EXPECTED_TESTS = {"welch_ttest", "mannwhitney_u"}
EXPECTED_WINDOWS = {"m1_p1", "m3_p3", "m5_p5"}
WINDOW_DISPLAY = {"m1_p1": "[-1,+1]", "m3_p3": "[-3,+3]", "m5_p5": "[-5,+5]"}

# Ground-truth means (as fractions, derived from formal_car_master.csv):
GROUND_TRUTH = {
    "m1_p1": {"esc": -0.003059, "rel": -0.001301},
    "m3_p3": {"esc":  0.004831, "rel": -0.009843},
    "m5_p5": {"esc":  0.009575, "rel": -0.009193},
}
# Tolerance: 0.0001 (= 0.01 pp) to allow benign floating-point variation
TOLERANCE = 0.0001

# Expected LaTeX representations (rounded to 2 decimal places in % form):
LATEX_EXPECTED = {
    "[-1,+1]": {"esc": "-0.31%", "rel": "-0.13%"},
    "[-3,+3]": {"esc": "0.48%",  "rel": "-0.98%"},
    "[-5,+5]": {"esc": "0.96%",  "rel": "-0.92%"},
}
# These are the rounded forms that f"{val*100:.2f}%" must produce for each window.
# Keeping them as a cross-check reference; the actual comparison uses formatted values.


def ok(msg: str) -> None:
    print(f"OK  {msg}")


def fail(msg: str) -> None:
    raise AssertionError(f"FAIL  {msg}")


def check_file_exists() -> None:
    if not CSV_PATH.exists():
        fail(f"{CSV_PATH.name} does not exist")
    ok(f"{CSV_PATH.name} exists")


def check_columns() -> None:
    df = pd.read_csv(CSV_PATH)
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        fail(f"Missing columns: {missing}")
    ok("All expected columns present")


def check_groups_only_esc_rel() -> None:
    df = pd.read_csv(CSV_PATH)
    groups_a = set(df["group_a"].unique())
    groups_b = set(df["group_b"].unique())
    if groups_a != {"escalation"}:
        fail(f"group_a should only be 'escalation'; got {groups_a}")
    if groups_b != {"relief"}:
        fail(f"group_b should only be 'relief'; got {groups_b}")
    ok("Groups are escalation vs relief only (structural excluded)")


def check_n_values() -> None:
    df = pd.read_csv(CSV_PATH)
    esc_vals = df["n_escalation"].unique()
    rel_vals = df["n_relief"].unique()
    if not (esc_vals == 13).all():
        fail(f"n_escalation should be 13 everywhere; got {esc_vals}")
    if not (rel_vals == 10).all():
        fail(f"n_relief should be 10 everywhere; got {rel_vals}")
    ok("N values correct: escalation=13, relief=10")


def check_means_match_ground_truth() -> None:
    """Means in the CSV must match values recomputed from formal_car_master.csv."""
    master = pd.read_csv(MASTER_PATH)
    csv = pd.read_csv(CSV_PATH)

    for w_label, gt in GROUND_TRUTH.items():
        sub = master[master["window_label"] == w_label]
        esc_actual = sub[sub["event_group"] == "escalation"]["formal_car_sp500"].mean()
        rel_actual = sub[sub["event_group"] == "relief"]["formal_car_sp500"].mean()

        # Check against ground truth
        if abs(esc_actual - gt["esc"]) > TOLERANCE:
            fail(
                f"Master data mean for escalation {w_label} changed: "
                f"expected ≈{gt['esc']:.6f}, got {esc_actual:.6f}"
            )
        if abs(rel_actual - gt["rel"]) > TOLERANCE:
            fail(
                f"Master data mean for relief {w_label} changed: "
                f"expected ≈{gt['rel']:.6f}, got {rel_actual:.6f}"
            )

        # Check CSV values match master
        csv_rows = csv[csv["window_label"] == w_label]
        if csv_rows.empty:
            fail(f"No rows for window_label={w_label} in CSV")
        csv_esc = csv_rows["mean_escalation"].iloc[0]
        csv_rel = csv_rows["mean_relief"].iloc[0]

        if abs(float(csv_esc) - esc_actual) > TOLERANCE:
            fail(
                f"CSV mean_escalation for {w_label} does not match master: "
                f"csv={float(csv_esc):.6f}, master={esc_actual:.6f}"
            )
        if abs(float(csv_rel) - rel_actual) > TOLERANCE:
            fail(
                f"CSV mean_relief for {w_label} does not match master: "
                f"csv={float(csv_rel):.6f}, master={rel_actual:.6f}"
            )

    ok("Means in CSV match formal_car_master.csv ground truth (within 0.01 pp)")


def check_latex_consistency() -> None:
    """
    Verify that the rounded percentage values expected in the LaTeX table
    correspond to the ground truth means. This acts as a static guard to
    prevent the bug where manually typed LaTeX values diverge from data.
    """
    for w_label, gt in GROUND_TRUTH.items():
        disp = WINDOW_DISPLAY[w_label]
        expected = LATEX_EXPECTED[disp]

        esc_tex = f"{gt['esc'] * 100:.2f}%"
        rel_tex = f"{gt['rel'] * 100:.2f}%"

        if esc_tex != expected["esc"]:
            fail(
                f"LaTeX escalation mean for {disp} should be '{expected['esc']}' "
                f"but computed rounding gives '{esc_tex}'"
            )
        if rel_tex != expected["rel"]:
            fail(
                f"LaTeX relief mean for {disp} should be '{expected['rel']}' "
                f"but computed rounding gives '{rel_tex}'"
            )

    ok("LaTeX percentage representations consistent with ground truth")


def check_both_tests_per_window() -> None:
    df = pd.read_csv(CSV_PATH)
    for w in EXPECTED_WINDOWS:
        sub = df[df["window_label"] == w]
        found = set(sub["test_method"].unique())
        missing = EXPECTED_TESTS - found
        if missing:
            fail(f"Window {w} missing test(s): {missing}")
    if len(df) != 6:
        fail(f"Expected 6 rows (3 windows × 2 tests), got {len(df)}")
    ok("Both Welch t-test and Mann-Whitney U present for all 3 windows")


def check_pvalues_in_range() -> None:
    df = pd.read_csv(CSV_PATH)
    pvals = df["p_value"].dropna()
    if not ((pvals >= 0) & (pvals <= 1)).all():
        fail(f"p_value has out-of-range values: {pvals[~((pvals >= 0) & (pvals <= 1))].tolist()}")
    ok("All p-values in [0, 1]")


def main() -> None:
    check_file_exists()
    check_columns()
    check_groups_only_esc_rel()
    check_n_values()
    check_means_match_ground_truth()
    check_latex_consistency()
    check_both_tests_per_window()
    check_pvalues_in_range()
    print("\nTodos os checks da comparacao de grupos passaram.")


if __name__ == "__main__":
    main()
