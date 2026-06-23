import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DATA"
FIGURES_DIR = BASE_DIR / "FIGURES"
SCRIPTS_DIR = BASE_DIR / "SCRIPTS"

WINDOW_ORDER = ["m1_p1", "m3_p3", "m5_p5"]
EXPECTED_N_TOTAL = 24
EXPECTED_N_ESCALATION = 13
EXPECTED_N_RELIEF = 10
EXPECTED_N_STRUCTURAL = 1

SUMMARY_FILE = DATA_DIR / "robust_car_summary.csv"
BY_GROUP_FILE = DATA_DIR / "robust_car_by_group.csv"
LOO_FILE = DATA_DIR / "leave_one_out_car_influence.csv"
MASTER_FILE = DATA_DIR / "formal_car_master.csv"

EXPECTED_FIGURES = [
    "robust_car_intervals_by_window.png",
    "car_distribution_by_window.png",
    "mean_median_robust_car_comparison.png",
    "leave_one_out_car_influence.png",
    "robust_car_by_group.png",
]

SUMMARY_REQUIRED_COLS = [
    "window_label", "n", "mean_car", "median_car", "std_car", "iqr_car", "mad_car",
    "trimmed_mean_car", "winsorized_mean_car", "min_car", "max_car",
    "bootstrap_ci_mean_lower", "bootstrap_ci_mean_upper",
    "bootstrap_ci_median_lower", "bootstrap_ci_median_upper",
    "ttest_pvalue", "wilcoxon_pvalue", "sign_test_pvalue",
    "n_positive", "n_negative", "n_neutral", "share_positive", "share_negative",
]

BY_GROUP_REQUIRED_COLS = [
    "event_group", "window_label", "n", "mean_car", "median_car",
    "trimmed_mean_car", "winsorized_mean_car", "iqr_car", "mad_car",
    "bootstrap_ci_mean_lower", "bootstrap_ci_mean_upper",
    "ttest_pvalue", "wilcoxon_pvalue", "sign_test_pvalue",
    "n_positive", "n_negative", "n_neutral", "share_positive", "share_negative",
]

LOO_REQUIRED_COLS = [
    "window_label", "excluded_event_id", "n_remaining",
    "full_mean_car", "loo_mean_car", "delta_mean_car",
    "full_median_car", "loo_median_car", "delta_median_car",
]


def ok(message: str) -> None:
    print(f"OK: {message}")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"ERRO: arquivo nao encontrado: {path}")


def validate_robust_car_summary() -> None:
    require_file(SUMMARY_FILE)
    df = pd.read_csv(SUMMARY_FILE)

    missing_cols = [c for c in SUMMARY_REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"ERRO: colunas ausentes em robust_car_summary.csv: {missing_cols}")

    present_windows = set(df["window_label"].unique())
    if present_windows != set(WINDOW_ORDER):
        raise ValueError(
            f"ERRO: janelas incorretas em robust_car_summary.csv. "
            f"Esperado: {set(WINDOW_ORDER)}, encontrado: {present_windows}"
        )

    if len(df) != 3:
        raise ValueError(f"ERRO: robust_car_summary.csv deve ter 3 linhas, encontrado {len(df)}")

    for window in WINDOW_ORDER:
        row = df.loc[df["window_label"] == window].iloc[0]
        if int(row["n"]) != EXPECTED_N_TOTAL:
            raise ValueError(
                f"ERRO: N incorreto para {window}: esperado {EXPECTED_N_TOTAL}, "
                f"encontrado {int(row['n'])}"
            )

    for pval_col in ["ttest_pvalue", "sign_test_pvalue"]:
        vals = df[pval_col].dropna()
        if vals.empty:
            raise ValueError(f"ERRO: {pval_col} esta totalmente vazio.")
        if not ((vals >= 0) & (vals <= 1)).all():
            raise ValueError(f"ERRO: {pval_col} com valores fora de [0, 1].")

    wilcoxon_vals = df["wilcoxon_pvalue"].dropna()
    if not wilcoxon_vals.empty:
        if not ((wilcoxon_vals >= 0) & (wilcoxon_vals <= 1)).all():
            raise ValueError("ERRO: wilcoxon_pvalue com valores fora de [0, 1].")

    for lo_col, hi_col in [
        ("bootstrap_ci_mean_lower", "bootstrap_ci_mean_upper"),
        ("bootstrap_ci_median_lower", "bootstrap_ci_median_upper"),
    ]:
        valid = df[[lo_col, hi_col]].dropna()
        if not valid.empty:
            if not (valid[lo_col] < valid[hi_col]).all():
                raise ValueError(
                    f"ERRO: intervalo inconsistente ({lo_col} >= {hi_col}) em alguma janela."
                )

    for window in WINDOW_ORDER:
        row = df.loc[df["window_label"] == window].iloc[0]
        n_check = int(row["n_positive"]) + int(row["n_negative"]) + int(row["n_neutral"])
        if n_check != int(row["n"]):
            raise ValueError(
                f"ERRO: n_positive + n_negative + n_neutral != n para janela {window}"
            )

    ok("robust_car_summary.csv valido")


def validate_robust_car_by_group() -> None:
    require_file(BY_GROUP_FILE)
    df = pd.read_csv(BY_GROUP_FILE)

    missing_cols = [c for c in BY_GROUP_REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"ERRO: colunas ausentes em robust_car_by_group.csv: {missing_cols}")

    expected_rows = 3 * 3  # 3 groups x 3 windows
    if len(df) != expected_rows:
        raise ValueError(
            f"ERRO: robust_car_by_group.csv deve ter {expected_rows} linhas, encontrado {len(df)}"
        )

    n_map = {
        "escalation": EXPECTED_N_ESCALATION,
        "relief": EXPECTED_N_RELIEF,
        "structural": EXPECTED_N_STRUCTURAL,
    }
    for group, expected_n in n_map.items():
        group_rows = df.loc[df["event_group"] == group]
        if group_rows.empty:
            raise ValueError(f"ERRO: grupo '{group}' ausente em robust_car_by_group.csv")
        if len(group_rows) != 3:
            raise ValueError(
                f"ERRO: grupo '{group}' deve ter 3 linhas, encontrado {len(group_rows)}"
            )
        for _, row in group_rows.iterrows():
            if int(row["n"]) != expected_n:
                raise ValueError(
                    f"ERRO: N incorreto para {group}/{row['window_label']}: "
                    f"esperado {expected_n}, encontrado {int(row['n'])}"
                )

    ok("robust_car_by_group.csv valido")


def validate_leave_one_out() -> None:
    require_file(LOO_FILE)
    df = pd.read_csv(LOO_FILE)

    missing_cols = [c for c in LOO_REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"ERRO: colunas ausentes em leave_one_out_car_influence.csv: {missing_cols}")

    expected_rows = 3 * EXPECTED_N_TOTAL
    if len(df) != expected_rows:
        raise ValueError(
            f"ERRO: leave_one_out_car_influence.csv deve ter {expected_rows} linhas, "
            f"encontrado {len(df)}"
        )

    for window in WINDOW_ORDER:
        subset = df.loc[df["window_label"] == window]
        if len(subset) != EXPECTED_N_TOTAL:
            raise ValueError(
                f"ERRO: janela {window} tem {len(subset)} linhas, esperado {EXPECTED_N_TOTAL}"
            )
        if not (subset["n_remaining"] == EXPECTED_N_TOTAL - 1).all():
            raise ValueError(
                f"ERRO: n_remaining incorreto para janela {window}. "
                f"Esperado {EXPECTED_N_TOTAL - 1}."
            )

    ok("leave_one_out_car_influence.csv valido")


def validate_figures() -> None:
    for figure_name in EXPECTED_FIGURES:
        path = FIGURES_DIR / figure_name
        if not path.exists():
            raise FileNotFoundError(f"ERRO: figura nao encontrada: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"ERRO: figura vazia: {path}")
        with Image.open(path) as img:
            width, height = img.size
        if width < 800 or height < 500:
            raise ValueError(
                f"ERRO: resolucao insuficiente em {figure_name}: {width}x{height}"
            )
    ok("figuras robustas encontradas e validas")


def validate_original_data_intact() -> None:
    require_file(MASTER_FILE)
    df = pd.read_csv(MASTER_FILE)
    expected_rows = 72  # 24 events x 3 windows
    if len(df) != expected_rows:
        raise ValueError(
            f"ERRO: formal_car_master.csv tem {len(df)} linhas, esperado {expected_rows}. "
            "Dados originais podem ter sido modificados."
        )
    expected_cols = ["event_id", "event_group", "window_label", "formal_car_sp500"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"ERRO: colunas ausentes em formal_car_master.csv: {missing}")
    # spot-check a known value: E08 m3_p3 should be approximately -0.1102
    e08_m3 = df.loc[
        (df["event_id"] == "E08") & (df["window_label"] == "m3_p3"), "formal_car_sp500"
    ]
    if e08_m3.empty:
        raise ValueError("ERRO: E08/m3_p3 ausente em formal_car_master.csv")
    if abs(float(e08_m3.values[0]) - (-0.1102467244752972)) > 1e-8:
        raise ValueError("ERRO: valor de E08/m3_p3 alterado em formal_car_master.csv")
    ok("dados originais intactos (formal_car_master.csv)")


def validate_reproducibility() -> None:
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        from build_robust_car_statistics import (
            BOOTSTRAP_N_RESAMPLES,
            BOOTSTRAP_SEED,
            compute_robust_stats_for_window,
        )
    except ImportError as exc:
        raise ImportError(
            f"ERRO: nao foi possivel importar build_robust_car_statistics: {exc}"
        ) from exc

    df = pd.read_csv(MASTER_FILE)
    saved = pd.read_csv(SUMMARY_FILE)

    for window in WINDOW_ORDER:
        values = df.loc[df["window_label"] == window, "formal_car_sp500"].dropna().values
        recomputed = compute_robust_stats_for_window(values, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED)
        saved_row = saved.loc[saved["window_label"] == window].iloc[0]

        for col in ["mean_car", "median_car", "bootstrap_ci_mean_lower", "bootstrap_ci_mean_upper"]:
            saved_val = float(saved_row[col])
            computed_val = float(recomputed[col])
            if abs(saved_val - computed_val) > 1e-10:
                raise ValueError(
                    f"ERRO: falha de reprodutibilidade em {window}/{col}: "
                    f"salvo={saved_val:.10f}, recomputado={computed_val:.10f}"
                )

    ok("reprodutibilidade confirmada com seed fixa")


def main() -> None:
    validate_original_data_intact()
    validate_robust_car_summary()
    validate_robust_car_by_group()
    validate_leave_one_out()
    validate_figures()
    validate_reproducibility()
    print("\nVALIDACAO CONCLUIDA")


if __name__ == "__main__":
    main()
