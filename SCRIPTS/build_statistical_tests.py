from pathlib import Path

import pandas as pd
from scipy import stats


BASE_DIR = Path(__file__).resolve().parent.parent
FORMAL_CAR_BY_EVENT_FILE = BASE_DIR / "DATA" / "formal_car_by_event.csv"
RETURN_VS_VOLATILITY_FILE = BASE_DIR / "DATA" / "return_vs_volatility.csv"

WINDOW_STAT_TESTS_FILE = BASE_DIR / "DATA" / "window_stat_tests.csv"
GROUP_STAT_TESTS_FILE = BASE_DIR / "DATA" / "group_stat_tests.csv"
CORRELATION_TESTS_FILE = BASE_DIR / "DATA" / "correlation_tests.csv"

WINDOW_ORDER = ["m1_p1", "m3_p3", "m5_p5"]
WINDOW_COMPARISONS = [("m1_p1", "m3_p3"), ("m3_p3", "m5_p5")]
GROUP_ORDER = ["escalation", "relief"]


def load_formal_car_by_event() -> pd.DataFrame:
    if not FORMAL_CAR_BY_EVENT_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {FORMAL_CAR_BY_EVENT_FILE}. "
            "Rode primeiro build_formal_car_tables.py."
        )

    formal_car_by_event = pd.read_csv(FORMAL_CAR_BY_EVENT_FILE)
    required_columns = ["event_id", "event_group", "window_label", "formal_car_sp500"]
    missing_columns = [column for column in required_columns if column not in formal_car_by_event.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes em formal_car_by_event.csv: {missing_columns}")

    return formal_car_by_event


def load_return_vs_volatility() -> pd.DataFrame:
    if not RETURN_VS_VOLATILITY_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {RETURN_VS_VOLATILITY_FILE}. "
            "Rode primeiro build_return_vs_volatility.py."
        )

    table = pd.read_csv(RETURN_VS_VOLATILITY_FILE)
    required_columns = ["event_group", "window_label", "formal_car_sp500", "volatility_sp500"]
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes em return_vs_volatility.csv: {missing_columns}")

    return table


def maybe_paired_ttest(sample_a: pd.Series, sample_b: pd.Series) -> tuple[float, float]:
    if len(sample_a) < 2 or len(sample_b) < 2:
        return float("nan"), float("nan")
    result = stats.ttest_rel(sample_a, sample_b, nan_policy="omit")
    return float(result.statistic), float(result.pvalue)


def maybe_wilcoxon(sample_a: pd.Series, sample_b: pd.Series) -> tuple[float, float]:
    if len(sample_a) < 2 or len(sample_b) < 2:
        return float("nan"), float("nan")
    diffs = sample_a - sample_b
    if diffs.dropna().eq(0).all():
        return float("nan"), float("nan")
    result = stats.wilcoxon(sample_a, sample_b, zero_method="wilcox", correction=False)
    return float(result.statistic), float(result.pvalue)


def maybe_welch_ttest(sample_a: pd.Series, sample_b: pd.Series) -> tuple[float, float]:
    if len(sample_a) < 2 or len(sample_b) < 2:
        return float("nan"), float("nan")
    result = stats.ttest_ind(sample_a, sample_b, equal_var=False, nan_policy="omit")
    return float(result.statistic), float(result.pvalue)


def maybe_mannwhitney(sample_a: pd.Series, sample_b: pd.Series) -> tuple[float, float]:
    if len(sample_a) < 1 or len(sample_b) < 1:
        return float("nan"), float("nan")
    result = stats.mannwhitneyu(sample_a, sample_b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def build_window_stat_tests(formal_car_by_event: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for scope_type, scope_value, subset in [
        ("all_events", "all_events", formal_car_by_event),
        *[
            (
                "event_group",
                event_group,
                formal_car_by_event.loc[formal_car_by_event["event_group"] == event_group],
            )
            for event_group in GROUP_ORDER
        ],
    ]:
        pivot = (
            subset.pivot_table(
                index="event_id",
                columns="window_label",
                values="formal_car_sp500",
                aggfunc="first",
            )
            .reindex(columns=WINDOW_ORDER)
        )

        for window_a, window_b in WINDOW_COMPARISONS:
            pair = pivot[[window_a, window_b]].dropna()
            sample_a = pair[window_a]
            sample_b = pair[window_b]

            mean_diff = (sample_b - sample_a).mean() if not pair.empty else float("nan")
            median_diff = (sample_b - sample_a).median() if not pair.empty else float("nan")

            t_stat, t_pvalue = maybe_paired_ttest(sample_a, sample_b)
            rows.append(
                {
                    "scope_type": scope_type,
                    "scope_value": scope_value,
                    "comparison_label": f"{window_a}_vs_{window_b}",
                    "test_method": "paired_ttest",
                    "n_pairs": int(len(pair)),
                    "window_a": window_a,
                    "window_b": window_b,
                    "mean_window_a": sample_a.mean() if not pair.empty else float("nan"),
                    "mean_window_b": sample_b.mean() if not pair.empty else float("nan"),
                    "mean_diff_b_minus_a": mean_diff,
                    "median_diff_b_minus_a": median_diff,
                    "statistic": t_stat,
                    "p_value": t_pvalue,
                    "significant_5pct": bool(pd.notna(t_pvalue) and t_pvalue < 0.05),
                }
            )

            w_stat, w_pvalue = maybe_wilcoxon(sample_a, sample_b)
            rows.append(
                {
                    "scope_type": scope_type,
                    "scope_value": scope_value,
                    "comparison_label": f"{window_a}_vs_{window_b}",
                    "test_method": "wilcoxon_signed_rank",
                    "n_pairs": int(len(pair)),
                    "window_a": window_a,
                    "window_b": window_b,
                    "mean_window_a": sample_a.mean() if not pair.empty else float("nan"),
                    "mean_window_b": sample_b.mean() if not pair.empty else float("nan"),
                    "mean_diff_b_minus_a": mean_diff,
                    "median_diff_b_minus_a": median_diff,
                    "statistic": w_stat,
                    "p_value": w_pvalue,
                    "significant_5pct": bool(pd.notna(w_pvalue) and w_pvalue < 0.05),
                }
            )

    return pd.DataFrame(rows)


def build_group_stat_tests(formal_car_by_event: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for window_label in WINDOW_ORDER:
        subset = formal_car_by_event.loc[formal_car_by_event["window_label"] == window_label]
        sample_escalation = subset.loc[
            subset["event_group"] == "escalation", "formal_car_sp500"
        ].dropna()
        sample_relief = subset.loc[
            subset["event_group"] == "relief", "formal_car_sp500"
        ].dropna()

        mean_diff = sample_escalation.mean() - sample_relief.mean()
        median_diff = sample_escalation.median() - sample_relief.median()

        t_stat, t_pvalue = maybe_welch_ttest(sample_escalation, sample_relief)
        rows.append(
            {
                "window_label": window_label,
                "group_a": "escalation",
                "group_b": "relief",
                "test_method": "welch_ttest",
                "n_group_a": int(len(sample_escalation)),
                "n_group_b": int(len(sample_relief)),
                "mean_group_a": sample_escalation.mean(),
                "mean_group_b": sample_relief.mean(),
                "mean_diff_a_minus_b": mean_diff,
                "median_diff_a_minus_b": median_diff,
                "statistic": t_stat,
                "p_value": t_pvalue,
                "significant_5pct": bool(pd.notna(t_pvalue) and t_pvalue < 0.05),
            }
        )

        mw_stat, mw_pvalue = maybe_mannwhitney(sample_escalation, sample_relief)
        rows.append(
            {
                "window_label": window_label,
                "group_a": "escalation",
                "group_b": "relief",
                "test_method": "mannwhitney_u",
                "n_group_a": int(len(sample_escalation)),
                "n_group_b": int(len(sample_relief)),
                "mean_group_a": sample_escalation.mean(),
                "mean_group_b": sample_relief.mean(),
                "mean_diff_a_minus_b": mean_diff,
                "median_diff_a_minus_b": median_diff,
                "statistic": mw_stat,
                "p_value": mw_pvalue,
                "significant_5pct": bool(pd.notna(mw_pvalue) and mw_pvalue < 0.05),
            }
        )

    return pd.DataFrame(rows)


def build_correlation_tests(return_vs_volatility: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for scope_type, scope_value, subset in [
        ("all_events", "all_events", return_vs_volatility),
        *[
            (
                "window_label",
                window_label,
                return_vs_volatility.loc[return_vs_volatility["window_label"] == window_label],
            )
            for window_label in WINDOW_ORDER
        ],
    ]:
        valid_subset = subset[["formal_car_sp500", "volatility_sp500"]].dropna()
        n_obs = len(valid_subset)

        if n_obs >= 2:
            pearson_stat, pearson_pvalue = stats.pearsonr(
                valid_subset["formal_car_sp500"],
                valid_subset["volatility_sp500"],
            )
            spearman_stat, spearman_pvalue = stats.spearmanr(
                valid_subset["formal_car_sp500"],
                valid_subset["volatility_sp500"],
            )
        else:
            pearson_stat = pearson_pvalue = spearman_stat = spearman_pvalue = float("nan")

        rows.append(
            {
                "scope_type": scope_type,
                "scope_value": scope_value,
                "test_method": "pearson",
                "n_obs": int(n_obs),
                "correlation": float(pearson_stat) if pd.notna(pearson_stat) else float("nan"),
                "p_value": float(pearson_pvalue) if pd.notna(pearson_pvalue) else float("nan"),
                "significant_5pct": bool(pd.notna(pearson_pvalue) and pearson_pvalue < 0.05),
            }
        )
        rows.append(
            {
                "scope_type": scope_type,
                "scope_value": scope_value,
                "test_method": "spearman",
                "n_obs": int(n_obs),
                "correlation": float(spearman_stat) if pd.notna(spearman_stat) else float("nan"),
                "p_value": float(spearman_pvalue) if pd.notna(spearman_pvalue) else float("nan"),
                "significant_5pct": bool(pd.notna(spearman_pvalue) and spearman_pvalue < 0.05),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    formal_car_by_event = load_formal_car_by_event()
    return_vs_volatility = load_return_vs_volatility()

    window_stat_tests = build_window_stat_tests(formal_car_by_event)
    group_stat_tests = build_group_stat_tests(formal_car_by_event)
    correlation_tests = build_correlation_tests(return_vs_volatility)

    WINDOW_STAT_TESTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    window_stat_tests.to_csv(WINDOW_STAT_TESTS_FILE, index=False)
    group_stat_tests.to_csv(GROUP_STAT_TESTS_FILE, index=False)
    correlation_tests.to_csv(CORRELATION_TESTS_FILE, index=False)

    print(f"Arquivo salvo em: {WINDOW_STAT_TESTS_FILE}")
    print(f"Arquivo salvo em: {GROUP_STAT_TESTS_FILE}")
    print(f"Arquivo salvo em: {CORRELATION_TESTS_FILE}")
    print("\nResumo dos testes entre janelas:")
    print(window_stat_tests.to_string(index=False))
    print("\nResumo dos testes entre grupos:")
    print(group_stat_tests.to_string(index=False))
    print("\nResumo das correlacoes:")
    print(correlation_tests.to_string(index=False))


if __name__ == "__main__":
    main()
