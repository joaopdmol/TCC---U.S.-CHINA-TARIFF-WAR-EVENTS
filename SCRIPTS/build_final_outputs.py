from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
FORMAL_CAR_BY_EVENT_FILE = BASE_DIR / "DATA" / "formal_car_by_event.csv"
VOLATILITY_BY_EVENT_FILE = BASE_DIR / "DATA" / "volatility_by_event.csv"
FORMAL_CAR_MASTER_FILE = BASE_DIR / "DATA" / "formal_car_master.csv"
WINDOW_STAT_TESTS_FILE = BASE_DIR / "DATA" / "window_stat_tests.csv"
GROUP_STAT_TESTS_FILE = BASE_DIR / "DATA" / "group_stat_tests.csv"
CORRELATION_TESTS_FILE = BASE_DIR / "DATA" / "correlation_tests.csv"
EVENTS_MASTER_FILE = BASE_DIR / "DATA" / "events_master.csv"
MASTER_COVERAGE_FILE = BASE_DIR / "DATA" / "event_windows_master_coverage.csv"

FINAL_CORE_FILE = BASE_DIR / "DATA" / "final_results_core.csv"
FINAL_EXPANDED_FILE = BASE_DIR / "DATA" / "final_results_expanded.csv"
FINAL_COMPARISON_FILE = BASE_DIR / "DATA" / "final_comparison_core_vs_expanded.csv"
FINAL_WINDOW_ANALYSIS_FILE = BASE_DIR / "DATA" / "final_window_analysis.csv"
RESULTS_SECTION_FILE = BASE_DIR / "RESULTS_SECTION.md"

WINDOW_ORDER = ["m1_p1", "m3_p3", "m5_p5"]
NEUTRAL_THRESHOLD = 0.005


def load_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    dataframe = pd.read_csv(path)
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes em {path.name}: {missing_columns}")
    return dataframe


def prepare_core_sample() -> pd.DataFrame:
    core = load_csv(
        FORMAL_CAR_BY_EVENT_FILE,
        [
            "event_id",
            "window_label",
            "formal_car_sp500",
            "event_group",
            "event_type",
            "description",
        ],
    ).copy()
    volatility = load_csv(
        VOLATILITY_BY_EVENT_FILE,
        [
            "event_id",
            "event_group",
            "event_type",
            "description",
            "window_label",
            "volatility_sp500",
        ],
    )
    core = core.merge(
        volatility,
        on=["event_id", "event_group", "event_type", "description", "window_label"],
        how="left",
    )
    core["car_sign"] = core["formal_car_sp500"].apply(classify_sign)
    return core


def prepare_expanded_sample() -> pd.DataFrame:
    expanded = load_csv(
        FORMAL_CAR_MASTER_FILE,
        [
            "event_id",
            "window_label",
            "formal_car_sp500",
            "volatility_sp500",
            "car_sign",
            "is_core_sample",
        ],
    ).copy()
    expanded["is_core_sample"] = expanded["is_core_sample"].astype(str).str.lower().eq("true")
    return expanded


def classify_sign(value: float) -> str:
    if pd.isna(value):
        return "missing"
    if value > NEUTRAL_THRESHOLD:
        return "positive"
    if value < -NEUTRAL_THRESHOLD:
        return "negative"
    return "neutral"


def build_summary_table(
    dataframe: pd.DataFrame,
    sample_label: str,
    sample_definition: str,
) -> pd.DataFrame:
    summary = (
        dataframe.groupby("window_label", observed=True)
        .agg(
            mean_formal_car_sp500=("formal_car_sp500", "mean"),
            median_formal_car_sp500=("formal_car_sp500", "median"),
            std_formal_car_sp500=("formal_car_sp500", "std"),
            mean_volatility_sp500=("volatility_sp500", "mean"),
            positive_share=("car_sign", lambda series: series.eq("positive").mean()),
            negative_share=("car_sign", lambda series: series.eq("negative").mean()),
            neutral_share=("car_sign", lambda series: series.eq("neutral").mean()),
            n_events=("event_id", "nunique"),
        )
        .reset_index()
    )
    summary["sample_label"] = sample_label
    summary["sample_definition"] = sample_definition
    summary["neutral_threshold"] = NEUTRAL_THRESHOLD
    summary["window_label"] = pd.Categorical(
        summary["window_label"],
        categories=WINDOW_ORDER,
        ordered=True,
    )
    summary = summary.sort_values("window_label").reset_index(drop=True)
    ordered_columns = [
        "sample_label",
        "sample_definition",
        "window_label",
        "mean_formal_car_sp500",
        "median_formal_car_sp500",
        "std_formal_car_sp500",
        "mean_volatility_sp500",
        "positive_share",
        "negative_share",
        "neutral_share",
        "n_events",
        "neutral_threshold",
    ]
    return summary[ordered_columns]


def build_comparison_table(core_summary: pd.DataFrame, expanded_summary: pd.DataFrame) -> pd.DataFrame:
    comparison = core_summary.merge(
        expanded_summary,
        on="window_label",
        suffixes=("_core", "_expanded"),
        how="inner",
    )
    comparison["diff_mean_formal_car_sp500_expanded_minus_core"] = (
        comparison["mean_formal_car_sp500_expanded"] - comparison["mean_formal_car_sp500_core"]
    )
    comparison["diff_mean_volatility_sp500_expanded_minus_core"] = (
        comparison["mean_volatility_sp500_expanded"] - comparison["mean_volatility_sp500_core"]
    )
    comparison["diff_positive_share_expanded_minus_core"] = (
        comparison["positive_share_expanded"] - comparison["positive_share_core"]
    )
    comparison["diff_negative_share_expanded_minus_core"] = (
        comparison["negative_share_expanded"] - comparison["negative_share_core"]
    )
    return comparison.sort_values("window_label").reset_index(drop=True)


def describe_window_tendency(m1: float, m3: float, m5: float) -> tuple[str, str]:
    if pd.isna(m1) or pd.isna(m3) or pd.isna(m5):
        return "insufficient_data", "Nao ha dados suficientes para comparar as tres janelas."
    if m1 < 0 and m5 > 0:
        return "reversal_toward_positive", "O efeito medio muda de sinal e aponta para reversao ao longo do tempo."
    if m1 > 0 and m5 < 0:
        return "reversal_toward_negative", "O efeito medio muda de sinal e aponta para deterioracao ao longo do tempo."
    if m1 * m5 > 0 and abs(m5) > abs(m1):
        return "persistence_with_amplification", "O sinal do CAR se preserva e ganha intensidade nas janelas mais longas."
    if m1 * m5 > 0 and abs(m5) < abs(m1):
        return "persistence_with_attenuation", "O sinal do CAR se preserva, mas enfraquece nas janelas mais longas."
    return "relative_stability", "O padrao medio permanece relativamente estavel entre as janelas."


def render_tendency_sentence(tendency_label: str) -> str:
    mapping = {
        "reversal_toward_positive": "um movimento de reversao ao longo do tempo, com melhora do CAR medio nas janelas mais longas",
        "reversal_toward_negative": "um movimento de deterioracao ao longo do tempo, com piora do CAR medio nas janelas mais longas",
        "persistence_with_amplification": "persistencia do sinal inicial, acompanhada de ampliacao de magnitude nas janelas mais longas",
        "persistence_with_attenuation": "persistencia do sinal inicial, mas com atenuacao de magnitude nas janelas mais longas",
        "relative_stability": "estabilidade relativa do CAR medio entre as janelas analisadas",
        "insufficient_data": "informacao insuficiente para uma comparacao completa entre as janelas",
    }
    return mapping.get(tendency_label, "um padrao temporal que requer cautela interpretativa")


def build_window_analysis(summary_table: pd.DataFrame) -> pd.DataFrame:
    results = []
    for row in summary_table[["sample_label", "window_label", "mean_formal_car_sp500", "n_events"]].itertuples(index=False):
        pass

    for sample_label, sample_df in summary_table.groupby("sample_label", observed=True):
        sample_df = sample_df.set_index("window_label")
        m1 = sample_df.loc["m1_p1", "mean_formal_car_sp500"]
        m3 = sample_df.loc["m3_p3", "mean_formal_car_sp500"]
        m5 = sample_df.loc["m5_p5", "mean_formal_car_sp500"]
        tendency_label, interpretation_note = describe_window_tendency(m1, m3, m5)

        results.append(
            {
                "sample_label": sample_label,
                "n_events": int(sample_df["n_events"].max()),
                "mean_formal_car_m1_p1": m1,
                "mean_formal_car_m3_p3": m3,
                "mean_formal_car_m5_p5": m5,
                "diff_m3_minus_m1": m3 - m1,
                "diff_m5_minus_m3": m5 - m3,
                "diff_m5_minus_m1": m5 - m1,
                "tendency_label": tendency_label,
                "interpretation_note": interpretation_note,
            }
        )

    return pd.DataFrame(results).sort_values("sample_label").reset_index(drop=True)


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value * 100:.2f}%"


def format_pp(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value * 100:.2f} p.p."


def format_pvalue(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value < 0.001:
        return "< 0.001"
    return f"{value:.3f}"


def get_stat_row(dataframe: pd.DataFrame, **filters: str) -> pd.Series:
    mask = pd.Series(True, index=dataframe.index)
    for column, value in filters.items():
        mask &= dataframe[column].eq(value)
    filtered = dataframe.loc[mask]
    if filtered.empty:
        raise ValueError(f"Nao foi possivel localizar linha estatistica com filtros: {filters}")
    return filtered.iloc[0]


def build_results_section(
    core_summary: pd.DataFrame,
    expanded_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    window_analysis: pd.DataFrame,
) -> str:
    window_tests = load_csv(
        WINDOW_STAT_TESTS_FILE,
        [
            "scope_type",
            "scope_value",
            "comparison_label",
            "test_method",
            "p_value",
            "mean_diff_b_minus_a",
            "significant_5pct",
        ],
    )
    group_tests = load_csv(
        GROUP_STAT_TESTS_FILE,
        [
            "window_label",
            "test_method",
            "p_value",
            "significant_5pct",
        ],
    )
    correlation_tests = load_csv(
        CORRELATION_TESTS_FILE,
        [
            "scope_type",
            "scope_value",
            "test_method",
            "correlation",
            "p_value",
            "significant_5pct",
        ],
    )
    events_master = load_csv(
        EVENTS_MASTER_FILE,
        ["event_id", "event_regime"],
    )
    master_coverage = load_csv(
        MASTER_COVERAGE_FILE,
        ["event_id", "is_complete"],
    )

    total_master_events = events_master["event_id"].nunique()
    covered_events = master_coverage.loc[master_coverage["is_complete"], "event_id"].nunique()
    excluded_events = total_master_events - covered_events

    core_m1 = core_summary.loc[core_summary["window_label"] == "m1_p1"].iloc[0]
    core_m3 = core_summary.loc[core_summary["window_label"] == "m3_p3"].iloc[0]
    core_m5 = core_summary.loc[core_summary["window_label"] == "m5_p5"].iloc[0]
    expanded_m1 = expanded_summary.loc[expanded_summary["window_label"] == "m1_p1"].iloc[0]
    expanded_m3 = expanded_summary.loc[expanded_summary["window_label"] == "m3_p3"].iloc[0]
    expanded_m5 = expanded_summary.loc[expanded_summary["window_label"] == "m5_p5"].iloc[0]

    core_window_analysis = window_analysis.loc[window_analysis["sample_label"] == "core"].iloc[0]
    expanded_window_analysis = window_analysis.loc[
        window_analysis["sample_label"] == "expanded_market_covered"
    ].iloc[0]

    window_test_m1_m3 = get_stat_row(
        window_tests,
        scope_type="all_events",
        scope_value="all_events",
        comparison_label="m1_p1_vs_m3_p3",
        test_method="paired_ttest",
    )
    window_test_m3_m5 = get_stat_row(
        window_tests,
        scope_type="all_events",
        scope_value="all_events",
        comparison_label="m3_p3_vs_m5_p5",
        test_method="paired_ttest",
    )
    window_test_m3_m5_wilcoxon = get_stat_row(
        window_tests,
        scope_type="all_events",
        scope_value="all_events",
        comparison_label="m3_p3_vs_m5_p5",
        test_method="wilcoxon_signed_rank",
    )
    group_tests_rows = []
    for window_label in WINDOW_ORDER:
        group_tests_rows.append(
            get_stat_row(
                group_tests,
                window_label=window_label,
                test_method="welch_ttest",
            )
        )

    correlation_pearson = get_stat_row(
        correlation_tests,
        scope_type="all_events",
        scope_value="all_events",
        test_method="pearson",
    )
    correlation_spearman = get_stat_row(
        correlation_tests,
        scope_type="all_events",
        scope_value="all_events",
        test_method="spearman",
    )

    comparison_m5 = comparison.loc[comparison["window_label"] == "m5_p5"].iloc[0]

    return f"""# 6. Resultados

## 6.1 Introducao da secao de resultados

Esta secao apresenta os resultados finais do event study em duas camadas complementares. A primeira corresponde a analise principal, baseada na amostra core de 17 eventos entre 2018 e 2020-02-14. A segunda corresponde a uma analise de robustez com a base expandida de eventos, organizada em `events_master.csv` e restrita aos eventos para os quais existe cobertura efetiva de mercado em `market_features.csv`.

A estrategia adotada preserva a separacao entre documentacao historica e inferencia financeira. Embora a cronologia mestre contenha {total_master_events} eventos, apenas {covered_events} contam com cobertura de mercado suficiente para integrar o event study expandido. Os {excluded_events} eventos restantes permanecem importantes para o enquadramento historico, mas nao entram na inferencia porque a base de mercado termina em 2020-12-31.

## 6.2 Resultados do event study (core)

Na amostra principal, o CAR formal medio do S&P 500 foi de {format_pct(core_m1['mean_formal_car_sp500'])} em `m1_p1`, {format_pct(core_m3['mean_formal_car_sp500'])} em `m3_p3` e {format_pct(core_m5['mean_formal_car_sp500'])} em `m5_p5`. O padrao agregado sugere {render_tendency_sentence(core_window_analysis['tendency_label'])}.

Essa leitura ganha forca quando se observa a distribuicao dos sinais. Em `m1_p1`, {format_pct(core_m1['negative_share'])} dos eventos foram classificados como negativos e {format_pct(core_m1['positive_share'])} como positivos. Em `m5_p5`, a participacao de eventos positivos sobe para {format_pct(core_m5['positive_share'])}, enquanto a de eventos negativos fica em {format_pct(core_m5['negative_share'])}. Em termos economicos, isso e consistente com a ideia de que parte da reacao inicial do mercado americano foi posteriormente amortecida ou revertida em janelas mais amplas.

## 6.3 Robustez com amostra expandida

A analise de robustez utiliza a cronologia expandida, mas considera apenas os eventos com cobertura efetiva de mercado. Nesse recorte, o CAR formal medio foi de {format_pct(expanded_m1['mean_formal_car_sp500'])} em `m1_p1`, {format_pct(expanded_m3['mean_formal_car_sp500'])} em `m3_p3` e {format_pct(expanded_m5['mean_formal_car_sp500'])} em `m5_p5`, com {int(expanded_m1['n_events'])} eventos por janela.

O comportamento agregado da amostra expandida coberta aponta para {render_tendency_sentence(expanded_window_analysis['tendency_label'])}. Em relacao ao trilho principal, os valores medios ficam menos extremos, o que e esperado quando se incorporam eventos adicionais do inicio do periodo pandemico. Ainda assim, o padrao geral de melhora relativa nas janelas mais longas permanece visivel, o que sugere que o resultado principal nao depende apenas do subconjunto original de 17 eventos.

## 6.4 Comparacao core vs expanded

Quando se compara diretamente a amostra principal com a amostra expandida coberta, o que muda e a intensidade media do efeito, nao a direcao geral da leitura temporal. Em `m5_p5`, por exemplo, o CAR medio passa de {format_pct(comparison_m5['mean_formal_car_sp500_core'])} no core para {format_pct(comparison_m5['mean_formal_car_sp500_expanded'])} na amostra expandida, uma diferenca de {format_pp(comparison_m5['diff_mean_formal_car_sp500_expanded_minus_core'])}.

Essa comparacao sugere que a inclusao dos eventos adicionais reduz a magnitude media do CAR positivo nas janelas mais longas, mas nao elimina a evidencia de acomodacao do choque ao longo do tempo. Em outras palavras, a robustez expandida torna o resultado mais conservador, sem alterar de forma substantiva a narrativa central do trabalho.

## 6.5 Relacao retorno vs volatilidade

Os resultados da correlacao entre retorno acumulado e volatilidade apontam para uma associacao negativa moderada. No conjunto agregado, a correlacao de Pearson entre `formal_car_sp500` e `volatility_sp500` foi de {correlation_pearson['correlation']:.3f} com `p = {format_pvalue(correlation_pearson['p_value'])}`, enquanto a correlacao de Spearman foi de {correlation_spearman['correlation']:.3f} com `p = {format_pvalue(correlation_spearman['p_value'])}`.

Isso sugere que janelas com CAR mais negativo tendem, em media, a conviver com maior volatilidade, embora a robustez desse padrao deva ser interpretada com cautela. A significancia aparece no coeficiente de Pearson, mas nao se repete no coeficiente de Spearman, o que indica um sinal empirico interessante, porem ainda moderado.

## 6.6 Interpretacao dos testes estatisticos

Os testes entre janelas nao indicaram evidencia estatisticamente significativa de diferenca entre `m1_p1` e `m3_p3` no conjunto total (`p = {format_pvalue(window_test_m1_m3['p_value'])}` no teste t pareado). O mesmo ocorre para `m3_p3` versus `m5_p5` no teste t pareado (`p = {format_pvalue(window_test_m3_m5['p_value'])}`). No teste nao parametrico de Wilcoxon para essa segunda comparacao, o p-value foi `p = {format_pvalue(window_test_m3_m5_wilcoxon['p_value'])}`, o que caracteriza no maximo evidencia sugestiva, e nao conclusiva, de diferenca entre janelas.

Nos testes entre grupos, tambem nao houve evidencia estatisticamente significativa de diferenca entre eventos de `escalation` e `relief` em nenhuma das tres janelas. Os p-values dos testes t de Welch foram {format_pvalue(group_tests_rows[0]['p_value'])} em `m1_p1`, {format_pvalue(group_tests_rows[1]['p_value'])} em `m3_p3` e {format_pvalue(group_tests_rows[2]['p_value'])} em `m5_p5`. Portanto, a leitura mais adequada nao e a de igualdade comprovada entre grupos, mas sim a de ausencia de evidencia estatistica forte para diferenca dada a amostra disponivel.

## 6.7 Limitacoes metodologicas

Ha pelo menos tres limitacoes importantes. Primeiro, o tamanho amostral continua relativamente pequeno, sobretudo quando a analise e particionada por grupo de evento. Segundo, a cronologia mestre vai ate 2026, mas a cobertura de mercado vai apenas ate 2020-12-31, o que impede incorporar a totalidade dos eventos expandidos na inferencia financeira. Terceiro, o retorno esperado foi modelado de forma simples, como media historica do proprio S&P 500 em uma janela de estimacao anterior ao evento, sem recorrer a market model, CAPM ou fatores adicionais.

Essas escolhas sao deliberadamente conservadoras. Elas tornam a estrategia empirica mais transparente e facilmente defensavel, mas ao mesmo tempo limitam a ambicao inferencial do estudo.

## 6.8 Sintese dos achados

Em sintese, a analise principal indica que os efeitos medios da guerra tarifaria sobre o mercado americano foram mais negativos ou ambivalentes em janelas curtas, mas tenderam a se tornar menos adversos em janelas mais amplas. A amostra expandida com cobertura de mercado confirma essa direcao geral, embora com magnitudes mais moderadas.

Os testes estatisticos nao fornecem evidencia forte de diferencas entre grupos ou entre janelas no nivel convencional de 5 por cento, o que recomenda cautela interpretativa. Ainda assim, o conjunto dos resultados descritivos, das correlacoes com volatilidade e da robustez expandida oferece evidencia consistente com a leitura de que os choques comerciais produziram reacoes financeiras heterogeneas e parcialmente reversiveis ao longo do tempo.
"""


def main() -> None:
    core = prepare_core_sample()
    expanded = prepare_expanded_sample()

    core_summary = build_summary_table(
        core,
        sample_label="core",
        sample_definition="Amostra principal com 17 eventos ate 2020-02-14.",
    )
    expanded_summary = build_summary_table(
        expanded,
        sample_label="expanded_market_covered",
        sample_definition="Amostra expandida com eventos do master que possuem cobertura de mercado ate 2020-12-31.",
    )
    comparison = build_comparison_table(core_summary, expanded_summary)
    window_analysis = build_window_analysis(pd.concat([core_summary, expanded_summary], ignore_index=True))
    results_section = build_results_section(core_summary, expanded_summary, comparison, window_analysis)

    FINAL_CORE_FILE.parent.mkdir(parents=True, exist_ok=True)
    core_summary.to_csv(FINAL_CORE_FILE, index=False)
    expanded_summary.to_csv(FINAL_EXPANDED_FILE, index=False)
    comparison.to_csv(FINAL_COMPARISON_FILE, index=False)
    window_analysis.to_csv(FINAL_WINDOW_ANALYSIS_FILE, index=False)
    RESULTS_SECTION_FILE.write_text(results_section, encoding="utf-8")

    print(f"Arquivo salvo em: {FINAL_CORE_FILE}")
    print(f"Arquivo salvo em: {FINAL_EXPANDED_FILE}")
    print(f"Arquivo salvo em: {FINAL_COMPARISON_FILE}")
    print(f"Arquivo salvo em: {FINAL_WINDOW_ANALYSIS_FILE}")
    print(f"Arquivo salvo em: {RESULTS_SECTION_FILE}")


if __name__ == "__main__":
    main()
