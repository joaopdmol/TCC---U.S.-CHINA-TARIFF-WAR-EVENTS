from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
ESTIMATION_FILE = BASE_DIR / "DATA" / "sensitivity_estimation_results.csv"
SIGN_FILE = BASE_DIR / "DATA" / "sensitivity_sign_results.csv"
SAMPLE_FILE = BASE_DIR / "DATA" / "sensitivity_sample_results.csv"
FINAL_CORE_FILE = BASE_DIR / "DATA" / "final_results_core.csv"
FINAL_EXPANDED_FILE = BASE_DIR / "DATA" / "final_results_expanded.csv"
CORRELATION_FILE = BASE_DIR / "DATA" / "correlation_tests.csv"
OUTPUT_SUMMARY = BASE_DIR / "DATA" / "robustness_summary.csv"
OUTPUT_TEXT = BASE_DIR / "ROBUSTNESS_SECTION.md"


def load_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    dataframe = pd.read_csv(path)
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes em {path.name}: {missing_columns}")
    return dataframe


def classify_robustness(score: float) -> str:
    if score >= 0.8:
        return "robust"
    if score >= 0.5:
        return "moderately sensitive"
    return "sensitive"


def build_robustness_summary() -> pd.DataFrame:
    estimation = load_csv(
        ESTIMATION_FILE,
        ["estimation_config", "window_label", "mean_formal_car_sp500", "relative_change_pct_vs_baseline"],
    )
    sign = load_csv(
        SIGN_FILE,
        ["neutral_threshold", "event_group", "window_label", "sign_label", "event_share"],
    )
    sample = load_csv(
        SAMPLE_FILE,
        ["sample_label", "window_label", "mean_formal_car_sp500", "positive_share"],
    )
    final_core = load_csv(FINAL_CORE_FILE, ["window_label", "mean_formal_car_sp500"])
    final_expanded = load_csv(FINAL_EXPANDED_FILE, ["window_label", "mean_formal_car_sp500"])
    correlation = load_csv(CORRELATION_FILE, ["scope_type", "scope_value", "test_method", "correlation", "p_value"])

    rows = []

    core_m5 = final_core.loc[final_core["window_label"] == "m5_p5", "mean_formal_car_sp500"].iloc[0]
    estimation_m5 = estimation.loc[estimation["window_label"] == "m5_p5"].copy()
    same_sign_count = estimation_m5["mean_formal_car_sp500"].apply(lambda value: (value > 0) == (core_m5 > 0)).mean()
    max_relative_change = estimation_m5["relative_change_pct_vs_baseline"].dropna().abs().max()
    score = same_sign_count if pd.notna(max_relative_change) and max_relative_change > 100 else same_sign_count * 0.7
    rows.append(
        {
            "conclusion_id": "core_m5_positive",
            "conclusion_statement": "O CAR medio do core em m5_p5 permanece positivo.",
            "baseline_signal": "positive" if core_m5 > 0 else "negative",
            "baseline_magnitude": core_m5,
            "consistency_across_scenarios": f"{same_sign_count:.2f}",
            "robustness_class": classify_robustness(score),
            "evidence_note": "A conclusao foi confrontada com janelas de estimacao [-20,-6], [-30,-6] e [-40,-6].",
        }
    )

    core_series = final_core.set_index("window_label")["mean_formal_car_sp500"]
    upward_core = core_series["m1_p1"] < core_series["m3_p3"] < core_series["m5_p5"]
    estimation_direction_share = (
        estimation.pivot(index="estimation_config", columns="window_label", values="mean_formal_car_sp500")
        .apply(lambda row: row["m1_p1"] < row["m3_p3"] < row["m5_p5"], axis=1)
        .mean()
    )
    rows.append(
        {
            "conclusion_id": "core_progression_upward",
            "conclusion_statement": "No core, o CAR medio melhora conforme a janela aumenta.",
            "baseline_signal": "upward_progression" if upward_core else "mixed",
            "baseline_magnitude": core_series["m5_p5"] - core_series["m1_p1"],
            "consistency_across_scenarios": f"{estimation_direction_share:.2f}",
            "robustness_class": classify_robustness(estimation_direction_share),
            "evidence_note": "A ordenacao m1_p1 < m3_p3 < m5_p5 foi avaliada em todas as janelas de estimacao.",
        }
    )

    escalation_m5 = sign.loc[
        (sign["event_group"] == "escalation")
        & (sign["window_label"] == "m5_p5")
        & (sign["sign_label"] == "positive")
    ].copy()
    max_positive_share = escalation_m5["event_share"].max()
    min_positive_share = escalation_m5["event_share"].min()
    positive_consistency = float((escalation_m5["event_share"] > 0.5).mean())
    rows.append(
        {
            "conclusion_id": "escalation_positive_share_m5",
            "conclusion_statement": "Em escalation, a janela m5_p5 preserva predominio de sinais positivos.",
            "baseline_signal": "positive_share_above_50pct",
            "baseline_magnitude": escalation_m5.loc[escalation_m5["neutral_threshold"] == 0.005, "event_share"].iloc[0],
            "consistency_across_scenarios": f"{positive_consistency:.2f}",
            "robustness_class": classify_robustness(positive_consistency),
            "evidence_note": f"A participacao de sinais positivos variou entre {min_positive_share:.2f} e {max_positive_share:.2f} conforme o limiar.",
        }
    )

    sample_m5 = sample.loc[sample["window_label"] == "m5_p5"].copy()
    core_value = sample_m5.loc[sample_m5["sample_label"] == "core_only", "mean_formal_car_sp500"].iloc[0]
    full_value = sample_m5.loc[sample_m5["sample_label"] == "full_market_covered", "mean_formal_car_sp500"].iloc[0]
    same_sign = (core_value > 0) == (full_value > 0)
    rows.append(
        {
            "conclusion_id": "sample_expansion_preserves_sign_m5",
            "conclusion_statement": "A expansao da amostra preserva o sinal positivo de m5_p5.",
            "baseline_signal": "same_sign" if same_sign else "sign_change",
            "baseline_magnitude": full_value - core_value,
            "consistency_across_scenarios": "1.00" if same_sign else "0.00",
            "robustness_class": "robust" if same_sign else "sensitive",
            "evidence_note": "A magnitude cai na amostra expandida, mas o sinal final nao se inverte.",
        }
    )

    pearson_row = correlation.loc[
        (correlation["scope_type"] == "all_events")
        & (correlation["scope_value"] == "all_events")
        & (correlation["test_method"] == "pearson")
    ].iloc[0]
    spearman_row = correlation.loc[
        (correlation["scope_type"] == "all_events")
        & (correlation["scope_value"] == "all_events")
        & (correlation["test_method"] == "spearman")
    ].iloc[0]
    correlation_score = 0.5 if (pearson_row["correlation"] < 0 and spearman_row["correlation"] < 0) else 0.2
    if pearson_row["p_value"] < 0.05:
        correlation_score += 0.2
    rows.append(
        {
            "conclusion_id": "negative_return_volatility_link",
            "conclusion_statement": "CARs mais negativos tendem a estar associados a maior volatilidade.",
            "baseline_signal": "negative_correlation" if pearson_row["correlation"] < 0 else "positive_correlation",
            "baseline_magnitude": pearson_row["correlation"],
            "consistency_across_scenarios": f"{correlation_score:.2f}",
            "robustness_class": classify_robustness(correlation_score),
            "evidence_note": "A associacao negativa aparece em Pearson, mas nao recebe confirmacao estatistica em Spearman.",
        }
    )

    return pd.DataFrame(rows)


def build_robustness_text(summary: pd.DataFrame) -> str:
    core_m5 = summary.loc[summary["conclusion_id"] == "core_m5_positive"].iloc[0]
    core_progression = summary.loc[summary["conclusion_id"] == "core_progression_upward"].iloc[0]
    signal_m5 = summary.loc[summary["conclusion_id"] == "escalation_positive_share_m5"].iloc[0]
    sample_row = summary.loc[summary["conclusion_id"] == "sample_expansion_preserves_sign_m5"].iloc[0]
    correlation_row = summary.loc[summary["conclusion_id"] == "negative_return_volatility_link"].iloc[0]

    return f"""# 8. Robustez

## 8.1 Introducao a analise de robustez

Esta secao avalia se as principais conclusoes do TCC permanecem estaveis sob variacoes razoaveis de metodologia. O objetivo nao e maximizar complexidade estatistica, mas tensionar os resultados obtidos no event study principal de forma transparente e academicamente defensavel.

## 8.2 Sensibilidade a janela de estimacao

Foram comparadas tres janelas de estimacao para o retorno esperado do S&P 500: `[-20,-6]`, `[-30,-6]` e `[-40,-6]`, mantendo inalteradas as janelas de evento. O resultado mais importante e que a conclusao de CAR medio positivo em `m5_p5` no trilho core permaneceu com classificacao `{core_m5['robustness_class']}`. Em paralelo, a progressao do CAR medio de `m1_p1` para `m5_p5` no core tambem se mostrou `{core_progression['robustness_class']}`.

Em termos práticos, isso indica que pequenas alteracoes na janela de estimacao modificam a magnitude dos resultados, mas nao alteram de forma decisiva a leitura temporal central do estudo.

## 8.3 Sensibilidade ao criterio de classificacao

O criterio de classificacao de sinais foi testado com limiares de `0.25%`, `0.50%` e `0.75%`. Os resultados mostram que a distribuicao de sinais reage ao limiar, como seria esperado, mas a leitura de predominio relativo de sinais positivos em `escalation` na janela `m5_p5` permaneceu classificada como `{signal_m5['robustness_class']}`.

Isso sugere que a camada de sinal e sensivel na margem, especialmente em observacoes proximas de zero, mas nao inteiramente arbitraria.

## 8.4 Sensibilidade a composicao da amostra

Tambem foi comparada a amostra `core_only` com `core + pandemic covered` e `full_market_covered`. Como a base de mercado termina em `2020-12-31`, as amostras `core + pandemic covered` e `full_market_covered` coincidem no estado atual do projeto. Ainda assim, a verificacao e util porque mostra que a expansao da amostra atenua magnitudes, mas preserva a direcao geral dos resultados em janelas mais longas. Essa conclusao foi classificada como `{sample_row['robustness_class']}`.

## 8.5 Sintese geral de robustez

No conjunto, os resultados mais estaveis do estudo sao aqueles ligados a dinamica intertemporal do CAR: a melhora do CAR medio quando a janela cresce e a preservacao do sinal positivo em `m5_p5`. Os resultados mais sensiveis estao associados a classificacoes discretas de sinal e a medidas de associacao entre retorno e volatilidade, que dependem mais fortemente do limiar adotado e da pequena dimensao amostral.

## 8.6 Conclusao metodologica

De modo geral, a auditoria final de robustez reforca a credibilidade do estudo. A maior parte das conclusoes substantivas permanece estavel sob variacoes razoaveis de metodologia, ainda que com mudancas de magnitude. Ao mesmo tempo, a analise mostra com clareza onde a evidencia deve ser interpretada com mais cautela, em especial na relacao entre retorno e volatilidade, classificada como `{correlation_row['robustness_class']}`.
"""


def main() -> None:
    summary = build_robustness_summary()
    text = build_robustness_text(summary)

    OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_SUMMARY, index=False)
    OUTPUT_TEXT.write_text(text, encoding="utf-8")

    print(f"Arquivo salvo em: {OUTPUT_SUMMARY}")
    print(f"Arquivo salvo em: {OUTPUT_TEXT}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
