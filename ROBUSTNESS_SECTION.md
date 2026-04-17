# 8. Robustez

## 8.1 Introducao a analise de robustez

Esta secao avalia se as principais conclusoes do TCC permanecem estaveis sob variacoes razoaveis de metodologia. O objetivo nao e maximizar complexidade estatistica, mas tensionar os resultados obtidos no event study principal de forma transparente e academicamente defensavel.

## 8.2 Sensibilidade a janela de estimacao

Foram comparadas tres janelas de estimacao para o retorno esperado do S&P 500: `[-20,-6]`, `[-30,-6]` e `[-40,-6]`, mantendo inalteradas as janelas de evento. O resultado mais importante e que a conclusao de CAR medio positivo em `m5_p5` no trilho core permaneceu com classificacao `robust`. Em paralelo, a progressao do CAR medio de `m1_p1` para `m5_p5` no core tambem se mostrou `robust`.

Em termos práticos, isso indica que pequenas alteracoes na janela de estimacao modificam a magnitude dos resultados, mas nao alteram de forma decisiva a leitura temporal central do estudo.

## 8.3 Sensibilidade ao criterio de classificacao

O criterio de classificacao de sinais foi testado com limiares de `0.25%`, `0.50%` e `0.75%`. Os resultados mostram que a distribuicao de sinais reage ao limiar, como seria esperado, mas a leitura de predominio relativo de sinais positivos em `escalation` na janela `m5_p5` permaneceu classificada como `robust`.

Isso sugere que a camada de sinal e sensivel na margem, especialmente em observacoes proximas de zero, mas nao inteiramente arbitraria.

## 8.4 Sensibilidade a composicao da amostra

Tambem foi comparada a amostra `core_only` com `core + pandemic covered` e `full_market_covered`. Como a base de mercado termina em `2020-12-31`, as amostras `core + pandemic covered` e `full_market_covered` coincidem no estado atual do projeto. Ainda assim, a verificacao e util porque mostra que a expansao da amostra atenua magnitudes, mas preserva a direcao geral dos resultados em janelas mais longas. Essa conclusao foi classificada como `robust`.

## 8.5 Sintese geral de robustez

No conjunto, os resultados mais estaveis do estudo sao aqueles ligados a dinamica intertemporal do CAR: a melhora do CAR medio quando a janela cresce e a preservacao do sinal positivo em `m5_p5`. Os resultados mais sensiveis estao associados a classificacoes discretas de sinal e a medidas de associacao entre retorno e volatilidade, que dependem mais fortemente do limiar adotado e da pequena dimensao amostral.

## 8.6 Conclusao metodologica

De modo geral, a auditoria final de robustez reforca a credibilidade do estudo. A maior parte das conclusoes substantivas permanece estavel sob variacoes razoaveis de metodologia, ainda que com mudancas de magnitude. Ao mesmo tempo, a analise mostra com clareza onde a evidencia deve ser interpretada com mais cautela, em especial na relacao entre retorno e volatilidade, classificada como `moderately sensitive`.
