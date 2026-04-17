# 6. Resultados

## 6.1 Introducao da secao de resultados

Esta secao apresenta os resultados finais do event study em duas camadas complementares. A primeira corresponde a analise principal, baseada na amostra core de 17 eventos entre 2018 e 2020-02-14. A segunda corresponde a uma analise de robustez com a base expandida de eventos, organizada em `events_master.csv` e restrita aos eventos para os quais existe cobertura efetiva de mercado em `market_features.csv`.

A estrategia adotada preserva a separacao entre documentacao historica e inferencia financeira. Embora a cronologia mestre contenha 67 eventos, apenas 24 contam com cobertura de mercado suficiente para integrar o event study expandido. Os 43 eventos restantes permanecem importantes para o enquadramento historico, mas nao entram na inferencia porque a base de mercado termina em 2020-12-31.

## 6.2 Resultados do event study (core)

Na amostra principal, o CAR formal medio do S&P 500 foi de -0.38% em `m1_p1`, -0.27% em `m3_p3` e 0.35% em `m5_p5`. O padrao agregado sugere um movimento de reversao ao longo do tempo, com melhora do CAR medio nas janelas mais longas.

Essa leitura ganha forca quando se observa a distribuicao dos sinais. Em `m1_p1`, 47.06% dos eventos foram classificados como negativos e 35.29% como positivos. Em `m5_p5`, a participacao de eventos positivos sobe para 58.82%, enquanto a de eventos negativos fica em 35.29%. Em termos economicos, isso e consistente com a ideia de que parte da reacao inicial do mercado americano foi posteriormente amortecida ou revertida em janelas mais amplas.

## 6.3 Robustez com amostra expandida

A analise de robustez utiliza a cronologia expandida, mas considera apenas os eventos com cobertura efetiva de mercado. Nesse recorte, o CAR formal medio foi de -0.23% em `m1_p1`, -0.16% em `m3_p3` e 0.11% em `m5_p5`, com 24 eventos por janela.

O comportamento agregado da amostra expandida coberta aponta para um movimento de reversao ao longo do tempo, com melhora do CAR medio nas janelas mais longas. Em relacao ao trilho principal, os valores medios ficam menos extremos, o que e esperado quando se incorporam eventos adicionais do inicio do periodo pandemico. Ainda assim, o padrao geral de melhora relativa nas janelas mais longas permanece visivel, o que sugere que o resultado principal nao depende apenas do subconjunto original de 17 eventos.

## 6.4 Comparacao core vs expanded

Quando se compara diretamente a amostra principal com a amostra expandida coberta, o que muda e a intensidade media do efeito, nao a direcao geral da leitura temporal. Em `m5_p5`, por exemplo, o CAR medio passa de 0.35% no core para 0.11% na amostra expandida, uma diferenca de -0.25 p.p..

Essa comparacao sugere que a inclusao dos eventos adicionais reduz a magnitude media do CAR positivo nas janelas mais longas, mas nao elimina a evidencia de acomodacao do choque ao longo do tempo. Em outras palavras, a robustez expandida torna o resultado mais conservador, sem alterar de forma substantiva a narrativa central do trabalho.

## 6.5 Relacao retorno vs volatilidade

Os resultados da correlacao entre retorno acumulado e volatilidade apontam para uma associacao negativa moderada. No conjunto agregado, a correlacao de Pearson entre `formal_car_sp500` e `volatility_sp500` foi de -0.299 com `p = 0.033`, enquanto a correlacao de Spearman foi de -0.165 com `p = 0.249`.

Isso sugere que janelas com CAR mais negativo tendem, em media, a conviver com maior volatilidade, embora a robustez desse padrao deva ser interpretada com cautela. A significancia aparece no coeficiente de Pearson, mas nao se repete no coeficiente de Spearman, o que indica um sinal empirico interessante, porem ainda moderado.

## 6.6 Interpretacao dos testes estatisticos

Os testes entre janelas nao indicaram evidencia estatisticamente significativa de diferenca entre `m1_p1` e `m3_p3` no conjunto total (`p = 0.868` no teste t pareado). O mesmo ocorre para `m3_p3` versus `m5_p5` no teste t pareado (`p = 0.261`). No teste nao parametrico de Wilcoxon para essa segunda comparacao, o p-value foi `p = 0.064`, o que caracteriza no maximo evidencia sugestiva, e nao conclusiva, de diferenca entre janelas.

Nos testes entre grupos, tambem nao houve evidencia estatisticamente significativa de diferenca entre eventos de `escalation` e `relief` em nenhuma das tres janelas. Os p-values dos testes t de Welch foram 0.776 em `m1_p1`, 0.258 em `m3_p3` e 0.270 em `m5_p5`. Portanto, a leitura mais adequada nao e a de igualdade comprovada entre grupos, mas sim a de ausencia de evidencia estatistica forte para diferenca dada a amostra disponivel.

## 6.7 Limitacoes metodologicas

Ha pelo menos tres limitacoes importantes. Primeiro, o tamanho amostral continua relativamente pequeno, sobretudo quando a analise e particionada por grupo de evento. Segundo, a cronologia mestre vai ate 2026, mas a cobertura de mercado vai apenas ate 2020-12-31, o que impede incorporar a totalidade dos eventos expandidos na inferencia financeira. Terceiro, o retorno esperado foi modelado de forma simples, como media historica do proprio S&P 500 em uma janela de estimacao anterior ao evento, sem recorrer a market model, CAPM ou fatores adicionais.

Essas escolhas sao deliberadamente conservadoras. Elas tornam a estrategia empirica mais transparente e facilmente defensavel, mas ao mesmo tempo limitam a ambicao inferencial do estudo.

## 6.8 Sintese dos achados

Em sintese, a analise principal indica que os efeitos medios da guerra tarifaria sobre o mercado americano foram mais negativos ou ambivalentes em janelas curtas, mas tenderam a se tornar menos adversos em janelas mais amplas. A amostra expandida com cobertura de mercado confirma essa direcao geral, embora com magnitudes mais moderadas.

Os testes estatisticos nao fornecem evidencia forte de diferencas entre grupos ou entre janelas no nivel convencional de 5 por cento, o que recomenda cautela interpretativa. Ainda assim, o conjunto dos resultados descritivos, das correlacoes com volatilidade e da robustez expandida oferece evidencia consistente com a leitura de que os choques comerciais produziram reacoes financeiras heterogeneas e parcialmente reversiveis ao longo do tempo.
