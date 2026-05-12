# Camada exploratoria de Machine Learning

## 1. Objetivo da camada exploratoria de ML

Esta secao apresenta uma camada complementar de Machine Learning para classificar o sinal do impacto financeiro dos eventos da guerra tarifaria EUA-China. O objetivo nao e substituir o event study, mas verificar se variaveis simples de caracterizacao do evento, da janela e da volatilidade local ajudam a distinguir eventos com impacto negativo de eventos sem impacto negativo.

## 2. Construcao do dataset

O dataset `DATA/ml_dataset.csv` foi construido a partir do trilho expandido com cobertura de mercado. A unidade de observacao e a combinacao evento-janela, totalizando 72 observacoes, 24 eventos cobertos e as janelas `m1_p1`, `m3_p3` e `m5_p5`.

O alvo principal e `target_negative_impact`, definido como 1 quando o `formal_car_sp500` e negativo e 0 caso contrario. O CAR formal nao foi usado como variavel explicativa. As features incluem metadados do evento, regime, janela, volatilidade local, numero de observacoes e indicadores de cobertura/estimacao.

## 3. Modelos utilizados

Foram estimados dois modelos simples e interpretaveis: Regressao Logistica e Random Forest. A Regressao Logistica oferece uma referencia linear e mais transparente. A Random Forest permite capturar alguma nao linearidade, mas foi mantida rasa e regularizada para reduzir sobreajuste em uma amostra pequena.

## 4. Metricas de avaliacao

A avaliacao usou `stratified_group_kfold` com 5 dobras. As metricas reportadas foram accuracy, precision, recall e F1, sempre comparadas a um baseline de classe majoritaria. O baseline obteve accuracy de 55.56%.

## 5. Resultados

A Regressao Logistica obteve accuracy de 43.06%, precision de 37.14%, recall de 40.62% e F1 de 38.81%. A Random Forest obteve accuracy de 41.67%, precision de 32.14%, recall de 28.12% e F1 de 30.00%.

Na validacao cruzada, os modelos nao superaram o baseline de classe majoritaria em accuracy. Esse resultado reforca que a camada de ML deve ser interpretada como diagnostico exploratorio, e nao como evidencia de capacidade preditiva robusta.

O melhor desempenho em F1 foi do modelo `logistic_regression`, com F1 de 38.81%. Ainda assim, os resultados devem ser lidos como evidencias exploratorias. A interpretacao adequada e que a amostra disponivel ainda nao sustenta classificacao robusta do sinal financeiro, embora as importancias de features possam ajudar a organizar hipoteses para discussao.

## 6. Limitacoes

A principal limitacao e o tamanho amostral reduzido. As observacoes tambem derivam de eventos historicos proximos e de tres janelas por evento, o que limita a independencia estatistica. Alem disso, algumas features, como volatilidade da propria janela, sao informativas para classificacao exploratoria, mas nao caracterizam uma previsao ex ante pura.

## 7. Por que ML e complementar ao event study

O event study continua sendo a metodologia principal porque mede diretamente retornos anormais e CAR em torno dos eventos. O ML entra apenas como camada complementar para organizar padroes e avaliar se os atributos ja calculados ajudam a classificar o sinal do impacto. Portanto, a interpretacao substantiva do TCC permanece ancorada no event study, nos testes estatisticos e nas analises de robustez.
