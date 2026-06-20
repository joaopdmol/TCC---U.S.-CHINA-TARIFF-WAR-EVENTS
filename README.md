# Trade Wars Data Analysis - TCC

Repositório do Trabalho de Conclusão de Curso de João Pedro Duarte Mól sobre impactos financeiros da guerra tarifária entre Estados Unidos e China.

Este projeto contém o pipeline empírico usado no TCC, incluindo organização da cronologia de eventos, construção de bases financeiras, estudo de eventos, retornos anormais, CAR formal simples, volatilidade, análises de robustez, testes estatísticos e uma camada exploratória de Machine Learning.

## Estrutura do projeto

- `DATA/`: bases de dados intermediárias e finais em CSV.
- `SCRIPTS/`: scripts reproduzíveis para construção, validação, visualização e análise.
- `FIGURES/`: figuras geradas para análise e uso na monografia.
- `NOTEBOOKS/`: notebooks de inspeção e apoio analítico.
- `partes/generated/`: tabelas LaTeX geradas para incorporação no texto final.
- `RESULTS_SECTION.md`, `ROBUSTNESS_SECTION.md`, `ML_SECTION.md` e `METHODOLOGY_SECTION.md`: seções textuais de apoio à monografia.

## Pipeline principal

O fluxo empírico do projeto segue a sequência:

1. Construção e validação da cronologia de eventos.
2. Download e preparação das séries financeiras.
3. Construção de janelas de evento.
4. Cálculo de retornos esperados, retornos anormais e CAR formal simples.
5. Geração de estatísticas descritivas, volatilidade e testes estatísticos leves.
6. Análises de robustez e sensibilidade.
7. Camada exploratória complementar de Machine Learning.
8. Geração de figuras e tabelas finais.

## Como reproduzir as validações principais

No Windows, usando o ambiente virtual local:

```powershell
.\.venv\Scripts\python.exe SCRIPTS\validate_events.py
.\.venv\Scripts\python.exe SCRIPTS\validate_market_features.py
.\.venv\Scripts\python.exe SCRIPTS\validate_event_windows.py
.\.venv\Scripts\python.exe SCRIPTS\validate_event_study.py
.\.venv\Scripts\python.exe SCRIPTS\validate_event_study_master.py
.\.venv\Scripts\python.exe SCRIPTS\validate_final_outputs.py
.\.venv\Scripts\python.exe SCRIPTS\validate_robustness.py
.\.venv\Scripts\python.exe SCRIPTS\validate_ml_outputs.py
```

## Observações metodológicas

A análise principal usa a amostra `core` de 17 eventos entre 2018 e 2020, enquanto a cronologia expandida é usada como análise de robustez quando há cobertura de mercado disponível. O estudo de eventos utiliza uma abordagem formal simples, com retorno esperado médio em janela de estimação anterior ao evento, e interpreta os resultados de forma conservadora.

## Autor acadêmico

João Pedro Duarte Mól.
