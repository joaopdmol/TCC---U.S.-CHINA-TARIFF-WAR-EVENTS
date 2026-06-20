# Trade Wars Data Analysis - TCC

## English

This repository contains the reproducible empirical pipeline developed for my undergraduate thesis. The project analyzes U.S. stock market reactions to key events of the United States-China tariff war, with primary emphasis on the S&P 500.

The workflow combines Event Study methodology, abnormal return and Cumulative Abnormal Return calculations, robustness and sensitivity checks, placebo validation, volatility analysis, statistical validation, figure generation, final analytical outputs, and an exploratory Machine Learning layer.

The purpose of the repository is to document the data science structure behind the thesis, including event chronology construction, market data processing, event-window generation, validation scripts, tables, figures, and final outputs. The project does not claim strict causal effects. Instead, it investigates temporal associations between tariff-war events and market behavior under transparent and reproducible methodological conditions.

## Português

Este repositório contém o pipeline empírico reprodutível desenvolvido para o meu Trabalho de Conclusão de Curso. O projeto analisa reações do mercado acionário norte-americano a eventos centrais da guerra tarifária entre Estados Unidos e China, com foco principal no índice S&P 500.

O fluxo de análise combina metodologia de Estudo de Eventos, cálculo de retornos anormais e Retornos Anormais Acumulados, testes de robustez e sensibilidade, validação placebo, análise de volatilidade, validação estatística, geração de figuras, outputs analíticos finais e uma camada exploratória de Machine Learning.

O objetivo do repositório é documentar a estrutura de ciência de dados por trás do TCC, incluindo construção da cronologia de eventos, processamento de dados de mercado, geração de janelas de evento, scripts de validação, tabelas, figuras e outputs finais. O projeto não afirma efeitos causais estritos. Em vez disso, investiga associações temporais entre eventos tarifários e comportamento de mercado sob condições metodológicas transparentes e reprodutíveis.

## Project Structure / Estrutura do Projeto

- `DATA/`: intermediate and final datasets in CSV format.
- `SCRIPTS/`: reproducible scripts for data construction, validation, visualization, and analysis.
- `FIGURES/`: figures generated for empirical analysis and thesis presentation.
- `NOTEBOOKS/`: inspection notebooks used to review intermediate and final outputs.
- `partes/generated/`: generated LaTeX tables for incorporation into the final text.
- `RESULTS_SECTION.md`, `ROBUSTNESS_SECTION.md`, `ML_SECTION.md`, and `METHODOLOGY_SECTION.md`: supporting text sections for the thesis.

## Main Empirical Workflow / Fluxo Empírico Principal

The empirical pipeline follows these steps:

1. Construction and validation of the tariff-war event chronology.
2. Download and preparation of financial market series.
3. Construction of event windows.
4. Calculation of expected returns, abnormal returns, and Cumulative Abnormal Returns.
5. Descriptive analysis, volatility analysis, and lightweight statistical tests.
6. Robustness and sensitivity checks.
7. Placebo validation.
8. Exploratory Machine Learning analysis.
9. Generation of final figures and tables.

## Reproducibility Checks / Validações de Reprodutibilidade

On Windows, using the local virtual environment:

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

## Methodological Note / Observação Metodológica

The primary analysis uses the `core` sample of 17 events between 2018 and 2020. The expanded chronology is used as a robustness layer when market-data coverage is available. The Event Study uses a transparent expected-return specification based on a pre-event estimation window, and the findings are interpreted conservatively as evidence of market reactions and temporal associations.

## Academic Author / Autor Acadêmico

João Pedro Duarte Mól.
