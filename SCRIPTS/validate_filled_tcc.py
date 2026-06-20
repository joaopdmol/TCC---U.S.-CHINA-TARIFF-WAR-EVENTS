from pathlib import Path
import re

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
FILLED_FILE = BASE_DIR / "FINAL_TCC_DRAFT_FILLED.md"
ML_RESULTS_FILE = BASE_DIR / "DATA" / "ml_model_results.csv"
EVENTS_FILE = BASE_DIR / "DATA" / "events.csv"
EVENTS_MASTER_FILE = BASE_DIR / "DATA" / "events_master.csv"
EVENT_WINDOWS_MASTER_COVERAGE_FILE = BASE_DIR / "DATA" / "event_windows_master_coverage.csv"
ML_DATASET_FILE = BASE_DIR / "DATA" / "ml_dataset.csv"

UNRESOLVED_PATTERNS = [
    "[VERIFY",
    "VERIFY:",
    "[INSERT FIGURE",
    "INSERT FIGURE:",
    "Figure X",
    "exact value",
    "higher/lower",
    "more/less",
    "consistent/varying",
    "more pronounced/attenuated",
    "statistical significance",
]

DISALLOWED_METHOD_TERMS = [
    "CAPM",
    "Fama-French",
    "Difference-in-Differences",
    "Synthetic Control",
    "NLP",
    "Natural Language Processing",
    "XGBoost",
    "Prophet",
    "LSTM",
    "Deep Learning",
    "Transformers",
]

ALLOWED_NON_APPLIED_CONTEXT = [
    "future",
    "promising",
    "current scope limits",
    "rather than",
    "not implemented",
    "not claim",
    "without claiming",
    "limitations",
    "might",
    "could",
]


def ok(message: str) -> None:
    print(f"OK: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"ERRO: {message}")


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def load_text() -> str:
    if not FILLED_FILE.exists():
        raise FileNotFoundError(f"ERRO: arquivo nao encontrado: {FILLED_FILE}")
    text = FILLED_FILE.read_text(encoding="utf-8")
    require(text.strip() != "", "FINAL_TCC_DRAFT_FILLED.md esta vazio.")
    return text


def validate_no_unresolved_markers(text: str) -> None:
    remaining = [pattern for pattern in UNRESOLVED_PATTERNS if pattern in text]
    require(not remaining, f"marcadores pendentes encontrados: {remaining}")
    ok("sem placeholders pendentes")


def validate_figures(text: str) -> None:
    figure_paths = re.findall(r"!\[[^\]]*\]\((FIGURES/[^)]+)\)", text)
    require(figure_paths, "nenhuma figura markdown encontrada no texto preenchido.")
    missing = [path for path in figure_paths if not (BASE_DIR / path).exists()]
    require(not missing, f"figuras referenciadas nao existem: {missing}")
    ok("todas as figuras referenciadas existem")


def validate_duplicate_sections(text: str) -> None:
    lines = [line.strip() for line in text.splitlines()]
    duplicate_checks = {
        "6.5 Future Research": lines.count("6.5 Future Research"),
        "6.6 Final Statement": lines.count("6.6 Final Statement"),
        "References": lines.count("References"),
    }
    errors = [f"{heading} aparece {count} vezes" for heading, count in duplicate_checks.items() if count != 1]
    require(not errors, "; ".join(errors))
    ok("sem duplicacoes obvias de secoes finais")


def validate_ml_metrics(text: str) -> None:
    ml_results = pd.read_csv(ML_RESULTS_FILE)
    expected = {
        "baseline_majority": ["accuracy"],
        "logistic_regression": ["accuracy", "f1"],
        "random_forest": ["accuracy", "f1"],
    }
    missing_values = []
    for model, metrics in expected.items():
        row = ml_results.loc[ml_results["model"] == model]
        require(not row.empty, f"modelo ausente em ml_model_results.csv: {model}")
        row = row.iloc[0]
        for metric in metrics:
            value = format_pct(row[metric])
            if value not in text:
                missing_values.append(f"{model}.{metric}={value}")
    require(not missing_values, f"metricas de ML nao encontradas no texto: {missing_values}")
    ok("metricas de ML conferem com ml_model_results.csv")


def validate_event_counts(text: str) -> None:
    core_events = pd.read_csv(EVENTS_FILE)
    events_master = pd.read_csv(EVENTS_MASTER_FILE)
    coverage = pd.read_csv(EVENT_WINDOWS_MASTER_COVERAGE_FILE)
    ml_dataset = pd.read_csv(ML_DATASET_FILE)

    complete_covered_events = coverage.loc[
        coverage["is_complete"].astype(str).str.lower().eq("true"),
        "event_id",
    ].nunique()
    ml_rows = len(ml_dataset)
    ml_events = ml_dataset["event_id"].nunique()

    dataset_match = re.search(r"The dataset comprises (\d+) identified tariff-war events", text)
    require(dataset_match is not None, "frase de contagem da base de eventos nao encontrada.")
    require(
        int(dataset_match.group(1)) == len(events_master),
        f"contagem da cronologia completa deveria ser {len(events_master)}.",
    )

    # Se esses numeros forem citados no texto, eles precisam ser os numeros reais do projeto.
    cited_numbers = {
        len(core_events): "core events",
        len(events_master): "master events",
        complete_covered_events: "market-covered master events",
        ml_rows: "ML event-window observations",
        ml_events: "ML covered events",
    }
    require(len(core_events) == 17, "events.csv deveria conter 17 eventos.")
    require(len(events_master) == 67, "events_master.csv deveria conter 67 eventos.")
    require(complete_covered_events == 24, "cobertura master deveria conter 24 eventos completos.")
    require(ml_rows == 72 and ml_events == 24, "ml_dataset.csv deveria conter 72 observacoes e 24 eventos.")
    ok("contagens de eventos e ML conferidas nos CSVs")


def validate_references(text: str) -> None:
    lines = [line.strip() for line in text.splitlines()]
    require("References" in lines, "secao References nao encontrada.")
    reference_index = lines.index("References")
    require(reference_index < len(lines) - 3, "secao References parece vazia.")
    ok("referencias presentes no final do arquivo")


def validate_non_applied_methods(text: str) -> None:
    violations = []
    body_text = text.split("\nReferences\n", maxsplit=1)[0]
    for line_number, line in enumerate(body_text.splitlines(), start=1):
        normalized = line.lower()
        for term in DISALLOWED_METHOD_TERMS:
            if term.lower() in normalized:
                allowed = any(context in normalized for context in ALLOWED_NON_APPLIED_CONTEXT)
                if not allowed:
                    violations.append(f"linha {line_number}: {term}")
    require(
        not violations,
        "metodos nao implementados parecem citados como aplicados: " + "; ".join(violations),
    )
    ok("metodos nao implementados aparecem apenas em contexto permitido")


def main() -> None:
    text = load_text()
    ok("FINAL_TCC_DRAFT_FILLED.md encontrado")
    validate_no_unresolved_markers(text)
    validate_figures(text)
    validate_duplicate_sections(text)
    validate_ml_metrics(text)
    validate_event_counts(text)
    validate_references(text)
    validate_non_applied_methods(text)
    print("\nVALIDACAO CONCLUIDA")


if __name__ == "__main__":
    main()
