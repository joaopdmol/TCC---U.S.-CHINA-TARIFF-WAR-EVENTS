from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DATA"

ML_DATASET_FILE = DATA_DIR / "ml_dataset.csv"
MODEL_RESULTS_FILE = DATA_DIR / "ml_model_results.csv"
CONFUSION_MATRICES_FILE = DATA_DIR / "ml_confusion_matrices.csv"
FEATURE_IMPORTANCE_FILE = DATA_DIR / "ml_feature_importance.csv"
ML_SECTION_FILE = BASE_DIR / "ML_SECTION.md"

TARGET_COLUMN = "target_negative_impact"
GROUP_COLUMN = "event_id"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "feature_volatility_sp500",
    "feature_n_obs",
    "feature_estimation_n_obs",
    "feature_estimation_sufficient_data",
    "feature_anchor_shift_calendar_days",
    "feature_window_size",
    "feature_event_year",
    "feature_event_month",
    "feature_is_core_sample",
    "feature_is_pandemic_period",
    "feature_is_post_pandemic",
    "feature_include_in_primary_analysis",
]
CATEGORICAL_FEATURES = [
    "feature_window_label",
    "feature_event_group",
    "feature_event_type",
    "feature_event_regime",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MODEL_LABELS = {
    "baseline_majority": "Baseline majoritaria",
    "logistic_regression": "Regressao Logistica",
    "random_forest": "Random Forest",
}


def load_dataset() -> pd.DataFrame:
    if not ML_DATASET_FILE.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {ML_DATASET_FILE}")

    dataframe = pd.read_csv(ML_DATASET_FILE)
    required_columns = [GROUP_COLUMN, TARGET_COLUMN] + FEATURE_COLUMNS
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes em ml_dataset.csv: {missing_columns}")
    if "formal_car_sp500" in FEATURE_COLUMNS:
        raise ValueError("formal_car_sp500 nao pode ser usado como feature.")
    return dataframe


def build_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", build_preprocessor(scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=4,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def choose_cv(y: pd.Series, groups: pd.Series):
    min_class_count = int(y.value_counts().min())
    n_groups = int(groups.nunique())
    n_splits = min(5, min_class_count, n_groups)
    if n_splits >= 2:
        return (
            StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE),
            "stratified_group_kfold",
            n_splits,
            True,
        )

    n_splits = min(3, min_class_count)
    if n_splits >= 2:
        return (
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE),
            "stratified_kfold_without_grouping",
            n_splits,
            False,
        )

    raise ValueError("Amostra insuficiente para validacao cruzada estratificada.")


def metric_row(
    *,
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    validation_strategy: str,
    n_splits: int,
    n_samples: int,
    n_events: int,
    baseline_accuracy: float,
) -> dict[str, object]:
    return {
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "validation_strategy": validation_strategy,
        "n_splits": n_splits,
        "n_samples": n_samples,
        "n_events": n_events,
        "majority_class": int(y_true.mode().iloc[0]),
        "baseline_accuracy": baseline_accuracy,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def confusion_rows(model_name: str, y_true: pd.Series, y_pred: np.ndarray) -> list[dict[str, object]]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    rows = []
    for actual_index, actual_label in enumerate([0, 1]):
        for predicted_index, predicted_label in enumerate([0, 1]):
            rows.append(
                {
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "actual_label": actual_label,
                    "predicted_label": predicted_label,
                    "count": int(matrix[actual_index, predicted_index]),
                }
            )
    return rows


def clean_feature_name(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "")


def extract_feature_importance(model_name: str, pipeline: Pipeline, x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    fitted_pipeline = pipeline.fit(x, y)
    feature_names = [
        clean_feature_name(name)
        for name in fitted_pipeline.named_steps["preprocessor"].get_feature_names_out()
    ]
    estimator = fitted_pipeline.named_steps["model"]

    if model_name == "logistic_regression":
        signed_weights = estimator.coef_[0]
        importance = np.abs(signed_weights)
    else:
        signed_weights = np.full(len(feature_names), np.nan)
        importance = estimator.feature_importances_

    return pd.DataFrame(
        {
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "feature": feature_names,
            "importance": importance,
            "signed_weight": signed_weights,
        }
    ).sort_values(["model", "importance"], ascending=[True, False])


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_ml_section(results: pd.DataFrame, dataset: pd.DataFrame) -> str:
    baseline = results.loc[results["model"] == "baseline_majority"].iloc[0]
    logistic = results.loc[results["model"] == "logistic_regression"].iloc[0]
    forest = results.loc[results["model"] == "random_forest"].iloc[0]
    best_model = results.loc[results["model"].isin(["logistic_regression", "random_forest"])].sort_values(
        "f1",
        ascending=False,
    ).iloc[0]
    if best_model["accuracy"] <= baseline["accuracy"]:
        performance_note = (
            "Na validacao cruzada, os modelos nao superaram o baseline de classe majoritaria em accuracy. "
            "Esse resultado reforca que a camada de ML deve ser interpretada como diagnostico exploratorio, "
            "e nao como evidencia de capacidade preditiva robusta."
        )
    else:
        performance_note = (
            "Na validacao cruzada, ao menos um modelo superou o baseline de classe majoritaria em accuracy. "
            "Mesmo assim, a interpretacao permanece exploratoria devido ao tamanho amostral reduzido."
        )

    return f"""# Camada exploratoria de Machine Learning

## 1. Objetivo da camada exploratoria de ML

Esta secao apresenta uma camada complementar de Machine Learning para classificar o sinal do impacto financeiro dos eventos da guerra tarifaria EUA-China. O objetivo nao e substituir o event study, mas verificar se variaveis simples de caracterizacao do evento, da janela e da volatilidade local ajudam a distinguir eventos com impacto negativo de eventos sem impacto negativo.

## 2. Construcao do dataset

O dataset `DATA/ml_dataset.csv` foi construido a partir do trilho expandido com cobertura de mercado. A unidade de observacao e a combinacao evento-janela, totalizando {len(dataset)} observacoes, {dataset['event_id'].nunique()} eventos cobertos e as janelas `m1_p1`, `m3_p3` e `m5_p5`.

O alvo principal e `target_negative_impact`, definido como 1 quando o `formal_car_sp500` e negativo e 0 caso contrario. O CAR formal nao foi usado como variavel explicativa. As features incluem metadados do evento, regime, janela, volatilidade local, numero de observacoes e indicadores de cobertura/estimacao.

## 3. Modelos utilizados

Foram estimados dois modelos simples e interpretaveis: Regressao Logistica e Random Forest. A Regressao Logistica oferece uma referencia linear e mais transparente. A Random Forest permite capturar alguma nao linearidade, mas foi mantida rasa e regularizada para reduzir sobreajuste em uma amostra pequena.

## 4. Metricas de avaliacao

A avaliacao usou `{logistic['validation_strategy']}` com {int(logistic['n_splits'])} dobras. As metricas reportadas foram accuracy, precision, recall e F1, sempre comparadas a um baseline de classe majoritaria. O baseline obteve accuracy de {format_pct(baseline['accuracy'])}.

## 5. Resultados

A Regressao Logistica obteve accuracy de {format_pct(logistic['accuracy'])}, precision de {format_pct(logistic['precision'])}, recall de {format_pct(logistic['recall'])} e F1 de {format_pct(logistic['f1'])}. A Random Forest obteve accuracy de {format_pct(forest['accuracy'])}, precision de {format_pct(forest['precision'])}, recall de {format_pct(forest['recall'])} e F1 de {format_pct(forest['f1'])}.

{performance_note}

O melhor desempenho em F1 foi do modelo `{best_model['model']}`, com F1 de {format_pct(best_model['f1'])}. Ainda assim, os resultados devem ser lidos como evidencias exploratorias. A interpretacao adequada e que a amostra disponivel ainda nao sustenta classificacao robusta do sinal financeiro, embora as importancias de features possam ajudar a organizar hipoteses para discussao.

## 6. Limitacoes

A principal limitacao e o tamanho amostral reduzido. As observacoes tambem derivam de eventos historicos proximos e de tres janelas por evento, o que limita a independencia estatistica. Alem disso, algumas features, como volatilidade da propria janela, sao informativas para classificacao exploratoria, mas nao caracterizam uma previsao ex ante pura.

## 7. Por que ML e complementar ao event study

O event study continua sendo a metodologia principal porque mede diretamente retornos anormais e CAR em torno dos eventos. O ML entra apenas como camada complementar para organizar padroes e avaliar se os atributos ja calculados ajudam a classificar o sinal do impacto. Portanto, a interpretacao substantiva do TCC permanece ancorada no event study, nos testes estatisticos e nas analises de robustez.
"""


def main() -> None:
    dataset = load_dataset()
    x = dataset[FEATURE_COLUMNS].copy()
    y = dataset[TARGET_COLUMN].astype(int)
    groups = dataset[GROUP_COLUMN]

    cv, validation_strategy, n_splits, uses_groups = choose_cv(y, groups)
    majority_class = int(y.mode().iloc[0])
    baseline_pred = np.full(len(y), majority_class)
    baseline_accuracy = accuracy_score(y, baseline_pred)

    results = [
        metric_row(
            model_name="baseline_majority",
            y_true=y,
            y_pred=baseline_pred,
            validation_strategy="majority_class_baseline",
            n_splits=0,
            n_samples=len(dataset),
            n_events=dataset["event_id"].nunique(),
            baseline_accuracy=baseline_accuracy,
        )
    ]
    confusion_output = confusion_rows("baseline_majority", y, baseline_pred)
    importance_frames = []

    for model_name, pipeline in build_models().items():
        if uses_groups:
            predictions = cross_val_predict(pipeline, x, y, cv=cv, groups=groups)
        else:
            predictions = cross_val_predict(pipeline, x, y, cv=cv)

        results.append(
            metric_row(
                model_name=model_name,
                y_true=y,
                y_pred=predictions,
                validation_strategy=validation_strategy,
                n_splits=n_splits,
                n_samples=len(dataset),
                n_events=dataset["event_id"].nunique(),
                baseline_accuracy=baseline_accuracy,
            )
        )
        confusion_output.extend(confusion_rows(model_name, y, predictions))
        importance_frames.append(extract_feature_importance(model_name, pipeline, x, y))

    results_df = pd.DataFrame(results)
    confusion_df = pd.DataFrame(confusion_output)
    importance_df = pd.concat(importance_frames, ignore_index=True)

    MODEL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(MODEL_RESULTS_FILE, index=False)
    confusion_df.to_csv(CONFUSION_MATRICES_FILE, index=False)
    importance_df.to_csv(FEATURE_IMPORTANCE_FILE, index=False)
    ML_SECTION_FILE.write_text(build_ml_section(results_df, dataset), encoding="utf-8")

    print(f"Arquivo salvo em: {MODEL_RESULTS_FILE}")
    print(f"Arquivo salvo em: {CONFUSION_MATRICES_FILE}")
    print(f"Arquivo salvo em: {FEATURE_IMPORTANCE_FILE}")
    print(f"Arquivo salvo em: {ML_SECTION_FILE}")
    print("\nResultados:")
    print(results_df[["model_label", "accuracy", "precision", "recall", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
