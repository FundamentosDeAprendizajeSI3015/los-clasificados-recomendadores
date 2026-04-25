from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

TARGETS = [
    "genero_libro_rec",
    "genero_musical_rec",
    "genero_serie_rec",
    "tipo_vino_rec",
]
NUMERIC_FEATURES = [
    "edad",
    "engagement_promedio",
    "valence_musical_pref",
    "energia_musical_pref",
]
CATEGORICAL_FEATURES = [
    "hora_lectura_preferida",
    "velocidad_lectura",
    "contenido_visual_pref",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Script general SVM: train/evaluate para los 4 targets"
    )
    parser.add_argument("--data", required=True, help="Ruta al CSV de entrada")
    parser.add_argument(
        "--task",
        choices=["train", "evaluate", "both"],
        default="both",
        help="Accion a ejecutar",
    )
    parser.add_argument(
        "--eval-data",
        default=None,
        help="CSV para evaluacion/prediccion (si no se indica, usa --data)",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporcion para test")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla")
    return parser.parse_args()


def build_pipeline(random_state: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def validate_feature_columns(df: pd.DataFrame) -> None:
    missing = [c for c in (NUMERIC_FEATURES + CATEGORICAL_FEATURES) if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de entrada: {missing}")


def train_target(df: pd.DataFrame, target: str, args: argparse.Namespace) -> None:
    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [target]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para {target}: {missing}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    pipeline = build_pipeline(args.random_state)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "target": target,
        "model": "SVC",
        "n_rows": int(len(df)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    report_dir = Path(__file__).resolve().parents[2] / "reports" / target
    report_dir.mkdir(parents=True, exist_ok=True)

    model_path = report_dir / "model.joblib"
    metrics_path = report_dir / "metrics.json"

    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[{target}] Modelo guardado en: {model_path}")
    print(f"[{target}] Metricas de train guardadas en: {metrics_path}")


def evaluate_target(df: pd.DataFrame, target: str) -> None:
    validate_feature_columns(df)

    report_dir = Path(__file__).resolve().parents[2] / "reports" / target
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path = report_dir / "model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo para {target}: {model_path}")

    model = joblib.load(model_path)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    predictions = model.predict(X)

    pred_df = df.copy()
    pred_df[f"pred_{target}"] = predictions
    predictions_path = report_dir / "predictions.csv"
    pred_df.to_csv(predictions_path, index=False)
    print(f"[{target}] Predicciones guardadas en: {predictions_path}")

    if target in df.columns:
        y_true = df[target]
        metrics = {
            "target": target,
            "model": "SVC",
            "accuracy": float(accuracy_score(y_true, predictions)),
            "f1_macro": float(f1_score(y_true, predictions, average="macro")),
            "classification_report": classification_report(
                y_true,
                predictions,
                output_dict=True,
            ),
        }
        metrics_path = report_dir / "evaluation.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[{target}] Evaluacion guardada en: {metrics_path}")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(f"No existe el CSV de entrada: {data_path}")

    train_df = pd.read_csv(data_path)

    if args.task in {"train", "both"}:
        for target in TARGETS:
            train_target(train_df, target, args)

    if args.task in {"evaluate", "both"}:
        eval_data_path = Path(args.eval_data) if args.eval_data else data_path
        if not eval_data_path.exists():
            raise FileNotFoundError(f"No existe el CSV de evaluacion: {eval_data_path}")

        eval_df = pd.read_csv(eval_data_path)
        for target in TARGETS:
            evaluate_target(eval_df, target)


if __name__ == "__main__":
    main()
