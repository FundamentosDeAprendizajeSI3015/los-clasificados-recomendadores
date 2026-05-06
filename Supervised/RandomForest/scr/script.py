from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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
        description="Script general RandomForest: train/evaluate para los 4 targets"
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
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
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
        "model": "RandomForestClassifier",
        "n_rows": int(len(df)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    report_dir = Path(__file__).resolve().parents[1] / "reports" / target
    report_dir.mkdir(parents=True, exist_ok=True)

    model_path = report_dir / "model.joblib"
    joblib.dump(pipeline, model_path)

    # Guardar métricas del test set como evaluation.json (fuente de verdad para comparativas)
    eval_path = report_dir / "evaluation.json"
    eval_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[{target}] Modelo guardado en: {model_path}")
    print(f"[{target}] Evaluacion (test set) guardada en: {eval_path}")


def evaluate_target(df: pd.DataFrame, target: str) -> None:
    validate_feature_columns(df)

    report_dir = Path(__file__).resolve().parents[1] / "reports" / target
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
