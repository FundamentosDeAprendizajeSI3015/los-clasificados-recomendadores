from __future__ import annotations

import argparse
import os
import json
import shutil
import sys
from pathlib import Path
import tempfile

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    """Define los argumentos de entrada del script."""
    parser = argparse.ArgumentParser(
        description="Script general Regresion Logistica: train/evaluate para los 4 targets"
    )
    parser.add_argument(
        "--data",
        default="load/dataset.csv",
        help="Ruta al CSV de entrada (por defecto: load/dataset.csv)",
    )
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
    parser.add_argument("--val-size", type=float, default=0.2, help="Proporcion para validacion")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla")
    return parser.parse_args()


def build_pipeline(random_state: int) -> Pipeline:
    """Construye el pipeline de preprocesamiento y clasificación."""
    preprocessor = ColumnTransformer(
        transformers=[
            # Variables numéricas: imputación con mediana y escalado.
            ("num", Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ), NUMERIC_FEATURES),
            # Variables categóricas: imputación con moda y codificación one-hot.
            ("cat", Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ), CATEGORICAL_FEATURES),
        ]
    )

    model = LogisticRegression(
        random_state=random_state,
        max_iter=2000,
        class_weight="balanced",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def validate_feature_columns(df: pd.DataFrame) -> None:
    """Verifica que el DataFrame tenga las columnas de entrada requeridas."""
    missing = [c for c in (NUMERIC_FEATURES + CATEGORICAL_FEATURES) if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de entrada: {missing}")


def load_dataframe_via_module(csv_path: Path) -> pd.DataFrame:
    """Carga el CSV usando load.load.load_data sin modificar ese módulo."""
    proj_root = Path(__file__).resolve().parents[3]
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))

    from load.load import load_data

    # Copia temporalmente el CSV a dataset.csv porque load_data lee ese nombre fijo.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dataset_path = Path(temp_dir) / "dataset.csv"
        shutil.copyfile(csv_path, temp_dataset_path)

        current_dir = Path.cwd()
        try:
            os.chdir(temp_dir)
            return load_data()
        finally:
            os.chdir(current_dir)


def train_target(df: pd.DataFrame, target: str, args: argparse.Namespace) -> None:
    """Entrena un modelo por target y guarda modelo y métricas."""
    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [target]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para {target}: {missing}")

    # Separar variables explicativas y la etiqueta objetivo.
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[target]

    if args.test_size + args.val_size >= 1.0:
        raise ValueError("La suma de --test-size y --val-size debe ser menor que 1.0")

    # Primer corte: reservar el conjunto de prueba.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # Segundo corte: dividir el resto en entrenamiento y validación.
    val_rel = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_rel,
        random_state=args.random_state,
        stratify=y_temp,
    )

    pipeline = build_pipeline(args.random_state)
    pipeline.fit(X_train, y_train)

    # Se reportan métricas sobre validación; el test queda reservado para uso externo.
    y_pred = pipeline.predict(X_val)
    metrics = {
        "target": target,
        "model": "LogisticRegression",
        "n_rows": int(len(df)),
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "f1_macro": float(f1_score(y_val, y_pred, average="macro")),
        "classification_report": classification_report(y_val, y_pred, output_dict=True),
    }

    report_dir = Path(__file__).resolve().parents[1] / "reports" / target
    report_dir.mkdir(parents=True, exist_ok=True)

    model_path = report_dir / "model.joblib"
    metrics_path = report_dir / "metrics.json"

    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[{target}] Modelo guardado en: {model_path}")
    print(f"[{target}] Metricas de train guardadas en: {metrics_path}")


def evaluate_target(df: pd.DataFrame, target: str) -> None:
    """Carga el modelo entrenado y genera predicciones/evaluación."""
    validate_feature_columns(df)

    # Los artefactos del modelo se guardan junto a la carpeta del script.
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

    if target in df.columns:
        y_true = df[target]
        metrics = {
            "target": target,
            "model": "LogisticRegression",
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
    """Punto de entrada principal: carga datos y ejecuta train/evaluate."""
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(f"No existe el CSV de entrada: {data_path}")

    # Cargar el dataset etiquetado que se pasa por --data.
    train_df = load_dataframe_via_module(data_path)

    if args.task in {"train", "both"}:
        for target in TARGETS:
            train_target(train_df, target, args)

    if args.task in {"evaluate", "both"}:
        eval_data_path = Path(args.eval_data) if args.eval_data else data_path

        if not eval_data_path.exists():
            raise FileNotFoundError(f"No existe el CSV de evaluacion: {eval_data_path}")

        eval_df = load_dataframe_via_module(eval_data_path)
        for target in TARGETS:
            evaluate_target(eval_df, target)


if __name__ == "__main__":
    main()
