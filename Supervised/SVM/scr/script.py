"""
Script de entrenamiento y evaluación utilizando Máquinas de Vectores de Soporte (SVM).

Este módulo permite entrenar y evaluar modelos de clasificación SVM para los cuatro
objetivos (targets) del sistema de recomendación, explorando distintas configuraciones 
de kernel (lineal, polinómico de grado 2 y 3, y base radial).
"""

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

# --- CONFIGURACIÓN ---
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

# Definición de los 4 experimentos solicitados
KERNEL_CONFIGS = [
    {"name": "kernel_lineal", "kernel": "linear", "degree": 1},
    {"name": "kernel_poly_d2", "kernel": "poly", "degree": 2},
    {"name": "kernel_poly_d3", "kernel": "poly", "degree": 3},
    {"name": "kernel_radial", "kernel": "rbf", "degree": 3},  # degree se ignora en rbf
]

def parse_args() -> argparse.Namespace:
    """Procesa y retorna los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Script SVM: Entrenamiento y evaluación con 4 kernels distintos."
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
        help="CSV para evaluacion (si no se indica, usa --data)",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporcion para test")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla")
    return parser.parse_args()


def build_pipeline(kernel_type: str, degree: int, random_state: int) -> Pipeline:
    """
    Crea el pipeline con el kernel y grado especificados.
    
    Argumentos:
        kernel_type (str): Tipo de kernel a utilizar ('linear', 'poly', 'rbf').
        degree (int): Grado del kernel (solo aplica para 'poly').
        random_state (int): Semilla para la reproducibilidad.
        
    Retorna:
        Pipeline: El pipeline configurado listo para ser entrenado.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = SVC(
        kernel=kernel_type,
        degree=degree,
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


def train_target(df: pd.DataFrame, target: str, args: argparse.Namespace) -> None:
    """
    Entrena modelos SVM con diferentes configuraciones de kernel para un objetivo específico.
    
    Argumentos:
        df (pd.DataFrame): DataFrame con los datos de entrenamiento.
        target (str): Nombre de la columna objetivo a predecir.
        args (argparse.Namespace): Argumentos de configuración.
    """
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # Entrenar un modelo por cada configuración de kernel
    for config in KERNEL_CONFIGS:
        print(f"[{target}] Entrenando configuración: {config['name']}...")
        
        pipeline = build_pipeline(config["kernel"], config["degree"], args.random_state)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        
        metrics = {
            "target": target,
            "kernel_used": config["name"],
            "params": {"kernel": config["kernel"], "degree": config["degree"]},
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }

        # Organizar carpetas por target y luego por kernel
        report_dir = Path(__file__).resolve().parents[1] / "reports" / target / config["name"]
        report_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(pipeline, report_dir / "model.joblib")
        # Guardar métricas del test set como evaluation.json (fuente de verdad para comparativas)
        (report_dir / "evaluation.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[{target}] Evaluacion (test set) guardada para {config['name']}")


def evaluate_target(df: pd.DataFrame, target: str) -> None:
    """
    Evalúa todos los kernels disponibles que han sido entrenados para un objetivo específico.
    
    Argumentos:
        df (pd.DataFrame): DataFrame con los datos de evaluación.
        target (str): Nombre de la columna objetivo.
    """
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    
    for config in KERNEL_CONFIGS:
        kernel_name = config["name"]
        model_path = Path(__file__).resolve().parents[1] / "reports" / target / kernel_name / "model.joblib"

        if not model_path.exists():
            print(f"  [!] Saltando {kernel_name}: No se encontró el modelo.")
            continue

        model = joblib.load(model_path)
        predictions = model.predict(X)

        # Guardar predicciones
        report_dir = model_path.parent
        pred_df = df.copy()
        pred_df[f"pred_{target}"] = predictions
        pred_df.to_csv(report_dir / "predictions.csv", index=False)
        print(f"[{target}] Predicciones guardadas para {kernel_name}")


def main() -> None:
    """Función principal que orquesta el flujo de entrenamiento y/o evaluación para SVM."""
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(f"No existe el CSV: {data_path}")

    df = pd.read_csv(data_path)

    if args.task in {"train", "both"}:
        for target in TARGETS:
            if target in df.columns:
                train_target(df, target, args)

    if args.task in {"evaluate", "both"}:
        eval_df = pd.read_csv(args.eval_data) if args.eval_data else df
        for target in TARGETS:
            evaluate_target(eval_df, target)


if __name__ == "__main__":
    main()