from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

TARGET_COLUMN = "genero_libro_rec"
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
    parser = argparse.ArgumentParser(description="Evalúa SVM para genero_libro_rec")
    parser.add_argument("--data", required=True, help="Ruta al CSV para evaluación/predicción")
    parser.add_argument("--model", default=None, help="Ruta al modelo .joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"No existe el CSV: {data_path}")

    report_dir = Path(__file__).resolve().parents[2] / "reports" / TARGET_COLUMN
    model_path = Path(args.model) if args.model else report_dir / "model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")

    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    missing_features = [
        c for c in (NUMERIC_FEATURES + CATEGORICAL_FEATURES) if c not in df.columns
    ]
    if missing_features:
        raise ValueError(f"Faltan columnas de entrada: {missing_features}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    predictions = model.predict(X)

    pred_df = df.copy()
    pred_df[f"pred_{TARGET_COLUMN}"] = predictions
    predictions_path = report_dir / "predictions.csv"
    report_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(predictions_path, index=False)

    print(f"Predicciones guardadas en: {predictions_path}")

    if TARGET_COLUMN in df.columns:
        y_true = df[TARGET_COLUMN]
        metrics = {
            "target": TARGET_COLUMN,
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
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Evaluación guardada en: {metrics_path}")


if __name__ == "__main__":
    main()
