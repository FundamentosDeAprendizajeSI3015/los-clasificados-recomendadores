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

TARGET_COLUMN = "tipo_vino_rec"
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
    parser = argparse.ArgumentParser(description="Entrena un SVM para tipo_vino_rec")
    parser.add_argument("--data", required=True, help="Ruta al CSV de entrenamiento")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción para test")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla")
    return parser.parse_args()


def build_pipeline(random_state: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
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


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(f"No existe el archivo CSV: {data_path}")

    df = pd.read_csv(data_path)
    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {missing}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]

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
        "target": TARGET_COLUMN,
        "model": "SVC",
        "n_rows": int(len(df)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    report_dir = Path(__file__).resolve().parents[2] / "reports" / TARGET_COLUMN
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path = report_dir / "model.joblib"
    metrics_path = report_dir / "metrics.json"

    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Modelo guardado en: {model_path}")
    print(f"Métricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
