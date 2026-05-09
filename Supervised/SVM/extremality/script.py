"""
Script de entrenamiento y evaluación utilizando SVM con Extremality MKL.

Usa kernels polinomiales débiles combinados mediante pesos de extremalidad
para entrenar clasificadores SVM con kernel precomputado (estrategia OvR)
para los cuatro objetivos del sistema de recomendación.

Uso:
    python script.py --data <ruta_al_csv>

El CSV debe ser el dataset ya preprocesado (dataset_processed.csv).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Rutas: agrega extremality_mkl al path de Python
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # .../Supervised/SVM/extremality
SVM_DIR    = SCRIPT_DIR.parent                        # .../Supervised/SVM
ROOT       = SVM_DIR.parents[1]                       # raíz del proyecto
EXTREMALITY_MKL_PATH = ROOT / "extremality_mkl"

sys.path.insert(0, str(EXTREMALITY_MKL_PATH))

from src.weak_polinomial_kernel import create_weak_kernels           # noqa: E402
from extremalitymkl.extremality_weights import kernel_extremaly_weights  # noqa: E402

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
TARGETS = [
    "genero_libro_rec",
    "genero_musical_rec",
    "genero_serie_rec",
    "tipo_vino_rec",
]

FEATURE_COLS = [
    "edad",
    "engagement_promedio",
    "valence_musical_pref",
    "energia_musical_pref",
    "hora_lectura_preferida_manana",
    "hora_lectura_preferida_noche",
    "hora_lectura_preferida_tarde",
    "velocidad_lectura_alta",
    "velocidad_lectura_baja",
    "velocidad_lectura_media",
    "contenido_visual_pref_anime",
    "contenido_visual_pref_documentales",
    "contenido_visual_pref_peliculas",
    "contenido_visual_pref_series cortas",
    "contenido_visual_pref_series largas",
]

# Hiperparámetros del MKL
NUM_KERNELS = 15   # número de kernels débiles
T_FEATURES  = 5   # máximo de features por kernel
MAX_DEGREE  = 3   # grado máximo polinomial
N_WEIGHT    = 2   # exponente para la combinación de pesos
C_SVM       = 1.0 # parámetro de regularización SVM


# ---------------------------------------------------------------------------
# Utilidades de kernel
# ---------------------------------------------------------------------------

def build_combined_kernel(KL: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Construye la matriz gram combinada a partir de los pesos de extremalidad.

    Args:
        KL:      Arreglo de kernels, shape (num_kernels, n1, n2).
        weights: Pesos, shape (num_kernels,).

    Returns:
        Gram matrix combinada, shape (n1, n2).
    """
    return np.einsum("ijk,i->jk", KL, weights)


# ---------------------------------------------------------------------------
# Entrenamiento y predicción OvR
# ---------------------------------------------------------------------------

def train_extremality_ovr(
    KL_train: np.ndarray,
    y_train: np.ndarray,
    n: int = N_WEIGHT,
) -> tuple[list[SVC], np.ndarray, list[np.ndarray]]:
    """
    Entrena clasificadores OvR usando SVM con kernel precomputado vía Extremality MKL.

    Para cada clase:
      1. Convierte etiquetas a binario (+1 / -1).
      2. Calcula pesos de extremalidad sobre los kernels débiles.
      3. Construye la gram matrix combinada.
      4. Entrena un SVC con kernel='precomputed'.

    Returns:
        (classifiers, classes, weights_per_class)
    """
    classes = np.unique(y_train)
    classifiers: list[SVC] = []
    weights_list: list[np.ndarray] = []

    for cls in classes:
        y_bin = np.where(y_train == cls, 1, -1).astype(int)

        # Pesos de extremalidad (orden natural: mejores kernels primero)
        result = kernel_extremaly_weights(KL_train, y_bin, n=n)
        w = result.w_1

        gram_train = build_combined_kernel(KL_train, w)

        svm = SVC(
            kernel="precomputed",
            C=C_SVM,
            class_weight="balanced",
        )
        svm.fit(gram_train, y_bin)

        classifiers.append(svm)
        weights_list.append(w)

    return classifiers, classes, weights_list


def predict_ovr(
    classifiers: list[SVC],
    classes: np.ndarray,
    weights_list: list[np.ndarray],
    KL_test: np.ndarray,
) -> np.ndarray:
    """
    Predicción OvR: elige la clase con mayor puntuación de decisión.

    Args:
        KL_test: shape (num_kernels, n_test, n_train)
    """
    n_test = KL_test.shape[1]
    scores = np.zeros((n_test, len(classes)))

    for i, (clf, w) in enumerate(zip(classifiers, weights_list)):
        gram_test = build_combined_kernel(KL_test, w)  # (n_test, n_train)
        scores[:, i] = clf.decision_function(gram_test)

    return classes[np.argmax(scores, axis=1)]


# ---------------------------------------------------------------------------
# Pipeline por target
# ---------------------------------------------------------------------------

def train_target(df: pd.DataFrame, target: str, args: argparse.Namespace) -> None:
    """
    Ejecuta el pipeline completo de Extremality MKL para un objetivo dado.

    Args:
        df:     DataFrame con el dataset ya procesado.
        target: Nombre de la columna objetivo.
        args:   Argumentos de línea de comandos.
    """
    # Verificar que todas las feature_cols existen
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  [!] Columnas faltantes para '{target}': {missing}. Saltando...")
        return

    X = df[FEATURE_COLS].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # Normalizar a [0, 1] (requerido por las métricas de kernel)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"  [{target}] Generando {NUM_KERNELS} kernels polinomiales débiles...")
    np.random.seed(args.random_state)
    KL_train, KL_test = create_weak_kernels(
        X_train, X_test,
        t=T_FEATURES,
        num_kernels=NUM_KERNELS,
        max_degree=MAX_DEGREE,
    )

    print(f"  [{target}] Entrenando SVM Extremality MKL (OvR)...")
    classifiers, classes, weights_list = train_extremality_ovr(KL_train, y_train)

    y_pred = predict_ovr(classifiers, classes, weights_list, KL_test)

    acc      = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))

    metrics = {
        "target": target,
        "method": "extremality_mkl",
        "params": {
            "num_kernels": NUM_KERNELS,
            "t_features": T_FEATURES,
            "max_degree": MAX_DEGREE,
            "n_weight": N_WEIGHT,
            "C": C_SVM,
            "strategy": "OvR",
        },
        "accuracy": acc,
        "f1_macro": f1_macro,
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
    }

    # Guardar en reports/<target>/extremality_mkl/
    report_dir = SVM_DIR / "reports" / target / "extremality_mkl"
    report_dir.mkdir(parents=True, exist_ok=True)

    (report_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (report_dir / "evaluation.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    model_data = {
        "classifiers": classifiers,
        "classes": classes,
        "weights_list": weights_list,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "target": target,
    }
    joblib.dump(model_data, report_dir / "model.joblib")

    print(f"  [{target}] Accuracy: {acc:.4f} | F1-macro: {f1_macro:.4f}")
    print(f"  [{target}] Reportes guardados en: {report_dir}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SVM con Extremality MKL usando kernels polinomiales débiles."
    )
    parser.add_argument(
        "--data", required=True,
        help="Ruta al CSV preprocesado (dataset_processed.csv).",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Proporción del conjunto de test (default: 0.2).",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Semilla aleatoria (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    """Orquesta el entrenamiento Extremality MKL para todos los targets."""
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el CSV: {data_path}")

    print(f"\nCargando datos desde: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset: {df.shape[0]} filas × {df.shape[1]} columnas\n")

    for target in TARGETS:
        if target not in df.columns:
            print(f"[!] Columna '{target}' no encontrada. Saltando...\n")
            continue
        print(f">>> Procesando target: {target}")
        train_target(df, target, args)

    print("[OK] Proceso completado.")


if __name__ == "__main__":
    main()
