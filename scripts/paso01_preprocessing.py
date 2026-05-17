"""
paso01_preprocessing.py — Carga, validación, escalado y codificación.

PASO OPCIONAL: Solo necesario si no existe dataset_processed.csv en data/.
Si ya existe, Main.py salta este paso automáticamente.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from config import (
    DATASET_RAW, DATASET_PROCESSED, DATA_DIR,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMNS,
    save_figure, save_json,
)

PASO = "paso01_preprocessing"


def run(resultados=None):
    """Ejecuta el preprocesamiento completo y guarda dataset_processed.csv."""
    import matplotlib.pyplot as plt

    print(f"  Cargando {DATASET_RAW.name}...")
    df = pd.read_csv(DATASET_RAW)
    print(f"  Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")

    # ── Validación ──
    missing_cols = [c for c in NUMERICAL_FEATURES + CATEGORICAL_FEATURES + TARGET_COLUMNS
                    if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes: {missing_cols}")

    # ── Escalar numéricas ──
    scaler = StandardScaler()
    df_processed = pd.DataFrame(index=df.index)
    scaled = scaler.fit_transform(df[NUMERICAL_FEATURES])
    df_processed[NUMERICAL_FEATURES] = scaled

    # ── Codificar categóricas (OHE) ──
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(df[CATEGORICAL_FEATURES])
    ohe_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
    df_processed[ohe_names] = encoded

    # ── Codificar targets (LabelEncoder) ──
    target_mappings = {}
    for target in TARGET_COLUMNS:
        le = LabelEncoder()
        df_processed[target] = le.fit_transform(df[target])
        target_mappings[target] = dict(zip(le.classes_.tolist(),
                                           range(len(le.classes_))))

    # ── Guardar CSV procesado ──
    df_processed.to_csv(DATASET_PROCESSED, index=False)
    print(f"  Guardado: {DATASET_PROCESSED}")

    # ── Reports ──

    # Gráfica: resumen del esquema
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Paso 01 — Resumen del Preprocesamiento", fontsize=16,
                 fontweight="bold", y=0.98)

    info = [
        (f"{df.shape[0]:,}", "Registros", "#FF6B6B"),
        (f"{df.shape[1]}", "Columnas originales", "#4ECDC4"),
        (f"{df_processed.shape[1]}", "Columnas procesadas", "#45B7D1"),
        (f"{df.isnull().sum().sum()}", "Valores faltantes", "#FFA07A"),
    ]
    for ax, (valor, label, color) in zip(axes.flatten(), info):
        ax.text(0.5, 0.5, valor, ha="center", va="center",
                fontsize=48, fontweight="bold", color=color)
        ax.text(0.5, 0.15, label, ha="center", va="center", fontsize=12)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    plt.tight_layout()
    save_figure(fig, PASO, "resumen_esquema")

    # JSON: resumen
    resumen = {
        "filas_originales": int(df.shape[0]),
        "columnas_originales": int(df.shape[1]),
        "columnas_procesadas": int(df_processed.shape[1]),
        "valores_faltantes": int(df.isnull().sum().sum()),
        "target_mappings": target_mappings,
        "features_numericas": NUMERICAL_FEATURES,
        "features_categoricas_ohe": list(ohe_names),
    }
    save_json(resumen, PASO, "resumen")

    return {"df_processed": df_processed}
