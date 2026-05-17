"""
paso05_supervised.py — Entrenamiento de modelos supervisados.

Entrena 4 tipos de modelo (RF, DecTree, LogReg, SVM) × 4 targets.
Cada modelo incluye su propio pipeline de preprocesamiento.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import (
    DATASET_RAW, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMNS,
    RANDOM_STATE, TEST_SIZE, PALETTE, MAX_SAMPLE,
    save_figure, save_json,
)

warnings.filterwarnings("ignore")

PASO = "paso05_supervised"


# ═══════════════════════════════════════════════════════════════════
# CONSTRUCTORES DE PIPELINES
# ═══════════════════════════════════════════════════════════════════

def _pipeline_rf():
    pre = ColumnTransformer(transformers=[
        ("num", "passthrough", NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocessor", pre),
                     ("model", RandomForestClassifier(
                         n_estimators=300, random_state=RANDOM_STATE,
                         class_weight="balanced"))])


def _pipeline_dt():
    pre = ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), NUMERICAL_FEATURES),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocessor", pre),
                     ("model", DecisionTreeClassifier(
                         random_state=RANDOM_STATE, class_weight="balanced",
                         max_depth=12, min_samples_split=10, min_samples_leaf=3))])


def _pipeline_lr():
    pre = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERICAL_FEATURES),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocessor", pre),
                     ("model", LogisticRegression(
                         random_state=RANDOM_STATE, max_iter=2000,
                         class_weight="balanced"))])


def _pipeline_svm(kernel="rbf", degree=3):
    pre = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocessor", pre),
                     ("model", SVC(
                         kernel=kernel, degree=degree, C=1.0, gamma="scale",
                         class_weight="balanced", random_state=RANDOM_STATE))])


# Modelos a entrenar
MODELOS = {
    "RandomForest": _pipeline_rf,
    "DecisionTree": _pipeline_dt,
    "LogisticRegression": _pipeline_lr,
    "SVM_lineal": lambda: _pipeline_svm("linear"),
    "SVM_poly_d2": lambda: _pipeline_svm("poly", 2),
    "SVM_poly_d3": lambda: _pipeline_svm("poly", 3),
    "SVM_radial": lambda: _pipeline_svm("rbf"),
}


def _entrenar_modelo(df, target, nombre_modelo, pipeline_fn):
    """Entrena un modelo para un target y retorna métricas."""
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    pipeline = pipeline_fn()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return {
        "modelo": nombre_modelo,
        "target": target,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "classification_report": classification_report(y_test, y_pred,
                                                        output_dict=True),
    }


def run(resultados=None):
    """Entrena todos los modelos × todos los targets."""

    df = pd.read_csv(DATASET_RAW)
    print(f"  Dataset: {df.shape[0]:,} filas × {df.shape[1]} columnas")

    # Sampling para SVM si el dataset es grande
    if MAX_SAMPLE and len(df) > MAX_SAMPLE:
        df_svm = df.sample(n=MAX_SAMPLE, random_state=RANDOM_STATE)
        print(f"  SVM usará muestra de {MAX_SAMPLE:,} filas")
    else:
        df_svm = df

    todos = []
    for nombre, pipeline_fn in MODELOS.items():
        is_svm = nombre.startswith("SVM")
        df_uso = df_svm if is_svm else df

        for target in TARGET_COLUMNS:
            print(f"    [{nombre}] → {target}...", end=" ")
            try:
                result = _entrenar_modelo(df_uso, target, nombre, pipeline_fn)
                print(f"acc={result['accuracy']:.3f}")
                todos.append(result)
            except Exception as e:
                print(f"ERROR: {e}")

    # ── Gráfica: F1 macro por modelo ─────────────────────────────────
    df_res = pd.DataFrame([{
        "Modelo": r["modelo"], "Target": r["target"],
        "Accuracy": r["accuracy"], "F1_Macro": r["f1_macro"],
    } for r in todos])

    modelos_unicos = df_res["Modelo"].unique()
    n_modelos = len(modelos_unicos)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Paso 05 — Modelos Supervisados", fontsize=14,
                 fontweight="bold")

    for ax, metric in zip(axes, ["Accuracy", "F1_Macro"]):
        pivot = df_res.pivot(index="Modelo", columns="Target", values=metric)
        pivot.plot(kind="bar", ax=ax, color=PALETTE[:4], alpha=0.85,
                   edgecolor="white", linewidth=1.5, width=0.7)
        ax.set_title(f"{metric} por Modelo", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(metric, fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.legend(fontsize=7, loc="lower right")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.axhline(y=0.7, color="white", linestyle="--", alpha=0.4)

    plt.tight_layout()
    save_figure(fig, PASO, "resultados_modelos")

    # ── JSON ─────────────────────────────────────────────────────────
    save_json(todos, PASO, "resultados")

    return {"resultados_modelos": todos, "df_resultados": df_res}
