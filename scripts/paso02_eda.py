"""
paso02_eda.py — Análisis Exploratorio de Datos.

Genera visualizaciones del dataset crudo: distribuciones, correlaciones,
targets y valores faltantes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import DATASET_RAW, PALETTE, save_figure, save_json

PASO = "paso02_eda"


def run(resultados=None):
    """Ejecuta el EDA completo sobre el dataset crudo."""

    df = pd.read_csv(DATASET_RAW)
    print(f"  Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # ── 1. Overview ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Paso 02 — Overview del Dataset", fontsize=16,
                 fontweight="bold", y=0.98)

    info = [
        (f"{len(df):,}", "Registros", PALETTE[0]),
        (f"{len(df.columns)}", "Características", PALETTE[1]),
        (f"{len(numeric_cols)}", "Numéricas", PALETTE[2]),
        (f"{len(categorical_cols)}", "Categóricas", PALETTE[3]),
    ]
    for ax, (val, label, color) in zip(axes.flatten(), info):
        ax.text(0.5, 0.5, val, ha="center", va="center",
                fontsize=48, fontweight="bold", color=color)
        ax.text(0.5, 0.15, label, ha="center", va="center", fontsize=12)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    plt.tight_layout()
    save_figure(fig, PASO, "overview")

    # ── 2. Distribuciones numéricas ──────────────────────────────────
    n = len(numeric_cols)
    cols_grid = 3
    rows_grid = (n + cols_grid - 1) // cols_grid

    fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(16, 4 * rows_grid))
    axes = axes.flatten()
    fig.suptitle("Distribuciones — Variables Numéricas", fontsize=14,
                 fontweight="bold")

    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        ax.hist(df[col].dropna(), bins=30, color=PALETTE[idx % len(PALETTE)],
                alpha=0.75, edgecolor="white", linewidth=1.2)
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3, linestyle="--")

    for idx in range(n, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    save_figure(fig, PASO, "distribuciones_numericas")

    # ── 3. Distribuciones categóricas ────────────────────────────────
    if categorical_cols:
        n = len(categorical_cols)
        cols_grid = 2
        rows_grid = (n + cols_grid - 1) // cols_grid

        fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(14, 5 * rows_grid))
        axes = axes.flatten()
        fig.suptitle("Distribuciones — Variables Categóricas", fontsize=14,
                     fontweight="bold")

        for idx, col in enumerate(categorical_cols):
            ax = axes[idx]
            counts = df[col].value_counts()
            ax.barh(range(len(counts)), counts.values,
                    color=PALETTE[idx % len(PALETTE)], alpha=0.8,
                    edgecolor="white", linewidth=1)
            ax.set_yticks(range(len(counts)))
            ax.set_yticklabels(counts.index, fontsize=9)
            ax.set_title(col, fontsize=10, fontweight="bold")
            ax.grid(axis="x", alpha=0.3, linestyle="--")

        for idx in range(n, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        save_figure(fig, PASO, "distribuciones_categoricas")

    # ── 4. Matriz de correlación ─────────────────────────────────────
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 0:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                    mask=mask, ax=ax, annot_kws={"fontsize": 8})
        ax.set_title("Matriz de Correlación", fontsize=13, fontweight="bold",
                     pad=20)
        plt.tight_layout()
        save_figure(fig, PASO, "correlacion")

    # ── 5. Distribución de targets ───────────────────────────────────
    target_cols = [col for col in df.columns if "rec" in col.lower()]

    if target_cols:
        n = len(target_cols)
        cols_grid = 2
        rows_grid = (n + cols_grid - 1) // cols_grid

        fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(14, 5 * rows_grid))
        axes = axes.flatten()
        fig.suptitle("Distribuciones de Targets", fontsize=14, fontweight="bold")

        for idx, col in enumerate(target_cols):
            ax = axes[idx]
            counts = df[col].value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
            bars = ax.bar(range(len(counts)), counts.values, color=colors,
                          alpha=0.8, edgecolor="white", linewidth=1.2)
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Cantidad", fontsize=10, fontweight="bold")
            ax.set_title(col, fontsize=10, fontweight="bold")
            ax.grid(axis="y", alpha=0.3, linestyle="--")

            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., h,
                        f"{int(h)}", ha="center", va="bottom", fontsize=8)

        for idx in range(n, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        save_figure(fig, PASO, "targets")

    # ── 6. Valores faltantes ─────────────────────────────────────────
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    total_missing = int(missing.sum())

    # ── JSON: estadísticas ───────────────────────────────────────────
    stats = {
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "numericas": numeric_cols,
        "categoricas": categorical_cols,
        "targets": target_cols,
        "valores_faltantes_total": total_missing,
        "estadisticas_descriptivas": df.describe().to_dict(),
    }
    save_json(stats, PASO, "estadisticas")

    if total_missing > 0:
        missing_df = pd.DataFrame({
            "Faltantes": missing, "Porcentaje": missing_pct
        }).sort_values("Faltantes", ascending=False)
        missing_df = missing_df[missing_df["Faltantes"] > 0]

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(missing_df)))
        ax.barh(missing_df.index, missing_df["Porcentaje"], color=colors,
                edgecolor="white", linewidth=1.5)
        ax.set_xlabel("Porcentaje (%)", fontsize=11, fontweight="bold")
        ax.set_title("Valores Faltantes por Columna", fontsize=13,
                     fontweight="bold", pad=20)
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        plt.tight_layout()
        save_figure(fig, PASO, "valores_faltantes")
    else:
        print("  No hay valores faltantes")

    return {"df_raw": df}
