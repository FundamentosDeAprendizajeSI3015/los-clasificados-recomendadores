"""
paso07_comparison.py — Comparación final de modelos y análisis cruzado.

Rankings por tarea, comparación de modelos base, distribución de
targets por cluster.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    DATASET_RAW, TARGET_COLUMNS, PALETTE, N_CLUSTERS,
    save_figure, save_json,
)

PASO = "paso07_comparison"


def run(resultados=None):
    """Genera comparaciones finales entre modelos."""

    paso06 = resultados.get("paso06") if resultados else None
    paso03 = resultados.get("paso03") if resultados else None

    if paso06 is None:
        print("  [AVISO] Se requieren resultados de paso06_metrics. Saltando.")
        return None

    df_metricas = paso06["df_metricas"]
    targets = df_metricas["Tarea"].unique().tolist()

    # ── 1. Ranking por tarea ─────────────────────────────────────────
    n_targets = len(targets)
    cols_grid = 2
    rows_grid = (n_targets + cols_grid - 1) // cols_grid

    fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(14, 5 * rows_grid))
    axes = axes.flatten()
    fig.suptitle("Paso 07 — Ranking de Modelos por Tarea (Accuracy)",
                 fontsize=14, fontweight="bold")

    ranking_data = {}
    for idx, target in enumerate(targets):
        ax = axes[idx]
        df_task = (df_metricas[df_metricas["Tarea"] == target]
                   .sort_values("Accuracy", ascending=True))

        max_acc = df_task["Accuracy"].max()
        bar_colors = [PALETTE[0] if acc == max_acc else PALETTE[2]
                      for acc in df_task["Accuracy"]]

        ax.barh(df_task["Modelo"], df_task["Accuracy"], color=bar_colors,
                alpha=0.85, edgecolor="white", linewidth=1)
        ax.set_xlim([max(0, df_task["Accuracy"].min() - 0.1),
                     min(1.0, max_acc + 0.05)])
        ax.set_xlabel("Accuracy", fontsize=9)
        ax.set_title(target, fontsize=10, fontweight="bold")
        ax.axvline(x=0.7, color="white", linestyle="--", alpha=0.4)
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.tick_params(axis="y", labelsize=8)

        for i, (_, row) in enumerate(df_task.iterrows()):
            ax.text(row["Accuracy"] + 0.003, i, f'{row["Accuracy"]:.3f}',
                    va="center", fontsize=8)

        # Mejor modelo para este target
        best = df_task.iloc[-1]
        ranking_data[target] = {
            "mejor_modelo": best["Modelo"],
            "accuracy": float(best["Accuracy"]),
            "f1_macro": float(best["F1_Macro"]),
        }

    for idx in range(n_targets, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    save_figure(fig, PASO, "ranking_por_tarea")

    # ── 2. Modelos base (sin SVM) ────────────────────────────────────
    modelos_base = ["RandomForest", "DecisionTree", "LogisticRegression"]
    df_base = df_metricas[df_metricas["Modelo"].isin(modelos_base)]

    if not df_base.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Paso 07 — Comparación Modelos Base", fontsize=14,
                     fontweight="bold")

        for ax, metric in zip(axes, ["Accuracy", "F1_Macro"]):
            pivot = df_base.pivot(index="Modelo", columns="Tarea",
                                  values=metric)
            pivot.plot(kind="bar", ax=ax, color=PALETTE[:4], alpha=0.85,
                       edgecolor="white", linewidth=1.5, width=0.7)
            ax.set_title(f"{metric} — Modelos Base", fontsize=11,
                         fontweight="bold")
            ax.set_xlabel(""); ax.set_ylabel(metric)
            ax.set_ylim([0, 1.1])
            ax.legend(fontsize=8, loc="lower right")
            ax.tick_params(axis="x", rotation=20)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.axhline(y=0.7, color="white", linestyle="--", alpha=0.4)

        plt.tight_layout()
        save_figure(fig, PASO, "modelos_base")

    # ── 3. Targets por cluster ───────────────────────────────────────
    if paso03 is not None:
        clusters = paso03["clusters"]
        df_raw = pd.read_csv(DATASET_RAW)

        # Asegurar misma longitud
        n = min(len(df_raw), len(clusters))
        df_cl = df_raw.iloc[:n].copy()
        df_cl["cluster"] = clusters[:n]

        target_cols = [c for c in TARGET_COLUMNS if c in df_cl.columns]

        if target_cols:
            n_t = len(target_cols)
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            fig.suptitle("Paso 07 — Distribución de Targets por Cluster",
                         fontsize=13, fontweight="bold")

            for idx, target in enumerate(target_cols[:4]):
                ax = axes[idx]
                ct = df_cl.groupby(["cluster", target]).size().unstack(
                    fill_value=0)
                ct_norm = ct.div(ct.sum(axis=1), axis=0)
                ct_norm.plot(kind="bar", ax=ax,
                             color=PALETTE[:len(ct_norm.columns)],
                             alpha=0.85, edgecolor="white", linewidth=1,
                             width=0.75)
                ax.set_title(target, fontsize=10, fontweight="bold")
                ax.set_xlabel("Cluster"); ax.set_ylabel("Proporción")
                ax.legend(fontsize=7, loc="upper right")
                ax.tick_params(axis="x", rotation=0)
                ax.grid(axis="y", alpha=0.3, linestyle="--")

            for idx in range(min(n_t, 4), len(axes)):
                fig.delaxes(axes[idx])

            plt.tight_layout()
            save_figure(fig, PASO, "targets_por_cluster")

    # ── JSON ─────────────────────────────────────────────────────────
    ranking_json = {
        "ranking_por_tarea": ranking_data,
        "resumen_global": {
            "mejor_accuracy_promedio": float(
                df_metricas.groupby("Modelo")["Accuracy"].mean().max()),
            "mejor_modelo_global": df_metricas.groupby("Modelo")[
                "Accuracy"].mean().idxmax(),
        },
    }
    save_json(ranking_json, PASO, "ranking")

    return None
