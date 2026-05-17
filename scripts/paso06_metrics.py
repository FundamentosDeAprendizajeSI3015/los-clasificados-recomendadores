"""
paso06_metrics.py — Consolidación de métricas de los modelos supervisados.

Genera heatmaps de accuracy, precision y recall a partir de los resultados
del paso 05.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import PALETTE, save_figure, save_json

PASO = "paso06_metrics"


def run(resultados=None):
    """Consolida métricas y genera heatmaps comparativos."""

    paso05 = resultados.get("paso05") if resultados else None
    if paso05 is None:
        print("  [AVISO] Se requieren resultados de paso05_supervised. Saltando.")
        return None

    todos = paso05["resultados_modelos"]

    # Construir DataFrame consolidado
    rows = []
    for r in todos:
        report = r.get("classification_report", {})
        macro = report.get("macro avg", {})
        weighted = report.get("weighted avg", {})
        rows.append({
            "Modelo": r["modelo"],
            "Tarea": r["target"],
            "Accuracy": r["accuracy"],
            "F1_Macro": r["f1_macro"],
            "Precision_Macro": macro.get("precision"),
            "Recall_Macro": macro.get("recall"),
            "Precision_Weighted": weighted.get("precision"),
            "Recall_Weighted": weighted.get("recall"),
        })

    df = pd.DataFrame(rows)
    print(f"  {len(df)} resultados consolidados ({df['Modelo'].nunique()} modelos × "
          f"{df['Tarea'].nunique()} tareas)")

    # ── 1. Heatmap de Accuracy ───────────────────────────────────────
    pivot_acc = df.pivot(index="Modelo", columns="Tarea", values="Accuracy")

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(pivot_acc, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=1, linecolor="#2a3f5f", ax=ax,
                cbar_kws={"label": "Accuracy"},
                annot_kws={"fontsize": 10, "fontweight": "bold"})
    ax.set_title("Paso 06 — Heatmap de Accuracy (Todos los Modelos)",
                 fontsize=13, fontweight="bold", pad=20)
    ax.set_xlabel("Tarea"); ax.set_ylabel("Modelo")
    ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    plt.tight_layout()
    save_figure(fig, PASO, "heatmap_accuracy")

    # ── 2. Precision y Recall ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle("Paso 06 — Precision y Recall Macro", fontsize=13,
                 fontweight="bold")

    pivot_prec = df.pivot(index="Modelo", columns="Tarea",
                          values="Precision_Macro")
    sns.heatmap(pivot_prec, annot=True, fmt=".3f", cmap="Blues",
                linewidths=1, linecolor="#2a3f5f", ax=axes[0],
                cbar_kws={"label": "Precision"},
                annot_kws={"fontsize": 9, "fontweight": "bold"})
    axes[0].set_title("Precision Macro", fontsize=11, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=20, labelsize=8)

    pivot_rec = df.pivot(index="Modelo", columns="Tarea",
                         values="Recall_Macro")
    sns.heatmap(pivot_rec, annot=True, fmt=".3f", cmap="Greens",
                linewidths=1, linecolor="#2a3f5f", ax=axes[1],
                cbar_kws={"label": "Recall"},
                annot_kws={"fontsize": 9, "fontweight": "bold"})
    axes[1].set_title("Recall Macro", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", rotation=20, labelsize=8)

    plt.tight_layout()
    save_figure(fig, PASO, "precision_recall")

    # ── 3. Resumen por modelo (promedio) ─────────────────────────────
    resumen_modelo = df.groupby("Modelo")[["Accuracy", "F1_Macro"]].mean()
    resumen_modelo = resumen_modelo.sort_values("Accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(resumen_modelo))
    w = 0.35
    ax.bar(x - w/2, resumen_modelo["Accuracy"], w, label="Accuracy",
           color=PALETTE[0], alpha=0.85, edgecolor="white")
    ax.bar(x + w/2, resumen_modelo["F1_Macro"], w, label="F1 Macro",
           color=PALETTE[1], alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(resumen_modelo.index, rotation=35, ha="right", fontsize=9)
    ax.set_ylim([0, 1.1])
    ax.set_title("Paso 06 — Promedio por Modelo", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axhline(y=0.7, color="white", linestyle="--", alpha=0.4)
    plt.tight_layout()
    save_figure(fig, PASO, "promedio_modelos")

    # ── JSON ─────────────────────────────────────────────────────────
    consolidado = {
        "metricas_completas": rows,
        "resumen_por_modelo": resumen_modelo.to_dict(),
        "mejor_modelo": resumen_modelo.index[0],
        "mejor_accuracy_promedio": float(resumen_modelo["Accuracy"].iloc[0]),
    }
    save_json(consolidado, PASO, "consolidadas")

    return {"df_metricas": df, "resumen_modelo": resumen_modelo}
