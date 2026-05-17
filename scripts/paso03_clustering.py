"""
paso03_clustering.py — Clustering no supervisado (KMeans).

Ejecuta 4 KMeans independientes (uno por dominio) sobre el dataset procesado,
calcula silhouette scores y genera el método del codo.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import (
    DATASET_PROCESSED, FEATURES_PROCESSED, COLUMNAS_DOMINIO, TARGETS_DOMINIO,
    N_CLUSTERS, RANDOM_STATE, N_INIT, PALETTE,
    save_figure, save_json,
)

PASO = "paso03_clustering"


def _clustering_dominio(df, columnas, nombre, n_clusters=N_CLUSTERS):
    """Aplica KMeans a un dominio y retorna etiquetas, modelo y silhouette."""
    cols = [c for c in columnas if c in df.columns]
    X = df[cols].values

    modelo = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=N_INIT)
    etiquetas = modelo.fit_predict(X)

    sil = (silhouette_score(X, etiquetas, sample_size=min(5000, len(etiquetas)))
           if len(np.unique(etiquetas)) > 1 else -1.0)

    print(f"    [{nombre}] Silhouette: {sil:.4f}  |  Cols: {len(cols)}")
    return pd.Series(etiquetas, name=f"kmeans_{nombre}"), modelo, sil


def _metodo_del_codo(df, dominio, max_k=10):
    """Calcula inercia y silhouette para K=2..max_k."""
    cols = [c for c in COLUMNAS_DOMINIO[dominio] if c in df.columns]
    X = df[cols].values
    rango = range(2, max_k + 1)
    inercias, silhouettes = [], []

    for k in rango:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = km.fit_predict(X)
        inercias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels,
                                            sample_size=min(5000, len(labels))))

    return list(rango), inercias, silhouettes


def run(resultados=None):
    """Ejecuta 4 KMeans independientes y genera reports."""

    df = pd.read_csv(DATASET_PROCESSED)
    print(f"  Dataset procesado: {df.shape[0]:,} filas × {df.shape[1]} columnas")

    # Separar features
    feature_cols = [c for c in FEATURES_PROCESSED if c in df.columns]
    X = df[feature_cols].copy()

    # ── Clustering por dominio ───────────────────────────────────────
    dominios = ["libro", "vino", "musica", "serie"]
    modelos, metricas, kmeans_labels = {}, {}, {}

    print("  Ejecutando 4 KMeans independientes:")
    for dominio in dominios:
        etiquetas, modelo, sil = _clustering_dominio(df, COLUMNAS_DOMINIO[dominio],
                                                      dominio)
        kmeans_labels[dominio] = etiquetas
        modelos[dominio] = modelo
        metricas[dominio] = sil

    # Clusters globales (usando todas las features)
    km_global = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=N_INIT)
    clusters_global = km_global.fit_predict(X)
    sil_global = silhouette_score(X, clusters_global,
                                   sample_size=min(5000, len(clusters_global)))
    print(f"    [global] Silhouette: {sil_global:.4f}")

    # ── Gráfica: Distribución de clusters ────────────────────────────
    unique, counts = np.unique(clusters_global, return_counts=True)
    colors = plt.cm.tab10(unique / max(unique.max(), 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Paso 03 — Distribución de Clusters (Global)",
                 fontsize=14, fontweight="bold")

    bars = axes[0].bar(unique, counts, color=colors, alpha=0.8,
                       edgecolor="white", linewidth=2)
    axes[0].set_xlabel("Cluster", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Cantidad", fontsize=11, fontweight="bold")
    axes[0].set_title("Conteo por Cluster", fontsize=12, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")
    for bar, c in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                     f"{int(c)}", ha="center", va="bottom", fontsize=10,
                     fontweight="bold")

    wedges, _, autotexts = axes[1].pie(
        counts, labels=[f"Cluster {i}" for i in unique],
        autopct="%1.1f%%", colors=colors, startangle=90,
        explode=[0.05] * len(unique))
    axes[1].set_title("Proporción", fontsize=12, fontweight="bold")
    for at in autotexts:
        at.set_color("white"); at.set_fontweight("bold")

    plt.tight_layout()
    save_figure(fig, PASO, "distribucion_clusters")

    # ── Gráfica: Método del codo (dominio global) ────────────────────
    ks, inercias, sils = _metodo_del_codo(df.assign(**{c: X[c] for c in feature_cols}),
                                           list(COLUMNAS_DOMINIO.keys())[0])
    # Recalculate for global features
    rango = range(2, 10)
    inercias_g, sils_g = [], []
    for k in rango:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = km.fit_predict(X)
        inercias_g.append(km.inertia_)
        sils_g.append(silhouette_score(X, labels, sample_size=min(5000, len(labels))))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Paso 03 — Selección de K (Global)", fontsize=13, fontweight="bold")

    axes[0].plot(list(rango), inercias_g, marker="o", color=PALETTE[0],
                 linewidth=2, markersize=8)
    axes[0].axvline(x=N_CLUSTERS, color=PALETTE[1], linestyle="--", alpha=0.8,
                    linewidth=1.5, label=f"K={N_CLUSTERS}")
    axes[0].set_title("Elbow (Inercia)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Inercia")
    axes[0].legend(); axes[0].grid(alpha=0.3, linestyle="--")

    axes[1].plot(list(rango), sils_g, marker="o", color=PALETTE[2],
                 linewidth=2, markersize=8)
    axes[1].axvline(x=N_CLUSTERS, color=PALETTE[1], linestyle="--", alpha=0.8,
                    linewidth=1.5, label=f"K={N_CLUSTERS}")
    axes[1].set_title("Silhouette Score", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette")
    axes[1].legend(); axes[1].grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    save_figure(fig, PASO, "elbow_silhouette")

    # ── JSON: métricas ───────────────────────────────────────────────
    metricas_json = {
        "n_clusters": N_CLUSTERS,
        "silhouette_global": float(sil_global),
        "inercia_global": float(km_global.inertia_),
        "silhouette_por_dominio": {k: float(v) for k, v in metricas.items()},
        "distribucion_clusters": {
            f"cluster_{i}": int(c) for i, c in zip(unique, counts)
        },
    }
    save_json(metricas_json, PASO, "metricas")

    return {
        "X": X,
        "clusters": clusters_global,
        "kmeans": km_global,
        "metricas": metricas,
        "modelos": modelos,
        "df_processed": df,
        "feature_cols": feature_cols,
    }
