"""
paso04_dimensionality.py — Reducción de dimensionalidad (PCA + t-SNE).

Visualiza clusters en 2D, varianza explicada y perfil por cluster.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from config import (
    DATASET_RAW, N_CLUSTERS, RANDOM_STATE, PALETTE, MAX_SAMPLE,
    FEATURES_PROCESSED, save_figure, save_json,
)

PASO = "paso04_dimensionality"


def run(resultados=None):
    """Ejecuta PCA y t-SNE sobre los resultados del clustering."""

    paso03 = resultados.get("paso03") if resultados else None
    if paso03 is None:
        print("  [AVISO] Se requieren resultados de paso03_clustering. Saltando.")
        return None

    X = paso03["X"]
    clusters = paso03["clusters"]
    kmeans = paso03["kmeans"]
    feature_cols = paso03["feature_cols"]

    # ── 1. PCA 2D ────────────────────────────────────────────────────
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Sampling para t-SNE si es necesario
    n_sample = min(MAX_SAMPLE, len(X)) if MAX_SAMPLE else len(X)
    rng = np.random.RandomState(RANDOM_STATE)
    idx_sample = rng.choice(len(X), n_sample, replace=False) if n_sample < len(X) else np.arange(len(X))

    print(f"  t-SNE sobre {n_sample:,} muestras...")
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30,
                max_iter=1000)
    X_tsne = tsne.fit_transform(X.values[idx_sample])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Paso 04 — Clusters en PCA y t-SNE", fontsize=14,
                 fontweight="bold")

    sc1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="tab10",
                          s=8, alpha=0.5)
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    axes[0].scatter(centroids_pca[:, 0], centroids_pca[:, 1], c="red",
                    marker="X", s=200, edgecolors="black", linewidth=2,
                    label="Centroides")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                       fontsize=11, fontweight="bold")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
                       fontsize=11, fontweight="bold")
    axes[0].set_title("PCA 2D", fontsize=12, fontweight="bold")
    axes[0].legend(); axes[0].grid(alpha=0.3, linestyle="--")
    plt.colorbar(sc1, ax=axes[0], label="Cluster")

    sc2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters[idx_sample],
                          cmap="tab10", s=8, alpha=0.5)
    axes[1].set_xlabel("t-SNE 1", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("t-SNE 2", fontsize=11, fontweight="bold")
    axes[1].set_title(f"t-SNE 2D (n={n_sample:,})", fontsize=12,
                      fontweight="bold")
    axes[1].grid(alpha=0.3, linestyle="--")
    plt.colorbar(sc2, ax=axes[1], label="Cluster")

    plt.tight_layout()
    save_figure(fig, PASO, "pca_tsne")

    # ── 2. Varianza explicada acumulada ──────────────────────────────
    pca_full = PCA()
    pca_full.fit(X)
    var_acum = np.cumsum(pca_full.explained_variance_ratio_)

    n_90 = int(np.argmax(var_acum >= 0.90)) + 1
    n_95 = int(np.argmax(var_acum >= 0.95)) + 1

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(var_acum) + 1), var_acum, marker="o", color=PALETTE[0],
            linewidth=2, markersize=6)
    ax.axhline(y=0.90, color=PALETTE[1], linestyle="--", linewidth=1.5,
               label=f"90% → {n_90} comp.")
    ax.axhline(y=0.95, color=PALETTE[3], linestyle="--", linewidth=1.5,
               label=f"95% → {n_95} comp.")
    ax.axvline(x=2, color="white", linestyle=":", linewidth=1.2, alpha=0.6,
               label=f"2D ({var_acum[1]*100:.1f}%)")
    ax.set_title("Paso 04 — Varianza Explicada Acumulada (PCA)", fontsize=13,
                 fontweight="bold")
    ax.set_xlabel("Componentes"); ax.set_ylabel("Varianza Acumulada")
    ax.set_ylim([0, 1.05]); ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    save_figure(fig, PASO, "varianza_pca")

    # ── 3. Perfil normalizado por cluster ────────────────────────────
    df_cl = X.copy()
    df_cl["cluster"] = clusters

    perfil = df_cl.groupby("cluster")[feature_cols].mean()
    perfil_norm = (perfil - perfil.min()) / (perfil.max() - perfil.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(max(14, len(feature_cols) * 0.95), 5))
    sns.heatmap(perfil_norm, annot=True, fmt=".2f", cmap="YlGn",
                linewidths=0.5, linecolor="#2a3f5f", ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Paso 04 — Perfil Normalizado por Cluster\n"
                 "(0=mínimo, 1=máximo)", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Feature"); ax.set_ylabel("Cluster")
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    plt.tight_layout()
    save_figure(fig, PASO, "perfil_clusters")

    # ── 4. Boxplots de features principales ──────────────────────────
    main_features = ["edad", "engagement_promedio", "valence_musical_pref",
                     "energia_musical_pref"]
    available = [f for f in main_features if f in X.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 5))
    if len(available) == 1:
        axes = [axes]
    fig.suptitle("Paso 04 — Features Principales por Cluster", fontsize=13,
                 fontweight="bold")

    for ax, feat in zip(axes, available):
        data = [X.iloc[clusters == i][feat].values for i in range(N_CLUSTERS)]
        bp = ax.boxplot(data, labels=[f"C{i}" for i in range(N_CLUSTERS)],
                        patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], plt.cm.tab10(range(N_CLUSTERS))):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_title(feat.replace("_", " ").title(), fontsize=10,
                     fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    save_figure(fig, PASO, "boxplots_features")

    # ── JSON ─────────────────────────────────────────────────────────
    metricas = {
        "varianza_2d": float(var_acum[1]),
        "componentes_90pct": n_90,
        "componentes_95pct": n_95,
        "varianza_por_componente": pca_full.explained_variance_ratio_.tolist(),
        "perfil_promedio_por_cluster": perfil.to_dict(),
    }
    save_json(metricas, PASO, "metricas")

    return None
