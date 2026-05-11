"""
Visualizaciones del proyecto Los Clasificados Recomendadores

Cubre todo el pipeline: análisis exploratorio del dataset, evaluación de los
modelos supervisados (Árbol de Decisión, Regresión Logística, Random Forest y
SVM con todos sus kernels), comparación global entre modelos y visualización
del clustering no supervisado con análisis de perfiles por cluster.

Todos los gráficos se guardan automáticamente en visuals_results/.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# todos los gráficos van a esta carpeta
OUTPUT_DIR = Path("visuals_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# paleta de colores que usamos en toda la suite
PALETTE = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#F7DC6F"]

sns.set_theme(style="whitegrid")

# tema oscuro para todos los gráficos de matplotlib
plt.rcParams.update({
    "figure.facecolor": "#0F1419",
    "axes.facecolor": "#1a1f2e",
    "axes.edgecolor": "#2a3f5f",
    "axes.labelcolor": "#E8E8E8",
    "xtick.color": "#E8E8E8",
    "ytick.color": "#E8E8E8",
    "text.color": "#E8E8E8",
    "grid.color": "#2a3f5f",
})

print("Librerías cargadas")
print(f"Resultados en: {OUTPUT_DIR.absolute()}\n")

# ============================================================================
# PASO 1: Cargar Dataset
# ============================================================================

print("=" * 70)
print("PASO 1: Cargar Dataset")
print("=" * 70)

# cargamos el dataset principal desde la carpeta load
df = pd.read_csv("../load/dataset.csv")
print(f"Dataset: {df.shape[0]:,} registros x {df.shape[1]} variables")
print(df.head())
print()

# ============================================================================
# PASO 2: Resumen General del Dataset
# ============================================================================

print("=" * 70)
print("PASO 2: Resumen General del Dataset")
print("=" * 70)

# cuadro con los 4 números clave del dataset para tener una visión general rápida
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Overview del Dataset", fontsize=16, fontweight="bold", y=0.98)

# total de registros
ax = axes[0, 0]
ax.text(0.5, 0.5, f"{len(df):,}", ha='center', va='center',
        fontsize=48, fontweight='bold', color=PALETTE[0])
ax.text(0.5, 0.1, "Registros", ha='center', va='center', fontsize=12)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

# total de columnas (features + targets)
ax = axes[0, 1]
ax.text(0.5, 0.5, f"{len(df.columns)}", ha='center', va='center',
        fontsize=48, fontweight='bold', color=PALETTE[1])
ax.text(0.5, 0.1, "Características", ha='center', va='center', fontsize=12)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

# columnas numéricas
ax = axes[1, 0]
numeric_count = len(df.select_dtypes(include=[np.number]).columns)
ax.text(0.5, 0.5, f"{numeric_count}", ha='center', va='center',
        fontsize=48, fontweight='bold', color=PALETTE[2])
ax.text(0.5, 0.1, "Numéricas", ha='center', va='center', fontsize=12)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

# columnas categóricas
ax = axes[1, 1]
categorical_count = len(df.select_dtypes(include=['object']).columns)
ax.text(0.5, 0.5, f"{categorical_count}", ha='center', va='center',
        fontsize=48, fontweight='bold', color=PALETTE[3])
ax.text(0.5, 0.1, "Categóricas", ha='center', va='center', fontsize=12)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

plt.tight_layout()
fig_path = OUTPUT_DIR / "01_overview_dataset.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Guardado: {fig_path.name}")
plt.close()

# ============================================================================
# Distribuciones - Variables Numéricas
# ============================================================================

print("\n" + "=" * 70)
print("Distribuciones - Variables Numéricas")
print("=" * 70)

# extraemos los nombres de las columnas numéricas para iterar sobre ellas
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
n = len(numeric_cols)
cols = 3
rows = (n + cols - 1) // cols  # calcula filas necesarias redondeando hacia arriba

fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
axes = axes.flatten()
fig.suptitle("Distribuciones - Variables Numéricas", fontsize=14, fontweight="bold")

for idx, col in enumerate(numeric_cols):
    ax = axes[idx]
    ax.hist(df[col].dropna(), bins=30, color=PALETTE[idx % 6],
            alpha=0.75, edgecolor="white", linewidth=1.2)
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3, linestyle='--')

# ocultamos los subplots que sobran si hay menos columnas que celdas
for idx in range(n, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
fig_path = OUTPUT_DIR / "02_distribuciones_numericas.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Guardado: {fig_path.name}")
plt.close()

# ============================================================================
# Distribuciones - Variables Categóricas
# ============================================================================

print("\n" + "=" * 70)
print("Distribuciones - Variables Categóricas")
print("=" * 70)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

if categorical_cols:
    n = len(categorical_cols)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = axes.flatten()
    fig.suptitle("Distribuciones - Variables Categóricas", fontsize=14, fontweight="bold")

    for idx, col in enumerate(categorical_cols):
        ax = axes[idx]
        counts = df[col].value_counts()
        # barras horizontales para que se lean mejor las etiquetas largas
        ax.barh(range(len(counts)), counts.values, color=PALETTE[idx % 6],
                alpha=0.8, edgecolor="white", linewidth=1)
        ax.set_yticks(range(len(counts)))
        ax.set_yticklabels(counts.index, fontsize=9)
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.grid(axis='x', alpha=0.3, linestyle='--')

    for idx in range(n, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "03_distribuciones_categoricas.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Guardado: {fig_path.name}")
    plt.close()
else:
    print("No hay variables categóricas")

# ============================================================================
# Matriz de Correlación
# ============================================================================

print("\n" + "=" * 70)
print("Matriz de Correlación")
print("=" * 70)

numeric_df = df.select_dtypes(include=[np.number])
if len(numeric_df.columns) > 0:
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    # máscara para mostrar solo el triángulo inferior y no duplicar info
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                mask=mask, ax=ax, annot_kws={"fontsize": 8})

    ax.set_title("Matriz de Correlación", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    fig_path = OUTPUT_DIR / "04_matriz_correlacion.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Guardado: {fig_path.name}")
    plt.close()
else:
    print("No hay variables numéricas para correlación")

# ============================================================================
# Distribuciones de Targets (los 4 recomendadores)
# ============================================================================

print("\n" + "=" * 70)
print("Distribuciones de Targets")
print("=" * 70)

# los targets son las columnas que tienen '_rec' en el nombre
target_cols = [col for col in df.columns if 'rec' in col.lower()]

if target_cols:
    n = len(target_cols)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = axes.flatten()
    fig.suptitle("Distribuciones de Targets", fontsize=14, fontweight="bold")

    for idx, col in enumerate(target_cols):
        ax = axes[idx]
        counts = df[col].value_counts()
        # cada clase con un color distinto para distinguirlas mejor
        colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
        ax.bar(range(len(counts)), counts.values, color=colors,
               alpha=0.8, edgecolor="white", linewidth=1.2)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel("Cantidad", fontsize=10, fontweight="bold")
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # anotar el conteo encima de cada barra
        for bar in ax.patches:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)

    for idx in range(n, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "05_distribucion_targets.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Guardado: {fig_path.name}")
    plt.close()
else:
    print("No hay columnas de targets encontradas")

# ============================================================================
# Valores Faltantes
# ============================================================================

print("\n" + "=" * 70)
print("Valores Faltantes")
print("=" * 70)

missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Faltantes': missing,
    'Porcentaje': missing_percent
}).sort_values('Faltantes', ascending=False)

# filtramos solo las columnas que sí tienen faltantes
missing_df = missing_df[missing_df['Faltantes'] > 0]

if len(missing_df) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(missing_df)))
    ax.barh(missing_df.index, missing_df['Porcentaje'], color=colors,
            edgecolor="white", linewidth=1.5)
    ax.set_xlabel("Porcentaje (%)", fontsize=11, fontweight="bold")
    ax.set_title("Valores Faltantes por Columna", fontsize=13, fontweight="bold", pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for i, (idx, row) in enumerate(missing_df.iterrows()):
        ax.text(row['Porcentaje'] + 0.5, i, f"{row['Porcentaje']:.1f}%",
                va='center', fontsize=9)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "06_valores_faltantes.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Guardado: {fig_path.name}")
    plt.close()
else:
    print("No hay valores faltantes en el dataset")

# ============================================================================
# MODELOS SUPERVISADOS - Carga de métricas
# ============================================================================

print("\n" + "=" * 70)
print("MODELOS SUPERVISADOS - Carga de Métricas")
print("=" * 70)

# los 4 recomendadores del proyecto
targets = ["genero_libro_rec", "genero_musical_rec", "genero_serie_rec", "tipo_vino_rec"]

# rutas a los reportes de cada modelo supervisado (leen evaluation.json)
models_info = {
    "Árbol de Decisión": "../Supervised/Árbol de Decisión/reports",
    "Regresión Logística": "../Supervised/Regresión Logística/reports",
    "RandomForest": "../Supervised/RandomForest/reports",
}

# los 5 kernels del SVM que evaluamos, incluyendo el extremality MKL
svm_kernels = ["kernel_lineal", "kernel_poly_d2", "kernel_poly_d3", "kernel_radial", "extremality_mkl"]
svm_base = "../Supervised/SVM/reports"


def load_eval(path):
    """Lee evaluation.json y devuelve (accuracy, f1_macro)."""
    with open(path) as f:
        data = json.load(f)
    return data.get("accuracy", 0), data.get("f1_macro", 0)


def load_class_metrics(path):
    """Extrae métricas por clase del classification_report de un evaluation.json."""
    with open(path) as f:
        data = json.load(f)
    report = data.get("classification_report", {})
    # excluimos las entradas de promedios, solo queremos las clases reales
    skip = {"accuracy", "macro avg", "weighted avg"}
    classes = [k for k in report.keys() if k not in skip]
    return {cls: report[cls] for cls in classes}


print("\nModelos disponibles:\n")
for model_name, base_path in models_info.items():
    available = sum(1 for t in targets if (Path(base_path) / t / "evaluation.json").exists())
    print(f"  {model_name}: {available}/{len(targets)} targets")

for kernel in svm_kernels:
    available = sum(1 for t in targets
                    if Path(f"{svm_base}/{t}/{kernel}/evaluation.json").exists())
    print(f"  SVM ({kernel}): {available}/{len(targets)} targets")

# ============================================================================
# Árbol de Decisión - F1 por Clase
# ============================================================================

print("\n" + "=" * 70)
print("Árbol de Decisión - F1 por Clase")
print("=" * 70)

base_path = "../Supervised/Árbol de Decisión/reports"

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
fig.suptitle("Árbol de Decisión — F1 por Clase", fontsize=14, fontweight="bold")

for idx, target in enumerate(targets):
    ax = axes[idx]
    eval_path = Path(base_path) / target / "evaluation.json"

    try:
        class_metrics = load_class_metrics(eval_path)
        classes = list(class_metrics.keys())
        f1_vals = [class_metrics[c]["f1-score"] for c in classes]
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(classes))]

        ax.bar(range(len(classes)), f1_vals, color=colors,
               alpha=0.85, edgecolor='white', linewidth=1.5)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=35, ha='right', fontsize=8)
        ax.set_ylim([0, 1.1])
        ax.set_ylabel("F1-Score", fontsize=9)
        ax.set_title(target, fontsize=10, fontweight="bold")
        # línea de referencia para ver de un vistazo qué clases superan 0.7
        ax.axhline(y=0.7, color='white', linestyle='--', alpha=0.4, linewidth=1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # valor encima de cada barra
        for bar, val in zip(ax.patches, f1_vals):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    except Exception:
        ax.text(0.5, 0.5, "Sin datos", ha='center', va='center', fontsize=10)
        ax.set_title(target, fontsize=10, fontweight="bold")

plt.tight_layout()
fig_path = OUTPUT_DIR / "07_arbol_f1_por_clase.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Guardado: {fig_path.name}")
plt.close()

# ============================================================================
# Regresión Logística - F1 por Clase
# ============================================================================

print("\n" + "=" * 70)
print("Regresión Logística - F1 por Clase")
print("=" * 70)

base_path = "../Supervised/Regresión Logística/reports"

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
fig.suptitle("Regresión Logística — F1 por Clase", fontsize=14, fontweight="bold")

for idx, target in enumerate(targets):
    ax = axes[idx]
    eval_path = Path(base_path) / target / "evaluation.json"

    try:
        class_metrics = load_class_metrics(eval_path)
        classes = list(class_metrics.keys())
        f1_vals = [class_metrics[c]["f1-score"] for c in classes]
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(classes))]

        ax.bar(range(len(classes)), f1_vals, color=colors,
               alpha=0.85, edgecolor='white', linewidth=1.5)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=35, ha='right', fontsize=8)
        ax.set_ylim([0, 1.1])
        ax.set_ylabel("F1-Score", fontsize=9)
        ax.set_title(target, fontsize=10, fontweight="bold")
        ax.axhline(y=0.7, color='white', linestyle='--', alpha=0.4, linewidth=1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        for bar, val in zip(ax.patches, f1_vals):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    except Exception:
        ax.text(0.5, 0.5, "Sin datos", ha='center', va='center', fontsize=10)
        ax.set_title(target, fontsize=10, fontweight="bold")

plt.tight_layout()
fig_path = OUTPUT_DIR / "08_logreg_f1_por_clase.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Guardado: {fig_path.name}")
plt.close()

# ============================================================================
# Random Forest - F1 por Clase
# ============================================================================

print("\n" + "=" * 70)
print("Random Forest - F1 por Clase")
print("=" * 70)

base_path = "../Supervised/RandomForest/reports"

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
fig.suptitle("Random Forest — F1 por Clase", fontsize=14, fontweight="bold")

for idx, target in enumerate(targets):
    ax = axes[idx]
    eval_path = Path(base_path) / target / "evaluation.json"

    try:
        class_metrics = load_class_metrics(eval_path)
        classes = list(class_metrics.keys())
        f1_vals = [class_metrics[c]["f1-score"] for c in classes]
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(classes))]

        ax.bar(range(len(classes)), f1_vals, color=colors,
               alpha=0.85, edgecolor='white', linewidth=1.5)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=35, ha='right', fontsize=8)
        ax.set_ylim([0, 1.1])
        ax.set_ylabel("F1-Score", fontsize=9)
        ax.set_title(target, fontsize=10, fontweight="bold")
        ax.axhline(y=0.7, color='white', linestyle='--', alpha=0.4, linewidth=1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        for bar, val in zip(ax.patches, f1_vals):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    except Exception:
        ax.text(0.5, 0.5, "Sin datos", ha='center', va='center', fontsize=10)
        ax.set_title(target, fontsize=10, fontweight="bold")

plt.tight_layout()
fig_path = OUTPUT_DIR / "09_rf_f1_por_clase.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Guardado: {fig_path.name}")
plt.close()

# ============================================================================
# SVM - Comparación de los 5 Kernels por Target
# ============================================================================

print("\n" + "=" * 70)
print("SVM - Comparación de Kernels")
print("=" * 70)

# etiquetas cortas para los 5 kernels en el eje x
kernel_labels = ["lineal", "poly_d2", "poly_d3", "radial", "ext_mkl"]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()
fig.suptitle("SVM — Accuracy y F1 por Kernel y Tarea", fontsize=14, fontweight="bold")

for idx, target in enumerate(targets):
    ax = axes[idx]
    accs = []
    f1s = []

    # leemos el evaluation.json de cada kernel para este target
    for kernel in svm_kernels:
        eval_path = Path(f"{svm_base}/{target}/{kernel}/evaluation.json")
        try:
            acc, f1 = load_eval(eval_path)
            accs.append(acc)
            f1s.append(f1)
        except Exception:
            # si no hay datos para ese kernel, ponemos 0 para no romper el plot
            accs.append(0)
            f1s.append(0)

    x = np.arange(len(kernel_labels))
    width = 0.3  # más delgado que antes porque ahora son 5 kernels

    # barras dobles: accuracy a la izquierda, F1 macro a la derecha
    ax.bar(x - width / 2, accs, width, label="Accuracy",
           color=PALETTE[0], alpha=0.85, edgecolor='white')
    ax.bar(x + width / 2, f1s, width, label="F1 Macro",
           color=PALETTE[1], alpha=0.85, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(kernel_labels, fontsize=8)
    ax.set_ylim([0, 1.1])
    ax.set_ylabel("Puntuación", fontsize=9)
    ax.set_title(target, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0.7, color='white', linestyle='--', alpha=0.4, linewidth=1)

plt.tight_layout()
fig_path = OUTPUT_DIR / "10_svm_kernels_comparacion.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Guardado: {fig_path.name}")
plt.close()

# ============================================================================
# Comparación de Modelos Base (Árbol, LogReg, RF)
# ============================================================================

print("\n" + "=" * 70)
print("Comparación de Modelos Base")
print("=" * 70)

# cargamos el CSV consolidado con todas las métricas de todos los modelos
df_metricas = pd.read_csv("../Metricas/metricas_consolidadas.csv")

# separamos los 3 modelos base del SVM para graficarlos de forma limpia
modelos_base = ["Árbol de Decisión", "Regresión Logística", "RandomForest"]
df_base = df_metricas[df_metricas["Modelo"].isin(modelos_base)]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Comparación de Modelos Base por Tarea", fontsize=14, fontweight="bold")

# subplot izquierdo: accuracy
pivot_acc = df_base.pivot(index="Modelo", columns="Tarea", values="Accuracy")
pivot_acc.plot(kind="bar", ax=axes[0], color=PALETTE[:4], alpha=0.85,
               edgecolor='white', linewidth=1.5, width=0.7)
axes[0].set_title("Accuracy por Modelo y Tarea", fontsize=11, fontweight="bold")
axes[0].set_xlabel("")
axes[0].set_ylabel("Accuracy", fontsize=10)
axes[0].set_ylim([0, 1.1])
axes[0].legend(fontsize=8, loc='lower right')
axes[0].tick_params(axis='x', rotation=20)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')
axes[0].axhline(y=0.7, color='white', linestyle='--', alpha=0.4, linewidth=1)

# subplot derecho: F1 macro
pivot_f1 = df_base.pivot(index="Modelo", columns="Tarea", values="F1_Macro")
pivot_f1.plot(kind="bar", ax=axes[1], color=PALETTE[:4], alpha=0.85,
              edgecolor='white', linewidth=1.5, width=0.7)
axes[1].set_title("F1 Macro por Modelo y Tarea", fontsize=11, fontweight="bold")
axes[1].set_xlabel("")
axes[1].set_ylabel("F1 Macro", fontsize=10)
axes[1].set_ylim([0, 1.1])
axes[1].legend(fontsize=8, loc='lower right')
axes[1].tick_params(axis='x', rotation=20)
axes[1].grid(axis='y', alpha=0.3, linestyle='--')
axes[1].axhline(y=0.7, color='white', linestyle='--', alpha=0.4, linewidth=1)

plt.tight_layout()
fig_path = OUTPUT_DIR / "11_comparacion_modelos_base.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Guardado: {fig_path.name}")
plt.close()

# ============================================================================
# Heatmap Global - Accuracy de Todos los Modelos vs Todas las Tareas
# ============================================================================

print("\n" + "=" * 70)
print("Heatmap Global de Modelos")
print("=" * 70)

# el heatmap permite ver de un vistazo qué modelo gana en cada tarea
pivot_heatmap = df_metricas.pivot(index="Modelo", columns="Tarea", values="Accuracy")

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(
    pivot_heatmap,
    annot=True, fmt=".3f",
    cmap="YlOrRd",
    linewidths=1,
    linecolor="#2a3f5f",
    ax=ax,
    cbar_kws={"label": "Accuracy"},
    annot_kws={"fontsize": 10, "fontweight": "bold"}
)
ax.set_title("Heatmap de Accuracy — Todos los Modelos x Todas las Tareas",
             fontsize=13, fontweight="bold", pad=20)
ax.set_xlabel("Tarea", fontsize=11)
ax.set_ylabel("Modelo", fontsize=11)
ax.tick_params(axis='x', rotation=20, labelsize=9)
ax.tick_params(axis='y', rotation=0, labelsize=9)

plt.tight_layout()
fig_path = OUTPUT_DIR / "12_heatmap_global_modelos.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Guardado: {fig_path.name}")
plt.close()

# ============================================================================
# Precision y Recall — Heatmaps por Modelo y Tarea
# ============================================================================

print("\n" + "=" * 70)
print("Precision y Recall por Modelo")
print("=" * 70)

# estos dos heatmaps complementan al de accuracy:
# sirven para ver si hay modelos que sacrifican recall por precision o viceversa
# un modelo puede tener buena accuracy pero recall bajo en clases difíciles

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
fig.suptitle("Precision y Recall Macro — Todos los Modelos x Todas las Tareas",
             fontsize=13, fontweight="bold")

# heatmap de precision (azules: más azul = más precisión)
pivot_prec = df_metricas.pivot(index="Modelo", columns="Tarea", values="Precision_Macro")
sns.heatmap(
    pivot_prec, annot=True, fmt=".3f", cmap="Blues",
    linewidths=1, linecolor="#2a3f5f", ax=axes[0],
    cbar_kws={"label": "Precision Macro"},
    annot_kws={"fontsize": 9, "fontweight": "bold"}
)
axes[0].set_title("Precision Macro", fontsize=11, fontweight="bold", pad=15)
axes[0].set_xlabel("Tarea", fontsize=10)
axes[0].set_ylabel("Modelo", fontsize=10)
axes[0].tick_params(axis='x', rotation=20, labelsize=8)
axes[0].tick_params(axis='y', rotation=0, labelsize=8)

# heatmap de recall (verdes: más verde = más recall)
pivot_rec = df_metricas.pivot(index="Modelo", columns="Tarea", values="Recall_Macro")
sns.heatmap(
    pivot_rec, annot=True, fmt=".3f", cmap="Greens",
    linewidths=1, linecolor="#2a3f5f", ax=axes[1],
    cbar_kws={"label": "Recall Macro"},
    annot_kws={"fontsize": 9, "fontweight": "bold"}
)
axes[1].set_title("Recall Macro", fontsize=11, fontweight="bold", pad=15)
axes[1].set_xlabel("Tarea", fontsize=10)
axes[1].set_ylabel("")
axes[1].tick_params(axis='x', rotation=20, labelsize=8)
axes[1].tick_params(axis='y', rotation=0, labelsize=8)

plt.tight_layout()
fig_path = OUTPUT_DIR / "12b_precision_recall_heatmaps.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Guardado: {fig_path.name}")
plt.close()

# ============================================================================
# Ranking de Modelos por Tarea
# ============================================================================

print("\n" + "=" * 70)
print("Ranking de Modelos por Tarea")
print("=" * 70)

# para cada tarea mostramos todos los modelos ordenados por accuracy
# el ganador se pinta en rojo (PALETTE[0]), el resto en azul (PALETTE[2])
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()
fig.suptitle("Ranking de Modelos por Tarea (Accuracy)", fontsize=14, fontweight="bold")

for idx, target in enumerate(targets):
    ax = axes[idx]
    df_task = (df_metricas[df_metricas["Tarea"] == target]
               .sort_values("Accuracy", ascending=True))

    max_acc = df_task["Accuracy"].max()
    bar_colors = [PALETTE[0] if acc == max_acc else PALETTE[2]
                  for acc in df_task["Accuracy"]]

    ax.barh(df_task["Modelo"], df_task["Accuracy"],
            color=bar_colors, alpha=0.85, edgecolor='white', linewidth=1)
    ax.set_xlim([0.4, 0.82])
    ax.set_xlabel("Accuracy", fontsize=9)
    ax.set_title(target, fontsize=10, fontweight="bold")
    ax.axvline(x=0.7, color='white', linestyle='--', alpha=0.4, linewidth=1)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.tick_params(axis='y', labelsize=8)

    # anotamos el accuracy al final de cada barra
    for i, (_, row) in enumerate(df_task.iterrows()):
        ax.text(row["Accuracy"] + 0.003, i, f'{row["Accuracy"]:.3f}',
                va='center', fontsize=8)

plt.tight_layout()
fig_path = OUTPUT_DIR / "13_ranking_modelos_por_tarea.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Guardado: {fig_path.name}")
plt.close()

# ============================================================================
# Dashboard Interactivo - Exploración de Datos
# ============================================================================

print("\n" + "=" * 70)
print("Dashboard Interactivo - Exploración de Datos")
print("=" * 70)

# generamos un HTML interactivo con plotly para explorar el dataset en el navegador
if len(numeric_cols) >= 2 and len(categorical_cols) > 0:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Histograma", "Box Plot", "Scatter Plot", "Distribución Categórica"),
        specs=[[{"type": "histogram"}, {"type": "box"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )

    # histograma de la primera variable numérica
    fig.add_trace(
        go.Histogram(x=df[numeric_cols[0]], name="Distribución",
                     marker_color=PALETTE[0], nbinsx=30),
        row=1, col=1
    )

    # box plot de la segunda variable numérica
    if len(numeric_cols) > 1:
        fig.add_trace(
            go.Box(y=df[numeric_cols[1]], name="Box Plot", marker_color=PALETTE[1]),
            row=1, col=2
        )

    # scatter entre las dos primeras variables numéricas
    if len(numeric_cols) >= 2:
        fig.add_trace(
            go.Scatter(x=df[numeric_cols[0]], y=df[numeric_cols[1]],
                       mode='markers', name="Scatter",
                       marker=dict(color=PALETTE[2], size=5)),
            row=2, col=1
        )

    # conteo de la primera variable categórica
    if categorical_cols:
        counts = df[categorical_cols[0]].value_counts()
        fig.add_trace(
            go.Bar(x=counts.index, y=counts.values, name="Conteos",
                   marker_color=PALETTE[3]),
            row=2, col=2
        )

    fig.update_layout(
        height=1000, width=1200,
        showlegend=False,
        paper_bgcolor="#0F1419",
        plot_bgcolor="#1a1f2e",
        font=dict(color="#E8E8E8"),
        title_text="Dashboard Interactivo — Exploración de Datos",
    )

    fig_path = OUTPUT_DIR / "14_dashboard_exploratorio.html"
    fig.write_html(str(fig_path))
    print(f"Guardado: {fig_path.name}")
else:
    print("Datos insuficientes para el dashboard")

# ============================================================================
# Clustering No Supervisado
# ============================================================================

print("\n" + "=" * 70)
print("Clustering No Supervisado")
print("=" * 70)

import sys
sys.path.append("../unsupervised")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# inicializamos todas las variables antes del try por si el import falla
clusters = None
X_cluster = None
N_CLUSTERS = None
RANDOM_STATE = None
N_INIT = None
kmeans = None
features_disponibles = []
df_cluster = None  # dataset original de clustering (con todas las columnas)

# intentamos importar los parámetros del script de clustering y reproducir el modelo
try:
    from clustering import (  # type: ignore
        RUTA_CSV, N_CLUSTERS, RANDOM_STATE, N_INIT
    )
    from sklearn.cluster import KMeans

    df_cluster = pd.read_csv(RUTA_CSV)

    # features que usamos para el clustering (las mismas que en el script original)
    features_cols = [
        "edad", "engagement_promedio", "valence_musical_pref", "energia_musical_pref",
        "hora_lectura_preferida_manana", "hora_lectura_preferida_noche", "hora_lectura_preferida_tarde",
        "velocidad_lectura_alta", "velocidad_lectura_baja", "velocidad_lectura_media",
        "contenido_visual_pref_anime", "contenido_visual_pref_documentales",
        "contenido_visual_pref_peliculas", "contenido_visual_pref_series cortas",
        "contenido_visual_pref_series largas"
    ]

    # filtramos solo las features que existan en el dataset por si cambió algo
    features_disponibles = [f for f in features_cols if f in df_cluster.columns]
    X_cluster = df_cluster[features_disponibles].fillna(0)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=N_INIT)
    clusters = kmeans.fit_predict(X_cluster)

    sil = silhouette_score(X_cluster, clusters)
    print(f"Clustering OK — {len(np.unique(clusters))} clusters, silhouette: {sil:.4f}")
except Exception as e:
    print(f"No se pudo cargar el clustering: {e}")

# ============================================================================
# Visualización de Clusters - PCA y t-SNE
# ============================================================================

if clusters is not None:
    print("\n" + "-" * 70)
    print("Clusters - PCA y t-SNE")
    print("-" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # PCA: reduce a 2D preservando la mayor varianza posible
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_cluster)

    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10',
                                s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
    # marcamos los centroides con X roja
    axes[0].scatter(pca.transform(kmeans.cluster_centers_)[:, 0],
                    pca.transform(kmeans.cluster_centers_)[:, 1],
                    c='red', marker='X', s=300, edgecolors='black', linewidth=2, label='Centroides')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11, fontweight='bold')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11, fontweight='bold')
    axes[0].set_title('Clusters - PCA (2D)', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3, linestyle='--')
    axes[0].legend()
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')

    # t-SNE: más lento que PCA pero muestra mejor la separación local entre clusters
    print("Calculando t-SNE...")
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X_cluster)

    scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='tab10',
                                s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
    axes[1].set_xlabel('t-SNE 1', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('t-SNE 2', fontsize=11, fontweight='bold')
    axes[1].set_title('Clusters - t-SNE (2D)', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3, linestyle='--')
    plt.colorbar(scatter2, ax=axes[1], label='Cluster')

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "15_clusters_pca_tsne.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Guardado: {fig_path.name}")
    plt.close()

    # ========================================================================
    # Distribución y Proporción de Clusters
    # ========================================================================

    print("\n" + "-" * 70)
    print("Distribución de Clusters")
    print("-" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    unique_clusters, counts = np.unique(clusters, return_counts=True)
    colors_clusters = plt.cm.tab10(unique_clusters)

    # conteo absoluto de usuarios por cluster
    bars = axes[0].bar(unique_clusters, counts, color=colors_clusters,
                       alpha=0.8, edgecolor='white', linewidth=2)
    axes[0].set_xlabel('Cluster', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Cantidad de Usuarios', fontsize=11, fontweight='bold')
    axes[0].set_title('Distribución de Clusters', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(count)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # pie chart con la proporción de cada cluster en el total
    colors_pie = [plt.cm.tab10(i) for i in unique_clusters]
    wedges, texts, autotexts = axes[1].pie(
        counts, labels=[f'Cluster {i}' for i in unique_clusters],
        autopct='%1.1f%%', colors=colors_pie, startangle=90,
        explode=[0.05] * len(unique_clusters)
    )
    axes[1].set_title('Proporción de Clusters', fontsize=12, fontweight='bold')

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "16_distribucion_clusters.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Guardado: {fig_path.name}")
    plt.close()

    # ========================================================================
    # Características Principales por Cluster
    # ========================================================================

    print("\n" + "-" * 70)
    print("Características por Cluster")
    print("-" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    fig.suptitle('Características Principales por Cluster', fontsize=14, fontweight='bold', y=0.995)

    # seleccionamos las 4 features más interpretables para los boxplots
    numeric_features = ['edad', 'engagement_promedio', 'valence_musical_pref', 'energia_musical_pref']

    for idx, feature in enumerate(numeric_features):
        if feature in X_cluster.columns:
            ax = axes[idx]
            # agrupamos los valores de la feature por cluster para el boxplot
            feature_data = [
                X_cluster.iloc[clusters == i, X_cluster.columns.get_loc(feature)].values
                for i in range(N_CLUSTERS)
            ]

            bp = ax.boxplot(feature_data, labels=[f'C{i}' for i in range(N_CLUSTERS)],
                            patch_artist=True, widths=0.6)

            # pintamos cada caja con el color de su cluster
            for patch, color in zip(bp['boxes'], plt.cm.tab10(range(N_CLUSTERS))):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_ylabel('Valor', fontsize=10, fontweight='bold')
            ax.set_xlabel('Cluster', fontsize=10, fontweight='bold')
            ax.set_title(feature.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')

    # cuadro con el resumen de tamaño de cada cluster
    ax = axes[-2]
    ax.axis('off')
    stats_text = "Resumen de Clusters:\n\n"
    for i in range(N_CLUSTERS):
        count = int(np.sum(clusters == i))
        pct = (count / len(clusters)) * 100
        stats_text += f"Cluster {i}: {count} usuarios ({pct:.1f}%)\n"
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # cuadro con las métricas de calidad del clustering
    ax = axes[-1]
    ax.axis('off')
    sil_score = silhouette_score(X_cluster, clusters)
    inertia = kmeans.inertia_
    info_text = "Métricas de Clustering:\n\n"
    info_text += f"Silhouette Score: {sil_score:.4f}\n"
    info_text += f"Inercia: {inertia:.2f}\n"
    info_text += f"Features usadas: {len(features_disponibles)}"
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "17_caracteristicas_por_cluster.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Guardado: {fig_path.name}")
    plt.close()

    # ========================================================================
    # Elbow + Silhouette para Selección de K
    # ========================================================================

    print("\n" + "-" * 70)
    print("Elbow y Silhouette — Selección de K")
    print("-" * 70)

    # probamos KMeans con K de 2 a 8 para justificar visualmente el K elegido
    # la inercia siempre baja con más clusters, el codo indica el punto de quiebre
    # el silhouette mide qué tan bien separados están los clusters (mayor = mejor)
    k_range = range(2, 9)
    inertias = []
    silhouettes = []

    print("Calculando KMeans para K=2 a 8...")
    for k in k_range:
        km_temp = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels_temp = km_temp.fit_predict(X_cluster)
        inertias.append(km_temp.inertia_)
        silhouettes.append(silhouette_score(X_cluster, labels_temp))

    fig, axes_elbow = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Selección de K — KMeans", fontsize=13, fontweight="bold")

    # elbow: buscamos el codo donde la inercia deja de bajar bruscamente
    axes_elbow[0].plot(list(k_range), inertias, marker="o",
                       color=PALETTE[0], linewidth=2, markersize=8)
    axes_elbow[0].axvline(x=N_CLUSTERS, color=PALETTE[1], linestyle="--",
                          alpha=0.8, linewidth=1.5, label=f"K elegido = {N_CLUSTERS}")
    axes_elbow[0].set_title("Elbow Method (Inercia)", fontsize=11, fontweight="bold")
    axes_elbow[0].set_xlabel("Número de Clusters (K)", fontsize=10)
    axes_elbow[0].set_ylabel("Inercia", fontsize=10)
    axes_elbow[0].legend(fontsize=9)
    axes_elbow[0].grid(alpha=0.3, linestyle="--")
    axes_elbow[0].set_xticks(list(k_range))

    # silhouette: buscamos el pico que indica la mejor separación entre clusters
    axes_elbow[1].plot(list(k_range), silhouettes, marker="o",
                       color=PALETTE[2], linewidth=2, markersize=8)
    axes_elbow[1].axvline(x=N_CLUSTERS, color=PALETTE[1], linestyle="--",
                          alpha=0.8, linewidth=1.5, label=f"K elegido = {N_CLUSTERS}")
    axes_elbow[1].set_title("Silhouette Score", fontsize=11, fontweight="bold")
    axes_elbow[1].set_xlabel("Número de Clusters (K)", fontsize=10)
    axes_elbow[1].set_ylabel("Silhouette Score", fontsize=10)
    axes_elbow[1].legend(fontsize=9)
    axes_elbow[1].grid(alpha=0.3, linestyle="--")
    axes_elbow[1].set_xticks(list(k_range))

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "18_elbow_silhouette.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Guardado: {fig_path.name}")
    plt.close()

    # ========================================================================
    # Varianza Explicada Acumulada — PCA
    # ========================================================================

    print("\n" + "-" * 70)
    print("PCA — Varianza Explicada Acumulada")
    print("-" * 70)

    # PCA completo (sin limitar componentes) para ver cuántos hacen falta
    # para explicar el 90% o 95% de la varianza total del dataset
    pca_full = PCA()
    pca_full.fit(X_cluster)
    var_acum = np.cumsum(pca_full.explained_variance_ratio_)

    # encontramos el número exacto de componentes para cada umbral
    n_90 = int(np.argmax(var_acum >= 0.90)) + 1
    n_95 = int(np.argmax(var_acum >= 0.95)) + 1

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(var_acum) + 1), var_acum,
            marker="o", color=PALETTE[0], linewidth=2, markersize=6)
    ax.axhline(y=0.90, color=PALETTE[1], linestyle="--", linewidth=1.5,
               label=f"90% varianza → {n_90} componentes")
    ax.axhline(y=0.95, color=PALETTE[3], linestyle="--", linewidth=1.5,
               label=f"95% varianza → {n_95} componentes")
    # marcamos cuánta varianza explican los 2 componentes que usamos en el PCA 2D
    ax.axvline(x=2, color='white', linestyle=":", linewidth=1.2, alpha=0.6,
               label=f"2D que usamos ({var_acum[1]*100:.1f}% varianza)")
    ax.set_title("Varianza Explicada Acumulada — PCA", fontsize=13, fontweight="bold")
    ax.set_xlabel("Número de Componentes", fontsize=11)
    ax.set_ylabel("Varianza Acumulada", fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "19_pca_varianza_acumulada.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Guardado: {fig_path.name}")
    print(f"  2 componentes explican: {var_acum[1]*100:.1f}% de la varianza")
    print(f"  90% de varianza con: {n_90} componentes")
    print(f"  95% de varianza con: {n_95} componentes")
    plt.close()

    # ========================================================================
    # Perfil Normalizado por Cluster
    # ========================================================================

    print("\n" + "-" * 70)
    print("Perfil Normalizado por Cluster")
    print("-" * 70)

    # calculamos el promedio de cada feature por cluster
    # luego normalizamos 0–1 por columna para poder comparar variables en distintas escalas
    # un valor de 1 significa que ese cluster tiene el mayor promedio en esa feature
    df_cluster_labels = X_cluster.copy()
    df_cluster_labels["cluster"] = clusters

    perfil = df_cluster_labels.groupby("cluster")[features_disponibles].mean()
    perfil_norm = (perfil - perfil.min()) / (perfil.max() - perfil.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(max(14, len(features_disponibles) * 0.95), 5))
    sns.heatmap(
        perfil_norm, annot=True, fmt=".2f", cmap="YlGn",
        linewidths=0.5, linecolor="#2a3f5f", ax=ax,
        annot_kws={"size": 8}
    )
    ax.set_title("Perfil Normalizado por Cluster\n(0 = mínimo del dataset, 1 = máximo)",
                 fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Feature", fontsize=10)
    ax.set_ylabel("Cluster", fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=9)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "20_perfil_clusters.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Guardado: {fig_path.name}")
    plt.close()

    # ========================================================================
    # Distribución de Targets por Cluster
    # ========================================================================

    print("\n" + "-" * 70)
    print("Distribución de Targets por Cluster")
    print("-" * 70)

    # agregamos los labels de cluster al dataset principal que tiene las columnas _rec
    # ambos datasets tienen 1000 filas en el mismo orden de registros
    df_targets_cl = df.iloc[:len(clusters)].copy()
    df_targets_cl["cluster"] = clusters

    fig, axes_tgt = plt.subplots(2, 2, figsize=(16, 12))
    axes_tgt = axes_tgt.flatten()
    fig.suptitle("Distribución de Recomendaciones por Cluster\n(proporción dentro de cada cluster)",
                 fontsize=13, fontweight="bold")

    for idx, target in enumerate(targets):
        ax = axes_tgt[idx]

        # tabla de proporciones: filas=clusters, columnas=clases del target
        ct = df_targets_cl.groupby(["cluster", target]).size().unstack(fill_value=0)
        ct_norm = ct.div(ct.sum(axis=1), axis=0)  # convertimos a proporciones

        ct_norm.plot(kind="bar", ax=ax, color=PALETTE[:len(ct_norm.columns)],
                     alpha=0.85, edgecolor='white', linewidth=1, width=0.75)
        ax.set_title(target, fontsize=10, fontweight="bold")
        ax.set_xlabel("Cluster", fontsize=9)
        ax.set_ylabel("Proporción", fontsize=9)
        ax.set_ylim([0, 0.55])
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(axis='x', rotation=0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "21_targets_por_cluster.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Guardado: {fig_path.name}")
    plt.close()

else:
    print("No hay datos de clustering disponibles")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "=" * 70)
print("PROCESO COMPLETADO")
print("=" * 70)
print(f"\nTodos los gráficos en: {OUTPUT_DIR.absolute()}\n")

archivos = sorted(OUTPUT_DIR.glob("*"))
print(f"{len(archivos)} archivo(s) generado(s):\n")
for i, archivo in enumerate(archivos, 1):
    print(f"  {i:2}. {archivo.name}")

print("\n" + "=" * 70)
