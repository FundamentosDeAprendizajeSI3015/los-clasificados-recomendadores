"""
clustering.py
=============
Módulo de aprendizaje no supervisado (KMeans) para el sistema de recomendación
multi-dominio. Toma el dataset preprocesado desde EDA y genera/valida las
etiquetas de los 4 targets:

    - genero_libro_rec   (5 clases)
    - tipo_vino_rec      (5 clases)
    - genero_musical_rec (5 clases)
    - genero_serie_rec   (5 clases)

Estructura del repositorio:
    los-clasificados-recomendadores/
    ├── EDA/
    │   └── data/
    │       └── dataset_processed.csv   ← datos de entrada
    └── unsupervised/
        └── clustering.py               ← este archivo

Retorno para modelos supervisados:
    resultado = {
        "X"                  : pd.DataFrame — features de entrada (usuarios)
        "y_libro"            : pd.Series    — etiquetas libro  (0–4)
        "y_vino"             : pd.Series    — etiquetas vino   (0–4)
        "y_musica"           : pd.Series    — etiquetas musica (0–4)
        "y_serie"            : pd.Series    — etiquetas serie  (0–4)
        "modelos"            : dict         — KMeans entrenado por dominio
        "metricas"           : dict         — silhouette_score por dominio
        "df_completo"        : pd.DataFrame — df original + etiquetas KMeans
        "cluster_labels"     : pd.DataFrame — solo las 4 columnas kmeans_*
    }
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────

# K=5 porque el dataset ya tiene 5 clases por cada target
N_CLUSTERS   = 5
RANDOM_STATE = 42
N_INIT       = 10

# Ruta al CSV relativa a este archivo (unsupervised/ → EDA/data/)
RUTA_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "EDA", "data", "dataset_processed.csv"
)

# ─────────────────────────────────────────────────────────────────────────────
# COLUMNAS POR DOMINIO
# El dataset ya viene escalado y con dummies aplicados desde el EDA.
# Cada dominio usa las features del usuario más relevantes para ese target.
# ─────────────────────────────────────────────────────────────────────────────

# Todas las features disponibles en el CSV (sin los 4 targets)
FEATURES_USUARIO = [
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

# Subconjunto de features por dominio (las más informativas para cada target)
COLUMNAS_DOMINIO = {
    "libro": [
        "edad",
        "engagement_promedio",
        "velocidad_lectura_alta",
        "velocidad_lectura_baja",
        "velocidad_lectura_media",
        "hora_lectura_preferida_manana",
        "hora_lectura_preferida_noche",
        "hora_lectura_preferida_tarde",
    ],
    "vino": [
        "edad",
        "engagement_promedio",
        "valence_musical_pref",
        "energia_musical_pref",
    ],
    "musica": [
        "valence_musical_pref",
        "energia_musical_pref",
        "edad",
        "hora_lectura_preferida_manana",
        "hora_lectura_preferida_noche",
        "hora_lectura_preferida_tarde",
    ],
    "serie": [
        "contenido_visual_pref_anime",
        "contenido_visual_pref_documentales",
        "contenido_visual_pref_peliculas",
        "contenido_visual_pref_series cortas",
        "contenido_visual_pref_series largas",
        "edad",
        "engagement_promedio",
    ],
}

# Nombres de los targets en el CSV
TARGETS = {
    "libro" : "genero_libro_rec",
    "vino"  : "tipo_vino_rec",
    "musica": "genero_musical_rec",
    "serie" : "genero_serie_rec",
}

# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────

def cargar_datos() -> pd.DataFrame:
    """
    Carga el dataset preprocesado desde EDA/data/dataset_processed.csv.

    La ruta se resuelve de forma relativa a este archivo, de modo que
    funciona sin importar desde dónde se ejecute el script.

    Returns
    -------
    pd.DataFrame
        DataFrame con features de usuario y las 4 columnas target.

    Raises
    ------
    FileNotFoundError
        Si el CSV no se encuentra en la ruta esperada.
    """
    ruta = RUTA_CSV
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró el dataset en:\n  {ruta}\n"
            "Verifica que la carpeta EDA/data/ exista y contenga dataset_processed.csv"
        )

    df = pd.read_csv(ruta)
    print(f"[clustering] Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    return df


# ─────────────────────────────────────────────
# FUNCIÓN AUXILIAR: clustering por dominio
# ─────────────────────────────────────────────

def _clustering_dominio(
    df: pd.DataFrame,
    columnas: list,
    nombre: str,
    n_clusters: int = N_CLUSTERS,
):
    """
    Aplica KMeans sobre las columnas del dominio indicado.

    El dataset ya viene escalado desde el EDA, por lo que no se aplica
    un scaler adicional aquí.

    Parámetros
    ----------
    df        : DataFrame completo de usuarios
    columnas  : features del dominio a usar
    nombre    : nombre del dominio (para logs)
    n_clusters: número de clusters

    Retorna
    -------
    etiquetas  : pd.Series con el cluster asignado (0 a n_clusters-1)
    modelo     : KMeans entrenado
    silhouette : float, métrica de calidad del clustering
    """
    cols_disponibles = [c for c in columnas if c in df.columns]
    if not cols_disponibles:
        raise ValueError(
            f"[{nombre}] Ninguna columna del dominio encontrada. "
            f"Esperadas: {columnas}"
        )

    X_dominio = df[cols_disponibles].values

    modelo = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )
    etiquetas = modelo.fit_predict(X_dominio)

    sil = (
        silhouette_score(X_dominio, etiquetas, sample_size=min(5000, len(etiquetas)))
        if len(np.unique(etiquetas)) > 1
        else -1.0
    )

    print(f"  [{nombre}] Silhouette: {sil:.4f}  |  Cols: {cols_disponibles}")
    return pd.Series(etiquetas, name=f"kmeans_{nombre}"), modelo, sil


# ─────────────────────────────────────────────
# FUNCIÓN AUXILIAR: método del codo
# ─────────────────────────────────────────────

def metodo_del_codo(df: pd.DataFrame, dominio: str, max_k: int = 12) -> None:
    """
    Grafica la inercia (WCSS) para distintos K en un dominio.
    Útil para confirmar que N_CLUSTERS=5 es razonable.

    Guarda la imagen como codo_{dominio}.png en la carpeta de este script.

    Parámetros
    ----------
    df     : DataFrame de usuarios
    dominio: clave en COLUMNAS_DOMINIO
    max_k  : máximo K a evaluar (default 12)
    """
    columnas = COLUMNAS_DOMINIO[dominio]
    cols_disponibles = [c for c in columnas if c in df.columns]
    X = df[cols_disponibles].values

    inercias = []
    rango_k = range(2, max_k + 1)
    for k in rango_k:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        km.fit(X)
        inercias.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(rango_k, inercias, marker="o")
    plt.axvline(x=N_CLUSTERS, color="red", linestyle="--", label=f"K={N_CLUSTERS} actual")
    plt.title(f"Método del codo — dominio: {dominio}")
    plt.xlabel("Número de clusters (K)")
    plt.ylabel("Inercia (WCSS)")
    plt.legend()
    plt.tight_layout()

    ruta_salida = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"codo_{dominio}.png")
    plt.savefig(ruta_salida, dpi=150)
    plt.close()
    print(f"[codo] Guardada: {ruta_salida}")


# ─────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────

def ejecutar_clustering(df: pd.DataFrame = None) -> dict:
    """
    Ejecuta los 4 clusterings independientes y retorna todo lo necesario
    para que los modelos supervisados puedan entrenarse.

    Parámetros
    ----------
    df : DataFrame preprocesado. Si es None, lo carga desde el CSV.

    Retorna
    -------
    dict con claves:
        X              — pd.DataFrame: features de entrada (sin targets)
        y_libro        — pd.Series:   etiquetas libro  (0–4)
        y_vino         — pd.Series:   etiquetas vino   (0–4)
        y_musica       — pd.Series:   etiquetas musica (0–4)
        y_serie        — pd.Series:   etiquetas serie  (0–4)
        modelos        — dict: KMeans entrenado por dominio
        metricas       — dict: silhouette_score por dominio
        df_completo    — pd.DataFrame: df original + columnas kmeans_*
        cluster_labels — pd.DataFrame: solo las 4 columnas kmeans_*

    Nota sobre los targets
    ----------------------
    El CSV ya contiene las etiquetas originales del EDA (genero_libro_rec,
    tipo_vino_rec, genero_musical_rec, genero_serie_rec). Este módulo genera
    etiquetas KMeans adicionales (kmeans_libro, kmeans_vino, kmeans_musica,
    kmeans_serie) que permiten comparar ambas asignaciones. Los modelos
    supervisados usan las etiquetas originales del EDA (y_libro, y_vino, etc.)
    por ser las más completas; las etiquetas KMeans quedan disponibles para
    análisis exploratorio o como feature adicional.
    """

    # 1. Cargar datos
    if df is None:
        df = cargar_datos()

    df = df.reset_index(drop=True)

    # Separar features
    X = df[[c for c in FEATURES_USUARIO if c in df.columns]].copy()

    print(f"\n[clustering] Features de entrada: {list(X.columns)}")
    print(f"[clustering] Filas: {len(X)}\n")

    # 2. Clustering por dominio
    dominios = ["libro", "vino", "musica", "serie"]
    modelos  = {}
    metricas = {}
    kmeans_labels = {}

    print("── Ejecutando 4 KMeans independientes ──")
    for dominio in dominios:
        etiquetas, modelo, sil = _clustering_dominio(
            df, COLUMNAS_DOMINIO[dominio], dominio
        )
        kmeans_labels[dominio] = etiquetas
        modelos[dominio]       = modelo
        metricas[dominio]      = sil

    # 3. df_completo: original + columnas kmeans_*
    df_completo = df.copy()
    for dominio, etiquetas in kmeans_labels.items():
        df_completo[f"kmeans_{dominio}"] = etiquetas.values

    cluster_labels = df_completo[[f"kmeans_{d}" for d in dominios]]

    # 4. Resumen
    print("\n══ Silhouette scores ══")
    for dominio, sil in metricas.items():
        calidad = "bueno" if sil > 0.5 else "aceptable" if sil > 0.2 else "revisar K"
        print(f"  {dominio:<10} {sil:.4f}  ({calidad})")

    return {
        "X"             : X,
        # Targets originales del EDA (usar estos para entrenar los modelos)
        "y_libro"       : df[TARGETS["libro"]],
        "y_vino"        : df[TARGETS["vino"]],
        "y_musica"      : df[TARGETS["musica"]],
        "y_serie"       : df[TARGETS["serie"]],
        # KMeans entrenados (para predecir cluster en usuarios nuevos)
        "modelos"       : modelos,
        "metricas"      : metricas,
        # DataFrames
        "df_completo"   : df_completo,
        "cluster_labels": cluster_labels,
    }


# ─────────────────────────────────────────────
# EJECUCIÓN DIRECTA (para pruebas)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    resultado = ejecutar_clustering()

    print("\n── Resultado ──")
    print("X shape:          ", resultado["X"].shape)
    print("X columnas:       ", list(resultado["X"].columns))
    print("y_libro valores:  ", sorted(resultado["y_libro"].unique()))
    print("y_vino valores:   ", sorted(resultado["y_vino"].unique()))
    print("y_musica valores: ", sorted(resultado["y_musica"].unique()))
    print("y_serie valores:  ", sorted(resultado["y_serie"].unique()))
    print("\ndf_completo cols: ", list(resultado["df_completo"].columns))
    print("cluster_labels:\n", resultado["cluster_labels"].head())