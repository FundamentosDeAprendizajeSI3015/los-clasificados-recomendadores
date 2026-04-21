"""
clustering.py
=============
Módulo de aprendizaje no supervisado (KMeans) para generar las etiquetas
de los 4 targets del sistema de recomendación:
    - cluster_libro   (10 clusters)
    - cluster_vino    (10 clusters)
    - cluster_musica  (10 clusters)
    - cluster_serie   (10 clusters)

Flujo:
    1. Recibe el DataFrame preprocesado desde eda.py  ← (en comentario hasta tener datos)
    2. Aplica KMeans independiente por dominio
    3. Retorna X (features), labels por target, modelos y scalers

Retorno para modelos supervisados:
    resultado = {
        "X"            : pd.DataFrame  — features de entrada (usuarios)
        "y_libro"      : pd.Series     — etiquetas cluster libro  (0–9)
        "y_vino"       : pd.Series     — etiquetas cluster vino   (0–9)
        "y_musica"     : pd.Series     — etiquetas cluster musica (0–9)
        "y_serie"      : pd.Series     — etiquetas cluster serie  (0–9)
        "modelos"      : dict          — KMeans entrenado por dominio
        "scalers"      : dict          — StandardScaler por dominio
        "metricas"     : dict          — silhouette_score por dominio
        "df_completo"  : pd.DataFrame  — df original + 4 columnas de etiquetas
    }
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────

N_CLUSTERS   = 10   # Posibilidades por cada variable de salida
RANDOM_STATE = 42
N_INIT       = 10   # Inicializaciones de KMeans para estabilidad

# ─────────────────────────────────────────────────────────────────────────────
# COLUMNAS POR DOMINIO
# Ajustar con los nombres reales cuando estén disponibles los datos del EDA.
# Estas son las variables del usuario que alimentan cada clustering.
# ─────────────────────────────────────────────────────────────────────────────

COLUMNAS_DOMINIO = {
    "libro": [
        # Comportamiento lector del usuario
        "engagement_score",
        "abandono_score",
        "complejidad_score",
        "ritmo_score",
        "emocional_score",
        "velocidad_lectura",
        "densidad_lectura",
        "duracion_promedio",
        "tasa_abandono",
        "sentimiento_promedio",
    ],
    "vino": [
        # Perfil sensorial / gustativo del usuario
        "pref_acidez",          # derivado de fixed_acidity / volatile_acidity
        "pref_dulzor",          # derivado de residual_sugar
        "pref_alcohol",         # derivado de alcohol
        "pref_cuerpo",          # derivado de density
        "pref_calidad_min",     # umbral mínimo de quality que acepta
        "pref_sulfatos",        # derivado de sulphates
    ],
    "musica": [
        # Preferencias musicales del usuario
        "pref_danceability",
        "pref_energy",
        "pref_valence",
        "pref_tempo",
        "pref_acousticness",
        "pref_instrumentalness",
        "pref_speechiness",
        "pref_liveness",
    ],
    "serie": [
        # Preferencias de contenido audiovisual
        "pref_runtime",         # duración preferida
        "pref_imdb_min",        # score mínimo aceptado
        "pref_tipo",            # Movie vs Series (encoded)
        "pref_genero_encoded",  # género favorito (encoded)
        "hora_consumo",         # momento del día que consume contenido
        "es_fin_semana_viewer", # si consume más en fin de semana
    ],
}

# ─────────────────────────────────────────────
# CARGA DE DATOS PREPROCESADOS
# ─────────────────────────────────────────────

def cargar_datos() -> pd.DataFrame:
    """
    Llama al módulo EDA para obtener el DataFrame preprocesado.

    PENDIENTE: descomentar cuando eda.py esté disponible.
    Se espera que eda.py exponga una función get_datos_procesados()
    que retorne un pd.DataFrame con las columnas de usuario ya limpias,
    normalizadas y sin nulos.

    Returns
    -------
    pd.DataFrame
        DataFrame con una fila por usuario y columnas de perfil.
    """

    # ── Descomentar cuando eda.py esté listo ──────────────────────────────
    # import sys, os
    # sys.path.append(os.path.join(os.path.dirname(__file__), "..", "eda"))
    # from eda import get_datos_procesados
    # df = get_datos_procesados()
    # return df
    # ──────────────────────────────────────────────────────────────────────

    # Placeholder: DataFrame vacío con columnas esperadas
    # (eliminar cuando se integre eda.py)
    todas_las_columnas = (
        ["user_id"]
        + COLUMNAS_DOMINIO["libro"]
        + COLUMNAS_DOMINIO["vino"]
        + COLUMNAS_DOMINIO["musica"]
        + COLUMNAS_DOMINIO["serie"]
    )
    print("[clustering] AVISO: eda.py aún no disponible. Usando DataFrame vacío.")
    return pd.DataFrame(columns=todas_las_columnas)


# ─────────────────────────────────────────────
# FUNCIÓN AUXILIAR: clustering por dominio
# ─────────────────────────────────────────────

def _clustering_dominio(
    df: pd.DataFrame,
    columnas: list[str],
    nombre: str,
    n_clusters: int = N_CLUSTERS,
) -> tuple[pd.Series, KMeans, StandardScaler, float]:
    """
    Aplica KMeans sobre las columnas del dominio indicado.

    Parámetros
    ----------
    df        : DataFrame completo de usuarios
    columnas  : columnas del dominio a usar como features
    nombre    : nombre del dominio (para logs)
    n_clusters: número de clusters (default 10)

    Retorna
    -------
    etiquetas   : pd.Series con el cluster asignado a cada usuario
    modelo      : KMeans entrenado
    scaler      : StandardScaler ajustado
    silhouette  : float, métrica de calidad del clustering
    """

    # Seleccionar solo columnas disponibles en el df
    cols_disponibles = [c for c in columnas if c in df.columns]
    if not cols_disponibles:
        raise ValueError(
            f"[{nombre}] Ninguna columna del dominio encontrada en el DataFrame. "
            f"Columnas esperadas: {columnas}"
        )

    X_dominio = df[cols_disponibles].copy()

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dominio)

    # KMeans
    modelo = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )
    etiquetas = modelo.fit_predict(X_scaled)

    # Silhouette score (mide qué tan bien separados están los clusters; rango -1 a 1)
    if len(np.unique(etiquetas)) > 1:
        sil = silhouette_score(X_scaled, etiquetas, sample_size=min(5000, len(X_scaled)))
    else:
        sil = -1.0

    print(f"[{nombre}] Silhouette score: {sil:.4f}  |  Columnas usadas: {cols_disponibles}")

    return pd.Series(etiquetas, name=f"cluster_{nombre}"), modelo, scaler, sil


# ─────────────────────────────────────────────
# FUNCIÓN AUXILIAR: método del codo (opcional)
# ─────────────────────────────────────────────

def metodo_del_codo(df: pd.DataFrame, dominio: str, max_k: int = 15) -> None:
    """
    Grafica la inercia (WCSS) para distintos valores de K en un dominio.
    Útil para validar que K=10 es razonable.

    Parámetros
    ----------
    df     : DataFrame de usuarios
    dominio: clave en COLUMNAS_DOMINIO ('libro', 'vino', 'musica', 'serie')
    max_k  : máximo K a evaluar
    """
    columnas = COLUMNAS_DOMINIO[dominio]
    cols_disponibles = [c for c in columnas if c in df.columns]
    X = StandardScaler().fit_transform(df[cols_disponibles])

    inercias = []
    rango_k = range(2, max_k + 1)
    for k in rango_k:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        km.fit(X)
        inercias.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(rango_k, inercias, marker="o")
    plt.title(f"Método del codo — dominio: {dominio}")
    plt.xlabel("Número de clusters (K)")
    plt.ylabel("Inercia (WCSS)")
    plt.tight_layout()
    plt.savefig(f"codo_{dominio}.png", dpi=150)
    plt.close()
    print(f"[codo] Gráfica guardada en codo_{dominio}.png")


# ─────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────

def ejecutar_clustering(df: pd.DataFrame | None = None) -> dict:
    """
    Ejecuta los 4 clusterings independientes y retorna todo lo necesario
    para que los modelos supervisados puedan entrenarse.

    Parámetros
    ----------
    df : DataFrame preprocesado. Si es None, se llama a cargar_datos().

    Retorna
    -------
    dict con claves:
        X            — pd.DataFrame: features de entrada (sin etiquetas)
        y_libro      — pd.Series:   etiquetas cluster libro  (0–9)
        y_vino       — pd.Series:   etiquetas cluster vino   (0–9)
        y_musica     — pd.Series:   etiquetas cluster musica (0–9)
        y_serie      — pd.Series:   etiquetas cluster serie  (0–9)
        modelos      — dict: KMeans entrenado por dominio
        scalers      — dict: StandardScaler ajustado por dominio
        metricas     — dict: silhouette_score por dominio
        df_completo  — pd.DataFrame: df original + 4 columnas de etiquetas
    """

    # 1. Cargar datos
    if df is None:
        df = cargar_datos()

    if df.empty:
        print("[clustering] DataFrame vacío. No se puede ejecutar el clustering.")
        return {}

    df = df.reset_index(drop=True)

    # 2. Ejecutar clustering por dominio
    dominios = ["libro", "vino", "musica", "serie"]
    modelos  = {}
    scalers  = {}
    metricas = {}
    etiquetas_dict = {}

    for dominio in dominios:
        print(f"\n── Clustering dominio: {dominio} ──")
        etiquetas, modelo, scaler, sil = _clustering_dominio(
            df,
            COLUMNAS_DOMINIO[dominio],
            dominio,
        )
        etiquetas_dict[dominio] = etiquetas
        modelos[dominio]  = modelo
        scalers[dominio]  = scaler
        metricas[dominio] = sil

    # 3. Construir df_completo (features + etiquetas)
    df_completo = df.copy()
    for dominio, etiquetas in etiquetas_dict.items():
        df_completo[f"cluster_{dominio}"] = etiquetas.values

    # 4. X: todas las columnas de perfil de usuario (excluye id y etiquetas)
    cols_etiquetas = [f"cluster_{d}" for d in dominios]
    cols_excluir   = ["user_id"] + cols_etiquetas
    X = df_completo.drop(columns=[c for c in cols_excluir if c in df_completo.columns])

    # 5. Resumen de métricas
    print("\n══ Silhouette scores ══")
    for dominio, sil in metricas.items():
        calidad = "bueno" if sil > 0.5 else "aceptable" if sil > 0.2 else "revisar K"
        print(f"  {dominio:<10} {sil:.4f}  ({calidad})")

    return {
        "X"           : X,
        "y_libro"     : etiquetas_dict["libro"],
        "y_vino"      : etiquetas_dict["vino"],
        "y_musica"    : etiquetas_dict["musica"],
        "y_serie"     : etiquetas_dict["serie"],
        "modelos"     : modelos,
        "scalers"     : scalers,
        "metricas"    : metricas,
        "df_completo" : df_completo,
    }


# ─────────────────────────────────────────────
# EJECUCIÓN DIRECTA (para pruebas)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    resultado = ejecutar_clustering()

    if resultado:
        print("\nForma de X:          ", resultado["X"].shape)
        print("Etiquetas libro:     ", resultado["y_libro"].value_counts().to_dict())
        print("Etiquetas vino:      ", resultado["y_vino"].value_counts().to_dict())
        print("Etiquetas musica:    ", resultado["y_musica"].value_counts().to_dict())
        print("Etiquetas serie:     ", resultado["y_serie"].value_counts().to_dict())
        print("\ndf_completo columnas:", list(resultado["df_completo"].columns))
