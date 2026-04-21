"""
Configuración centralizada del proyecto EDA — Clasificación Multi-Output.

Este módulo define todas las constantes, rutas y parámetros reutilizables
del análisis exploratorio. Al importarlo desde cualquier notebook o script,
se garantiza consistencia en nombres de columnas, paletas y rutas.

Ejemplo de uso:
    from src.config import NUMERICAL_FEATURES, TARGET_COLUMNS, DEFAULT_DATASET_PATH
"""

from pathlib import Path

# ╔══════════════════════════════════════════════════════════════════╗
# ║                         RUTAS                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "dataset.csv"
PLOTS_DIR = PROJECT_ROOT / "plots"

# ╔══════════════════════════════════════════════════════════════════╗
# ║                  DEFINICIÓN DE COLUMNAS                         ║
# ╚══════════════════════════════════════════════════════════════════╝

NUMERICAL_FEATURES: list[str] = [
    "edad",
    "engagement_promedio",
    "valence_musical_pref",
    "energia_musical_pref",
]

CATEGORICAL_FEATURES: list[str] = [
    "hora_lectura_preferida",
    "velocidad_lectura",
    "contenido_visual_pref",
]

TARGET_COLUMNS: list[str] = [
    "genero_libro_rec",
    "tipo_vino_rec",
    "genero_musical_rec",
    "genero_serie_rec",
]

ALL_FEATURES: list[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
ALL_COLUMNS: list[str] = ALL_FEATURES + TARGET_COLUMNS

# Tipos esperados para validación en carga
EXPECTED_DTYPES: dict[str, str] = {
    "edad": "int",
    "engagement_promedio": "float",
    "valence_musical_pref": "float",
    "energia_musical_pref": "float",
    "hora_lectura_preferida": "object",
    "velocidad_lectura": "object",
    "contenido_visual_pref": "object",
    "genero_libro_rec": "object",
    "tipo_vino_rec": "object",
    "genero_musical_rec": "object",
    "genero_serie_rec": "object",
}

# ╔══════════════════════════════════════════════════════════════════╗
# ║                  NOMBRES LEGIBLES                               ║
# ╚══════════════════════════════════════════════════════════════════╝

TARGET_DISPLAY_NAMES: dict[str, str] = {
    "genero_libro_rec": "Género de Libro Recomendado",
    "tipo_vino_rec": "Tipo de Vino Recomendado",
    "genero_musical_rec": "Género Musical Recomendado",
    "genero_serie_rec": "Género de Serie Recomendado",
}

FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "edad": "Edad",
    "hora_lectura_preferida": "Hora de Lectura Preferida",
    "velocidad_lectura": "Velocidad de Lectura",
    "engagement_promedio": "Engagement Promedio",
    "valence_musical_pref": "Valencia Musical Preferida",
    "energia_musical_pref": "Energía Musical Preferida",
    "contenido_visual_pref": "Contenido Visual Preferido",
}

ALL_DISPLAY_NAMES: dict[str, str] = {**FEATURE_DISPLAY_NAMES, **TARGET_DISPLAY_NAMES}


def display_name(col: str) -> str:
    """Retorna el nombre legible de una columna, o el nombre original si no existe."""
    return ALL_DISPLAY_NAMES.get(col, col)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                PARÁMETROS DE VISUALIZACIÓN                      ║
# ╚══════════════════════════════════════════════════════════════════╝

FIGSIZE_SMALL = (8, 5)
FIGSIZE_MEDIUM = (12, 7)
FIGSIZE_LARGE = (16, 10)
FIGSIZE_WIDE = (18, 6)
FIGSIZE_EXTRA_WIDE = (20, 6)

# Paletas de colores
PALETTE_MAIN = "viridis"
PALETTE_CATEGORICAL = "Set2"
PALETTE_DIVERGING = "RdBu_r"
PALETTE_SEQUENTIAL = "YlOrRd"

# Colores Plotly para gráficos 3D/interactivos
PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_COLORSCALE = "Viridis"

# Estilo matplotlib
PLOT_STYLE = "seaborn-v0_8-whitegrid"

# ╔══════════════════════════════════════════════════════════════════╗
# ║              PARÁMETROS DE REDUCCIÓN DE DIMENSIONALIDAD         ║
# ╚══════════════════════════════════════════════════════════════════╝

RANDOM_STATE = 42

# PCA
PCA_N_COMPONENTS_2D = 2
PCA_N_COMPONENTS_3D = 3

# UMAP
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "euclidean"
UMAP_N_COMPONENTS_2D = 2
UMAP_N_COMPONENTS_3D = 3
