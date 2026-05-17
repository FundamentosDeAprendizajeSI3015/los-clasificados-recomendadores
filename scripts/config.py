"""
config.py — Configuración centralizada del pipeline.
Rutas, constantes, features, paleta de colores y funciones helper.
"""

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════
# RUTAS
# ═══════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
GRAFICAS_DIR = REPORTS_DIR / "graficas"
JSON_DIR = REPORTS_DIR / "json"

DATASET_RAW = DATA_DIR / "dataset.csv"
DATASET_PROCESSED = DATA_DIR / "dataset_processed.csv"

# ═══════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CLUSTERS = 5
N_INIT = 10
MAX_SAMPLE = 50_000  # Para operaciones pesadas (t-SNE, SVM). None = todo.

# ═══════════════════════════════════════════════════════════════════
# FEATURES Y TARGETS
# ═══════════════════════════════════════════════════════════════════
NUMERICAL_FEATURES = [
    "edad", "engagement_promedio",
    "valence_musical_pref", "energia_musical_pref",
]
CATEGORICAL_FEATURES = [
    "hora_lectura_preferida", "velocidad_lectura", "contenido_visual_pref",
]
TARGET_COLUMNS = [
    "genero_libro_rec", "tipo_vino_rec",
    "genero_musical_rec", "genero_serie_rec",
]

FEATURES_PROCESSED = [
    "edad", "engagement_promedio", "valence_musical_pref", "energia_musical_pref",
    "hora_lectura_preferida_manana", "hora_lectura_preferida_noche",
    "hora_lectura_preferida_tarde",
    "velocidad_lectura_alta", "velocidad_lectura_baja", "velocidad_lectura_media",
    "contenido_visual_pref_anime", "contenido_visual_pref_documentales",
    "contenido_visual_pref_peliculas", "contenido_visual_pref_series cortas",
    "contenido_visual_pref_series largas",
]

COLUMNAS_DOMINIO = {
    "libro": ["edad", "engagement_promedio", "velocidad_lectura_alta",
              "velocidad_lectura_baja", "velocidad_lectura_media",
              "hora_lectura_preferida_manana", "hora_lectura_preferida_noche",
              "hora_lectura_preferida_tarde"],
    "vino": ["edad", "engagement_promedio", "valence_musical_pref",
             "energia_musical_pref"],
    "musica": ["valence_musical_pref", "energia_musical_pref", "edad",
               "hora_lectura_preferida_manana", "hora_lectura_preferida_noche",
               "hora_lectura_preferida_tarde"],
    "serie": ["contenido_visual_pref_anime", "contenido_visual_pref_documentales",
              "contenido_visual_pref_peliculas", "contenido_visual_pref_series cortas",
              "contenido_visual_pref_series largas", "edad", "engagement_promedio"],
}

TARGETS_DOMINIO = {
    "libro": "genero_libro_rec",
    "vino": "tipo_vino_rec",
    "musica": "genero_musical_rec",
    "serie": "genero_serie_rec",
}

# ═══════════════════════════════════════════════════════════════════
# VISUAL
# ═══════════════════════════════════════════════════════════════════
PALETTE = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#F7DC6F"]

PLOT_STYLE = {
    "figure.facecolor": "#0F1419",
    "axes.facecolor": "#1a1f2e",
    "axes.edgecolor": "#2a3f5f",
    "axes.labelcolor": "#E8E8E8",
    "xtick.color": "#E8E8E8",
    "ytick.color": "#E8E8E8",
    "text.color": "#E8E8E8",
    "grid.color": "#2a3f5f",
}


# ═══════════════════════════════════════════════════════════════════
# FUNCIONES HELPER
# ═══════════════════════════════════════════════════════════════════

def aplicar_tema():
    """Aplica el tema oscuro a matplotlib + seaborn."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(PLOT_STYLE)


def limpiar_reports():
    """Elimina todo dentro de reports/ para evitar mezclas entre ejecuciones."""
    for subdir in [GRAFICAS_DIR, JSON_DIR]:
        if subdir.exists():
            shutil.rmtree(subdir)
        subdir.mkdir(parents=True, exist_ok=True)
    print("[config] Directorio reports/ limpiado\n")


def save_figure(fig, paso: str, nombre: str, dpi=300):
    """Guarda figura en reports/graficas/{paso}_{nombre}.png y la cierra."""
    GRAFICAS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{paso}_{nombre}.png"
    path = GRAFICAS_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [grafica] {filename}")
    return path


def save_json(data, paso: str, nombre: str):
    """Guarda dict como JSON en reports/json/{paso}_{nombre}.json."""
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{paso}_{nombre}.json"
    path = JSON_DIR / filename
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"  [json] {filename}")
    return path
