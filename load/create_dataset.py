"""
Generador de un conjunto de datos sintético para el proyecto de sistemas de recomendación.

Variables objetivo y sus clases correspondientes:
──────────────────────────────────────────────────────────────────
  genero_libro_rec   (5 clases): thriller, romance, ciencia ficción, fantasía, no ficción
  tipo_vino_rec      (5 clases): bajo en acidez, afrutado, seco, dulce, espumoso
  genero_musical_rec (5 clases): pop, rock, electrónica, jazz, reggaetón
  genero_serie_rec   (5 clases): drama, comedia, acción, ciencia ficción, terror

Total de registros : 1,000 (balanceado → 200 registros por clase en cada objetivo)
──────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Reproducibilidad ─────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

N = 1000  # Número total de registros a generar

# ── Clases objetivo (5 por objetivo → 200 registros cada una para mantener balance) ───
TARGET_CLASSES = {
    "genero_libro_rec":   ["thriller", "romance", "ciencia ficcion", "fantasia", "no ficcion"],
    "tipo_vino_rec":      ["bajo en acidez", "afrutado", "seco", "dulce", "espumoso"],
    "genero_musical_rec": ["pop", "rock", "electronica", "jazz", "reggaeton"],
    "genero_serie_rec":   ["drama", "comedia", "accion", "ciencia ficcion", "terror"],
}

ROWS_PER_CLASS = N // 5  # 200

# ── Opciones para características categóricas ─────────────────────
HORAS_LECTURA = ["manana", "tarde", "noche"]
VELOCIDAD_LECTURA = ["baja", "media", "alta"]
CONTENIDO_VISUAL = ["peliculas", "series largas", "series cortas", "documentales", "anime"]


# ── Función auxiliar: restringe un valor dentro del rango [lo, hi] ─
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ── Generación de características condicionadas por clases objetivo ─
# Cada perfil define tendencias realistas para que los clasificadores 
# puedan aprender patrones significativos, añadiendo ruido para mayor realismo.

PROFILES = {
    # ── genero_libro_rec profiles ──
    "libro_thriller":         {"edad_mu": 32, "edad_sigma": 6,  "hora_w": [0.15, 0.25, 0.60], "vel_w": [0.10, 0.30, 0.60], "engage_mu": 0.80, "valence_mu": 0.40, "energia_mu": 0.70, "visual_w": [0.15, 0.40, 0.20, 0.15, 0.10]},
    "libro_romance":          {"edad_mu": 27, "edad_sigma": 7,  "hora_w": [0.20, 0.40, 0.40], "vel_w": [0.15, 0.50, 0.35], "engage_mu": 0.70, "valence_mu": 0.75, "energia_mu": 0.45, "visual_w": [0.30, 0.35, 0.15, 0.10, 0.10]},
    "libro_ciencia_ficcion":  {"edad_mu": 30, "edad_sigma": 8,  "hora_w": [0.25, 0.30, 0.45], "vel_w": [0.10, 0.35, 0.55], "engage_mu": 0.85, "valence_mu": 0.50, "energia_mu": 0.65, "visual_w": [0.20, 0.25, 0.10, 0.20, 0.25]},
    "libro_fantasia":         {"edad_mu": 24, "edad_sigma": 6,  "hora_w": [0.20, 0.35, 0.45], "vel_w": [0.10, 0.40, 0.50], "engage_mu": 0.82, "valence_mu": 0.65, "energia_mu": 0.60, "visual_w": [0.20, 0.20, 0.10, 0.10, 0.40]},
    "libro_no_ficcion":       {"edad_mu": 40, "edad_sigma": 10, "hora_w": [0.45, 0.35, 0.20], "vel_w": [0.25, 0.45, 0.30], "engage_mu": 0.65, "valence_mu": 0.50, "energia_mu": 0.40, "visual_w": [0.15, 0.20, 0.10, 0.45, 0.10]},

    # ── tipo_vino_rec profiles ──
    "vino_bajo_en_acidez":    {"edad_mu": 35, "edad_sigma": 8,  "hora_w": [0.30, 0.35, 0.35], "vel_w": [0.20, 0.50, 0.30], "engage_mu": 0.60, "valence_mu": 0.55, "energia_mu": 0.40, "visual_w": [0.25, 0.30, 0.15, 0.20, 0.10]},
    "vino_afrutado":          {"edad_mu": 26, "edad_sigma": 5,  "hora_w": [0.20, 0.45, 0.35], "vel_w": [0.15, 0.45, 0.40], "engage_mu": 0.72, "valence_mu": 0.80, "energia_mu": 0.65, "visual_w": [0.30, 0.25, 0.20, 0.10, 0.15]},
    "vino_seco":              {"edad_mu": 42, "edad_sigma": 9,  "hora_w": [0.35, 0.35, 0.30], "vel_w": [0.25, 0.45, 0.30], "engage_mu": 0.55, "valence_mu": 0.40, "energia_mu": 0.35, "visual_w": [0.20, 0.25, 0.10, 0.35, 0.10]},
    "vino_dulce":             {"edad_mu": 23, "edad_sigma": 5,  "hora_w": [0.15, 0.40, 0.45], "vel_w": [0.10, 0.35, 0.55], "engage_mu": 0.78, "valence_mu": 0.85, "energia_mu": 0.75, "visual_w": [0.25, 0.25, 0.20, 0.05, 0.25]},
    "vino_espumoso":          {"edad_mu": 30, "edad_sigma": 7,  "hora_w": [0.20, 0.35, 0.45], "vel_w": [0.15, 0.40, 0.45], "engage_mu": 0.75, "valence_mu": 0.70, "energia_mu": 0.70, "visual_w": [0.25, 0.30, 0.15, 0.10, 0.20]},

    # ── genero_musical_rec profiles ──
    "musica_pop":             {"edad_mu": 25, "edad_sigma": 6,  "hora_w": [0.20, 0.40, 0.40], "vel_w": [0.10, 0.40, 0.50], "engage_mu": 0.75, "valence_mu": 0.80, "energia_mu": 0.70, "visual_w": [0.25, 0.30, 0.20, 0.05, 0.20]},
    "musica_rock":            {"edad_mu": 33, "edad_sigma": 8,  "hora_w": [0.20, 0.30, 0.50], "vel_w": [0.15, 0.35, 0.50], "engage_mu": 0.78, "valence_mu": 0.50, "energia_mu": 0.85, "visual_w": [0.20, 0.35, 0.15, 0.15, 0.15]},
    "musica_electronica":     {"edad_mu": 24, "edad_sigma": 5,  "hora_w": [0.10, 0.25, 0.65], "vel_w": [0.05, 0.30, 0.65], "engage_mu": 0.82, "valence_mu": 0.60, "energia_mu": 0.90, "visual_w": [0.15, 0.25, 0.20, 0.05, 0.35]},
    "musica_jazz":            {"edad_mu": 40, "edad_sigma": 10, "hora_w": [0.30, 0.35, 0.35], "vel_w": [0.25, 0.50, 0.25], "engage_mu": 0.60, "valence_mu": 0.65, "energia_mu": 0.30, "visual_w": [0.25, 0.20, 0.10, 0.35, 0.10]},
    "musica_reggaeton":       {"edad_mu": 22, "edad_sigma": 4,  "hora_w": [0.10, 0.30, 0.60], "vel_w": [0.10, 0.30, 0.60], "engage_mu": 0.80, "valence_mu": 0.85, "energia_mu": 0.90, "visual_w": [0.20, 0.25, 0.25, 0.05, 0.25]},

    # ── genero_serie_rec profiles ──
    "serie_drama":            {"edad_mu": 34, "edad_sigma": 8,  "hora_w": [0.15, 0.30, 0.55], "vel_w": [0.15, 0.45, 0.40], "engage_mu": 0.80, "valence_mu": 0.45, "energia_mu": 0.45, "visual_w": [0.20, 0.45, 0.15, 0.10, 0.10]},
    "serie_comedia":          {"edad_mu": 27, "edad_sigma": 7,  "hora_w": [0.20, 0.40, 0.40], "vel_w": [0.15, 0.45, 0.40], "engage_mu": 0.70, "valence_mu": 0.80, "energia_mu": 0.60, "visual_w": [0.25, 0.30, 0.25, 0.05, 0.15]},
    "serie_accion":           {"edad_mu": 28, "edad_sigma": 6,  "hora_w": [0.15, 0.30, 0.55], "vel_w": [0.10, 0.30, 0.60], "engage_mu": 0.82, "valence_mu": 0.55, "energia_mu": 0.85, "visual_w": [0.30, 0.30, 0.15, 0.05, 0.20]},
    "serie_ciencia_ficcion":  {"edad_mu": 30, "edad_sigma": 7,  "hora_w": [0.25, 0.30, 0.45], "vel_w": [0.10, 0.35, 0.55], "engage_mu": 0.85, "valence_mu": 0.50, "energia_mu": 0.65, "visual_w": [0.15, 0.25, 0.10, 0.15, 0.35]},
    "serie_terror":           {"edad_mu": 26, "edad_sigma": 6,  "hora_w": [0.10, 0.20, 0.70], "vel_w": [0.10, 0.35, 0.55], "engage_mu": 0.78, "valence_mu": 0.30, "energia_mu": 0.75, "visual_w": [0.25, 0.35, 0.15, 0.05, 0.20]},
}

# Mapeo de la tupla (columna_objetivo, etiqueta_clase) al nombre del perfil correspondiente
PROFILE_MAP = {
    ("genero_libro_rec",   "thriller"):         "libro_thriller",
    ("genero_libro_rec",   "romance"):          "libro_romance",
    ("genero_libro_rec",   "ciencia ficcion"):  "libro_ciencia_ficcion",
    ("genero_libro_rec",   "fantasia"):         "libro_fantasia",
    ("genero_libro_rec",   "no ficcion"):       "libro_no_ficcion",

    ("tipo_vino_rec",      "bajo en acidez"):   "vino_bajo_en_acidez",
    ("tipo_vino_rec",      "afrutado"):         "vino_afrutado",
    ("tipo_vino_rec",      "seco"):             "vino_seco",
    ("tipo_vino_rec",      "dulce"):            "vino_dulce",
    ("tipo_vino_rec",      "espumoso"):         "vino_espumoso",

    ("genero_musical_rec", "pop"):              "musica_pop",
    ("genero_musical_rec", "rock"):             "musica_rock",
    ("genero_musical_rec", "electronica"):      "musica_electronica",
    ("genero_musical_rec", "jazz"):             "musica_jazz",
    ("genero_musical_rec", "reggaeton"):        "musica_reggaeton",

    ("genero_serie_rec",   "drama"):            "serie_drama",
    ("genero_serie_rec",   "comedia"):          "serie_comedia",
    ("genero_serie_rec",   "accion"):           "serie_accion",
    ("genero_serie_rec",   "ciencia ficcion"):  "serie_ciencia_ficcion",
    ("genero_serie_rec",   "terror"):           "serie_terror",
}


def generate_features_from_profile(profile: dict, n: int) -> dict:
    """Genera columnas de características para *n* registros siguiendo el perfil dado."""
    edad = np.clip(rng.normal(profile["edad_mu"], profile["edad_sigma"], n).astype(int), 18, 65)
    hora = rng.choice(HORAS_LECTURA, size=n, p=profile["hora_w"])
    velocidad = rng.choice(VELOCIDAD_LECTURA, size=n, p=profile["vel_w"])
    engagement = np.round(
        np.clip(rng.normal(profile["engage_mu"], 0.12, n), 0.0, 1.0), 2
    )
    valence = np.round(
        np.clip(rng.normal(profile["valence_mu"], 0.12, n), 0.0, 1.0), 2
    )
    energia = np.round(
        np.clip(rng.normal(profile["energia_mu"], 0.12, n), 0.0, 1.0), 2
    )
    visual = rng.choice(CONTENIDO_VISUAL, size=n, p=profile["visual_w"])

    return {
        "edad": edad,
        "hora_lectura_preferida": hora,
        "velocidad_lectura": velocidad,
        "engagement_promedio": engagement,
        "valence_musical_pref": valence,
        "energia_musical_pref": energia,
        "contenido_visual_pref": visual,
    }


def build_dataset() -> pd.DataFrame:
    """
    Construye el conjunto de datos completo de 1000 registros.

    Estrategia:
      1. Selecciona el *primer* objetivo (genero_libro_rec) como el principal y
         genera características condicionadas a sus clases (200 registros c/u → balanceado).
      2. Para los 3 objetivos restantes, asigna clases basadas en la afinidad
         de las características usando un mecanismo de puntuación suave, luego
         re-muestrea para forzar un balance exacto.
    """

    # ── Paso 1: Objetivo primario balanceado + características condicionadas ───
    rows = []
    for cls in TARGET_CLASSES["genero_libro_rec"]:
        profile = PROFILES[PROFILE_MAP[("genero_libro_rec", cls)]]
        feats = generate_features_from_profile(profile, ROWS_PER_CLASS)
        for i in range(ROWS_PER_CLASS):
            row = {k: v[i] for k, v in feats.items()}
            row["genero_libro_rec"] = cls
            rows.append(row)

    df = pd.DataFrame(rows)

    # ── Paso 2: Asignación de los demás objetivos vía puntuación y balanceo ─
    for target_col in ["tipo_vino_rec", "genero_musical_rec", "genero_serie_rec"]:
        classes = TARGET_CLASSES[target_col]

        # Calcular puntuaciones de afinidad para cada clase
        scores = np.zeros((N, len(classes)))
        for j, cls in enumerate(classes):
            profile = PROFILES[PROFILE_MAP[(target_col, cls)]]
            # Similitud numérica (distancia tipo gaussiana)
            edad_score = np.exp(-0.5 * ((df["edad"].values - profile["edad_mu"]) / profile["edad_sigma"]) ** 2)
            engage_score = np.exp(-0.5 * ((df["engagement_promedio"].values - profile["engage_mu"]) / 0.12) ** 2)
            valence_score = np.exp(-0.5 * ((df["valence_musical_pref"].values - profile["valence_mu"]) / 0.12) ** 2)
            energia_score = np.exp(-0.5 * ((df["energia_musical_pref"].values - profile["energia_mu"]) / 0.12) ** 2)
            scores[:, j] = edad_score * engage_score * valence_score * energia_score

        # Normalizar para obtener probabilidades por fila
        row_sums = scores.sum(axis=1, keepdims=True)
        probs = scores / row_sums

        # Asignar clase mediante una extracción aleatoria ponderada
        assignments = np.array([rng.choice(classes, p=probs[i]) for i in range(N)])

        # ── Forzar balance mediante intercambio de clases ──
        # Reordenamos las asignaciones para que cada clase tenga exactamente 200 registros
        target_count = ROWS_PER_CLASS
        for _ in range(50):  # pasadas de balanceo iterativas
            class_counts = {c: np.sum(assignments == c) for c in classes}
            over = [c for c in classes if class_counts[c] > target_count]
            under = [c for c in classes if class_counts[c] < target_count]
            if not over or not under:
                break
            for oc in over:
                excess = class_counts[oc] - target_count
                if excess <= 0:
                    continue
                oc_indices = np.where(assignments == oc)[0]
                # seleccionar las filas con menor afinidad hacia esta clase sobrerrepresentada
                oc_class_idx = classes.index(oc)
                weakest = oc_indices[np.argsort(probs[oc_indices, oc_class_idx])[:excess]]
                for idx in weakest:
                    # asignar a la clase más subrepresentada con una afinidad decente
                    for uc in sorted(under, key=lambda c: class_counts[c]):
                        if class_counts[uc] < target_count:
                            assignments[idx] = uc
                            class_counts[oc] -= 1
                            class_counts[uc] += 1
                            break

        df[target_col] = assignments

    # ── Mezclar registros de forma aleatoria ─────────────────────
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return df


def main():
    df = build_dataset()

    # ── Guardar resultados ───────────────────────────────────────
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "../dataset.csv"
    df.to_csv(out_path, index=False)

    # ── Mostrar resumen en consola ───────────────────────────────
    print(f"Conjunto de datos guardado en {out_path}")
    print(f"Dimensiones: {df.shape}\n")

    for col in TARGET_CLASSES:
        print(f"── {col} ──")
        print(df[col].value_counts().to_string())
        print()


if __name__ == "__main__":
    main()