"""
Fake dataset generator for the recommender system project.

Target variables and their classes:
──────────────────────────────────────────────────────────────────
  genero_libro_rec   (5 classes): thriller, romance, ciencia ficcion, fantasia, no ficcion
  tipo_vino_rec      (5 classes): bajo en acidez, afrutado, seco, dulce, espumoso
  genero_musical_rec (5 classes): pop, rock, electronica, jazz, reggaeton
  genero_serie_rec   (5 classes): drama, comedia, accion, ciencia ficcion, terror

Total rows : 1,000  (balanced → 200 rows per class in each target)
──────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── reproducibility ──────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

N = 1_000_000  # total number of rows

# ── target classes (5 per target → 200000 rows each for balance) ───
TARGET_CLASSES = {
    "genero_libro_rec":   ["thriller", "romance", "ciencia ficcion", "fantasia", "no ficcion"],
    "tipo_vino_rec":      ["bajo en acidez", "afrutado", "seco", "dulce", "espumoso"],
    "genero_musical_rec": ["pop", "rock", "electronica", "jazz", "reggaeton"],
    "genero_serie_rec":   ["drama", "comedia", "accion", "ciencia ficcion", "terror"],
}

ROWS_PER_CLASS = N // 5  # 200000

# ── categorical feature options ─────────────────────────────────
HORAS_LECTURA = ["manana", "tarde", "noche"]
VELOCIDAD_LECTURA = ["baja", "media", "alta"]
CONTENIDO_VISUAL = ["peliculas", "series largas", "series cortas", "documentales", "anime"]


# ── helper: clamp a value within [lo, hi] ───────────────────────
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ── feature generation conditioned on target classes ─────────────
# Each profile defines realistic tendencies so classifiers can learn
# meaningful patterns, while still adding noise for realism.

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

# Map from (target_col, class_label) → profile key
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
    """Generate feature columns for *n* rows following the given profile."""
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
    Build the full 1000-row dataset.

    Strategy:
      1. Pick the *first* target (genero_libro_rec) as the primary driver and
         generate features conditioned on its classes (200 rows each → balanced).
      2. For the remaining 3 targets, assign classes based on feature affinity
         using a soft scoring mechanism, then resample to enforce exact balance.
    """

    # ── Step 1: balanced primary target + conditioned features ───
    rows = []
    for cls in TARGET_CLASSES["genero_libro_rec"]:
        profile = PROFILES[PROFILE_MAP[("genero_libro_rec", cls)]]
        feats = generate_features_from_profile(profile, ROWS_PER_CLASS)
        for i in range(ROWS_PER_CLASS):
            row = {k: v[i] for k, v in feats.items()}
            row["genero_libro_rec"] = cls
            rows.append(row)

    df = pd.DataFrame(rows)

    # ── Step 2: assign remaining targets via scoring + balancing ─
    for target_col in ["tipo_vino_rec", "genero_musical_rec", "genero_serie_rec"]:
        classes = TARGET_CLASSES[target_col]

        # Compute affinity scores for each class
        scores = np.zeros((N, len(classes)))
        for j, cls in enumerate(classes):
            profile = PROFILES[PROFILE_MAP[(target_col, cls)]]
            # numerical similarity (gaussian-like distance)
            edad_score = np.exp(-0.5 * ((df["edad"].values - profile["edad_mu"]) / profile["edad_sigma"]) ** 2)
            engage_score = np.exp(-0.5 * ((df["engagement_promedio"].values - profile["engage_mu"]) / 0.12) ** 2)
            valence_score = np.exp(-0.5 * ((df["valence_musical_pref"].values - profile["valence_mu"]) / 0.12) ** 2)
            energia_score = np.exp(-0.5 * ((df["energia_musical_pref"].values - profile["energia_mu"]) / 0.12) ** 2)
            scores[:, j] = edad_score * engage_score * valence_score * energia_score

        # Normalize to probabilities per row
        row_sums = scores.sum(axis=1, keepdims=True)
        probs = scores / row_sums

        # Assign class by weighted random draw
        assignments = np.array([rng.choice(classes, p=probs[i]) for i in range(N)])

        # ── Enforce balance via swapping ──
        # We reshuffle assignments so that each class has exactly 200 rows
        target_count = ROWS_PER_CLASS
        for _ in range(50):  # iterative balancing passes
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
                # pick the rows with lowest affinity for this overrepresented class
                oc_class_idx = classes.index(oc)
                weakest = oc_indices[np.argsort(probs[oc_indices, oc_class_idx])[:excess]]
                for idx in weakest:
                    # assign to the most underrepresented class with decent affinity
                    for uc in sorted(under, key=lambda c: class_counts[c]):
                        if class_counts[uc] < target_count:
                            assignments[idx] = uc
                            class_counts[oc] -= 1
                            class_counts[uc] += 1
                            break

        df[target_col] = assignments

    # ── Shuffle rows ─────────────────────────────────────────────
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return df


def main():
    df = build_dataset()

    # ── Save ─────────────────────────────────────────────────────
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "../dataset.csv"
    df.to_csv(out_path, index=False)

    # ── Print summary ────────────────────────────────────────────
    print(f"Dataset saved to {out_path}")
    print(f"Shape: {df.shape}\n")

    for col in TARGET_CLASSES:
        print(f"── {col} ──")
        print(df[col].value_counts().to_string())
        print()


if __name__ == "__main__":
    main()