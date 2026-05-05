# Supervised

Módulo de aprendizaje supervisado. Cada algoritmo entrena y evalúa **4 targets de clasificación** a partir de un mismo conjunto de features de usuario.

## Dataset

### Features de entrada

| Feature | Tipo | Ejemplo |
| --- | --- | --- |
| `edad` | numérica | 28 |
| `hora_lectura_preferida` | categórica | noche |
| `velocidad_lectura` | categórica | alta |
| `engagement_promedio` | numérica | 0.75 |
| `valence_musical_pref` | numérica | 0.8 |
| `energia_musical_pref` | numérica | 0.7 |
| `contenido_visual_pref` | categórica | series largas |

### Targets

| Target | Ejemplo |
| --- | --- |
| `genero_libro_rec` | thriller |
| `tipo_vino_rec` | bajo en acidez, alta graduación |
| `genero_musical_rec` | pop energético |
| `genero_serie_rec` | drama |

## Algoritmos

Todos los scripts comparten la misma interfaz CLI y el mismo flujo: preprocesamiento → entrenamiento con split estratificado (80/20) → evaluación con accuracy, F1 macro y classification report.

| Algoritmo | Modelo | Preprocesamiento numérico | Preprocesamiento categórico | Hiperparámetros clave |
| --- | --- | --- | --- | --- |
| **Árbol de Decisión** | `DecisionTreeClassifier` | `SimpleImputer(median)` | `SimpleImputer(most_frequent)` + `OneHotEncoder` | `balanced`, `max_depth=12`, `min_samples_split=10`, `min_samples_leaf=3` |
| **Regresión Logística** | `LogisticRegression` | `SimpleImputer(median)` + `StandardScaler` | `SimpleImputer(most_frequent)` + `OneHotEncoder` | `balanced`, `max_iter=2000` |
| **RandomForest** | `RandomForestClassifier` | passthrough | `OneHotEncoder` | `n_estimators=300`, `balanced` |
| **SVM** | `SVC` | `StandardScaler` | `OneHotEncoder` | 4 kernels (linear, poly d2, poly d3, rbf), `balanced`, `C=1.0`, `gamma=scale` |

## Estructura de carpetas

```text
Supervised/
├── README.md
├── Árbol de Decisión/
│   └── scr/script.py
├── Regresión Logística/
│   └── scr/script.py
├── RandomForest/
│   ├── scr/script.py
│   └── reports/<target>/
└── SVM/
    ├── scr/script.py
    └── reports/<target>/<kernel>/
```

Cada algoritmo genera sus resultados en `reports/<target>/` al ejecutarse. SVM además subdivide por kernel.

## Uso

Todos los scripts comparten la misma interfaz CLI:

```bash
python "<algoritmo>/scr/script.py" --data <ruta_csv> --task both
```

**Ejemplos:**

```bash
python "Árbol de Decisión/scr/script.py" --data datos.csv --task both
python "Regresión Logística/scr/script.py" --data datos.csv --task both
python "RandomForest/scr/script.py"        --data datos.csv --task both
python "SVM/scr/script.py"                 --data datos.csv --task both
```

> **⚠️ Nota sobre SVM:** A diferencia de los otros 3 scripts, SVM guarda los reports **relativos al directorio de trabajo actual** (`Path("reports")`), no relativos al script (`Path(__file__).parents[2] / "reports"`). Para que los reports queden dentro de `SVM/`, ejecutar desde la carpeta `SVM/`:
> ```bash
> cd SVM && python scr/script.py --data <ruta_csv> --task both
> ```
> Además, SVM no valida la existencia de `--eval-data` ni las columnas de entrada antes de evaluar.

### Opciones CLI

| Flag | Descripción | Default |
| --- | --- | --- |
| `--data` | CSV de entrada (requerido) | — |
| `--task` | `train`, `evaluate` o `both` | `both` |
| `--eval-data` | CSV alternativo para evaluación | usa `--data` |
| `--test-size` | Proporción para test | `0.2` |
| `--random-state` | Semilla | `42` |
