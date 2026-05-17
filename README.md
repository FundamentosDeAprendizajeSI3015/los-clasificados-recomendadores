# Los Clasificados Recomendadores

Sistema de recomendación multi-dominio basado en aprendizaje automático que, dado el perfil de un usuario, recomienda simultáneamente un **género de libro**, un **tipo de vino**, un **género musical** y un **género de serie/película**.

---

## ¿Qué hace el proyecto?

El sistema toma variables de perfil de usuario (edad, engagement, preferencias de contenido, preferencias musicales, hábitos de lectura) y produce **4 recomendaciones en paralelo**, una por dominio. Cada recomendación es una clasificación en 5 categorías posibles.

El pipeline completo combina:
- **Aprendizaje no supervisado** (KMeans) para analizar y validar la estructura de los datos
- **Aprendizaje supervisado** (Random Forest, SVM, Regresión Logística, Árbol de Decisión) para entrenamiento y predicción de los 4 targets
- **Análisis exploratorio** para comprensión y preprocesamiento del dataset
- **Reducción de dimensionalidad** (PCA, t-SNE) para visualización y validación
- **Visualización de métricas y resultados** para evaluación comparativa de modelos

---

## Arquitectura del Proyecto

```
gitCLUSTER_local/
│
├── data/                              ← Datos del proyecto
│   ├── dataset.csv                    ← Dataset crudo (1M registros × 11 columnas)
│   └── dataset_processed.csv          ← Dataset preprocesado (escalado + OHE + LabelEncoder)
│
├── reports/                           ← Resultados generados automáticamente
│   ├── graficas/                      ← Imágenes PNG para lectura humana
│   │   ├── paso01_preprocessing_*.png
│   │   ├── paso02_eda_*.png
│   │   ├── paso03_clustering_*.png
│   │   ├── paso04_dimensionality_*.png
│   │   ├── paso05_supervised_*.png
│   │   ├── paso06_metrics_*.png
│   │   └── paso07_comparison_*.png
│   └── json/                          ← Datos JSON estructurados
│       ├── paso01_preprocessing_*.json
│       ├── paso02_eda_*.json
│       └── ...
│
├── scripts/                           ← Pipeline modular
│   ├── config.py                      ← Configuración centralizada
│   ├── Main.py                        ← Orquestador del pipeline
│   ├── paso01_preprocessing.py        ← Carga, limpieza y escalado (opcional)
│   ├── paso02_eda.py                  ← Análisis exploratorio de datos
│   ├── paso03_clustering.py           ← Clustering KMeans no supervisado
│   ├── paso04_dimensionality.py       ← Reducción de dimensionalidad (PCA + t-SNE)
│   ├── paso05_supervised.py           ← Entrenamiento de modelos supervisados
│   ├── paso06_metrics.py              ← Consolidación de métricas
│   └── paso07_comparison.py           ← Comparación final y rankings
│
├── new/                               ← Código original de cada integrante (referencia)
│
└── README.md                          ← Este archivo
```

---

## Pipeline Completo

```
data/dataset.csv
     │
     ▼
[ paso01 ] Preprocessing (OPCIONAL)
     │     Carga → Validación → Escalado → OHE → LabelEncoder
     │     Genera: data/dataset_processed.csv
     ▼
[ paso02 ] EDA
     │     Overview → Distribuciones → Correlación → Targets → Faltantes
     ▼
[ paso03 ] Clustering
     │     4 KMeans independientes (libro, vino, música, serie)
     │     + KMeans global → Silhouette + Elbow
     ▼
[ paso04 ] Dimensionality
     │     PCA 2D + t-SNE 2D → Varianza explicada
     │     Perfil normalizado + Boxplots por cluster
     ▼
[ paso05 ] Supervised
     │     7 modelos × 4 targets = 28 entrenamientos
     │     RF, DecTree, LogReg, SVM (lineal, poly d2, poly d3, radial)
     ▼
[ paso06 ] Metrics
     │     Heatmaps de Accuracy, Precision, Recall
     │     Promedio por modelo
     ▼
[ paso07 ] Comparison
           Ranking por tarea → Modelos base → Targets por cluster
```

> **Nota:** Cada paso genera automáticamente reports en `reports/graficas/` (imágenes PNG) y `reports/json/` (datos estructurados). Al inicio de cada ejecución, el pipeline **limpia completamente** la carpeta `reports/` para evitar mezclas entre ejecuciones.

---

## Descripción de cada Paso

### `config.py` — Configuración Centralizada
Contiene todas las rutas, constantes, listas de features/targets, paleta de colores, tema visual oscuro y funciones helper (`save_figure`, `save_json`, `limpiar_reports`). Todos los demás módulos importan desde aquí para garantizar consistencia.

---

### Paso 01 — `paso01_preprocessing.py` · **Nicolás** (carga) + **Agustín** (preprocesamiento)
**Estado:** Opcional — se salta si `data/dataset_processed.csv` ya existe.

| Qué hace | Detalle |
|---|---|
| Carga datos crudos | Lee `data/dataset.csv` |
| Valida esquema | Verifica que todas las columnas esperadas existen |
| Escala numéricas | `StandardScaler` sobre edad, engagement, valence, energía |
| Codifica categóricas | `OneHotEncoder` sobre hora_lectura, velocidad, contenido_visual |
| Codifica targets | `LabelEncoder` (string → 0–4) para los 4 targets |
| Exporta | Guarda `data/dataset_processed.csv` |

**Reports:** `paso01_preprocessing_resumen_esquema.png`, `paso01_preprocessing_resumen.json`

---

### Paso 02 — `paso02_eda.py` · **Agustín**
Análisis exploratorio sobre el dataset **crudo** (`dataset.csv`) para entender la naturaleza de los datos antes de cualquier transformación.

| Qué hace | Detalle |
|---|---|
| Overview | Panel con conteo de registros, columnas, numéricas, categóricas |
| Distribuciones numéricas | Histogramas de edad, engagement, valence, energía |
| Distribuciones categóricas | Barras horizontales de hora_lectura, velocidad, contenido_visual |
| Correlación | Heatmap triangular con coeficientes de Pearson |
| Targets | Distribución de las 5 clases por cada uno de los 4 recomendadores |
| Valores faltantes | Detección y visualización de nulos |

**Reports:** `paso02_eda_overview.png`, `paso02_eda_distribuciones_numericas.png`, `paso02_eda_distribuciones_categoricas.png`, `paso02_eda_correlacion.png`, `paso02_eda_targets.png`, `paso02_eda_estadisticas.json`

---

### Paso 03 — `paso03_clustering.py` · **Camilo**
Clustering no supervisado con KMeans sobre el dataset **procesado** (`dataset_processed.csv`).

| Qué hace | Detalle |
|---|---|
| 4 KMeans por dominio | Libro (8 features), Vino (4), Música (6), Serie (7) |
| KMeans global | Sobre las 15 features procesadas, K=5 |
| Silhouette score | Calidad del clustering por dominio y global |
| Método del codo | Inercia y silhouette para K=2..9, justificando K=5 |

**Reports:** `paso03_clustering_distribucion_clusters.png`, `paso03_clustering_elbow_silhouette.png`, `paso03_clustering_metricas.json`

---

### Paso 04 — `paso04_dimensionality.py` · **Isabella**
Reducción de dimensionalidad y visualización de la estructura de clusters.

| Qué hace | Detalle |
|---|---|
| PCA 2D | Proyección con centroides marcados y varianza explicada |
| t-SNE 2D | Embedding no lineal (con sampling para eficiencia) |
| Varianza acumulada | Curva PCA completa con umbrales 90% y 95% |
| Perfil por cluster | Heatmap normalizado (0=min, 1=max) de todas las features |
| Boxplots | Distribución de features principales por cluster |

**Reports:** `paso04_dimensionality_pca_tsne.png`, `paso04_dimensionality_varianza_pca.png`, `paso04_dimensionality_perfil_clusters.png`, `paso04_dimensionality_boxplots_features.png`, `paso04_dimensionality_metricas.json`

---

### Paso 05 — `paso05_supervised.py` · **Luis / Alejandro**
Entrenamiento de 7 modelos supervisados sobre los 4 targets (28 entrenamientos).

| Modelo | Pipeline interno |
|---|---|
| **RandomForest** | passthrough (numéricas) + OHE (categóricas), 300 estimators |
| **DecisionTree** | Imputer mediana + OHE, max_depth=12 |
| **LogisticRegression** | Imputer + Scaler + OHE, max_iter=2000 |
| **SVM Lineal** | Scaler + OHE, kernel=linear |
| **SVM Poly d2** | Scaler + OHE, kernel=poly grado 2 |
| **SVM Poly d3** | Scaler + OHE, kernel=poly grado 3 |
| **SVM Radial** | Scaler + OHE, kernel=rbf |

> **Nota:** SVM usa un submuestreo configurable (`MAX_SAMPLE` en `config.py`) para datasets grandes, ya que su complejidad es O(n²).

Cada modelo usa `train_test_split` con `stratify` y `class_weight="balanced"`.

**Reports:** `paso05_supervised_resultados_modelos.png`, `paso05_supervised_resultados.json`

---

### Paso 06 — `paso06_metrics.py` · **Martín**
Consolidación y comparación visual de todas las métricas generadas en el Paso 05.

| Qué hace | Detalle |
|---|---|
| Heatmap Accuracy | Todos los modelos × todas las tareas |
| Heatmap Precision/Recall | Precision Macro (azul) y Recall Macro (verde) |
| Promedio por modelo | Barras de accuracy y F1 promediado sobre las 4 tareas |

**Reports:** `paso06_metrics_heatmap_accuracy.png`, `paso06_metrics_precision_recall.png`, `paso06_metrics_promedio_modelos.png`, `paso06_metrics_consolidadas.json`

---

### Paso 07 — `paso07_comparison.py` · **Todos**
Comparación final con rankings, análisis cruzado clustering × supervisado.

| Qué hace | Detalle |
|---|---|
| Ranking por tarea | Modelos ordenados por accuracy para cada target |
| Modelos base | Comparación de RF vs DecTree vs LogReg |
| Targets por cluster | Proporciones de cada recomendación dentro de cada cluster |

**Reports:** `paso07_comparison_ranking_por_tarea.png`, `paso07_comparison_modelos_base.png`, `paso07_comparison_targets_por_cluster.png`, `paso07_comparison_ranking.json`

---

## Dataset — Columnas

### Features de entrada (7 crudas → 15 procesadas)

| Columna | Tipo | Descripción |
|---|---|---|
| `edad` | numérica | Edad del usuario (18–65) |
| `engagement_promedio` | numérica | Nivel de engagement general (0–1) |
| `valence_musical_pref` | numérica | Preferencia de positividad musical (0–1) |
| `energia_musical_pref` | numérica | Preferencia de energía musical (0–1) |
| `hora_lectura_preferida` | categórica | mañana / tarde / noche |
| `velocidad_lectura` | categórica | baja / media / alta |
| `contenido_visual_pref` | categórica | películas / series largas / cortas / documentales / anime |

### Targets de salida (4 × 5 clases)

| Columna | Clases |
|---|---|
| `genero_libro_rec` | thriller, romance, ciencia ficción, fantasía, no ficción |
| `tipo_vino_rec` | bajo en acidez, afrutado, seco, dulce, espumoso |
| `genero_musical_rec` | pop, rock, electrónica, jazz, reggaetón |
| `genero_serie_rec` | drama, comedia, acción, ciencia ficción, terror |

---

## Instalación y Uso

```bash
# Instalar dependencias
pip install -r new/requirements.txt

# Ejecutar el pipeline completo
cd scripts
python Main.py
```

Para activar el preprocesamiento (si necesitas regenerar `dataset_processed.csv`), edita `Main.py` y cambia:

```python
RUN_PREPROCESSING = True
```

---

## Dependencias

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.13.0
scikit-learn>=1.3.0
scipy>=1.11.0
```

---

## Equipo

| Nombre | Módulo(s) principal(es) | Paso(s) |
|---|---|---|
| **Agustín** | EDA y preprocesamiento | paso01, paso02 |
| **Nicolás** | Carga de datos | paso01 |
| **Camilo** | Clustering no supervisado | paso03 |
| **Isabella** | Visualización y dimensionalidad | paso04 |
| **Luis / Alejandro** | Modelos supervisados | paso05 |
| **Martín** | Métricas y consolidación | paso06 |
| **Todos** | Comparación final | paso07 |

---

## Configuración Avanzada

Todas las constantes del pipeline se centralizan en `scripts/config.py`:

| Variable | Default | Descripción |
|---|---|---|
| `RANDOM_STATE` | 42 | Semilla para reproducibilidad |
| `TEST_SIZE` | 0.2 | Proporción para test set |
| `N_CLUSTERS` | 5 | Número de clusters KMeans |
| `MAX_SAMPLE` | 50,000 | Máximo de filas para SVM y t-SNE (None = sin límite) |

---
