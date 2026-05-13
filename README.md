# 🎯 Los Clasificados Recomendadores

Sistema de recomendación multi-dominio basado en aprendizaje automático que, dado el perfil de un usuario, recomienda simultáneamente un **género de libro**, un **tipo de vino**, un **género musical** y un **género de serie/película**.

---

## ¿Qué hace el proyecto?

El sistema toma variables de perfil de usuario (edad, engagement, preferencias de contenido, preferencias musicales, hábitos de lectura) y produce 4 recomendaciones en paralelo, una por dominio. Cada recomendación es una clasificación en 5 categorías posibles.

El pipeline completo combina:
- **Aprendizaje no supervisado** (KMeans) para analizar y validar la estructura de los datos
- **Aprendizaje supervisado** (Random Forest y SVM) para entrenamiento y predicción de los 4 targets
- **Análisis exploratorio** para construcción y preprocesamiento del dataset de usuarios
- **Visualización de métricas y resultados** para evaluación de los modelos

---

## Datasets de origen

El dataset de usuarios fue construido sintetizando información de 4 fuentes independientes:

| Dominio | Dataset original | Variable target generada |
|---|---|---|
| 📚 Libros | Sesiones de lectura con métricas de abandono y engagement | `genero_libro_rec` |
| 🍷 Vinos | Características físico-químicas y calidad | `tipo_vino_rec` |
| 🎵 Música | Atributos de audio por región y contexto socioeconómico | `genero_musical_rec` |
| 🎬 Series/Películas | Catálogo Netflix con scores IMDB y clasificaciones | `genero_serie_rec` |

El dataset final procesado tiene **1000 usuarios × 19 columnas** (15 features + 4 targets), con targets de 5 clases cada uno (0–4).

---

## Estructura del repositorio

```
los-clasificados-recomendadores/
│
├── EDA/                          # Análisis exploratorio y construcción del dataset
│   ├── data/
│   │   ├── dataset.csv           # Dataset crudo
│   │   └── dataset_processed.csv # Dataset limpio y escalado (entrada al pipeline)
│   ├── src/                      # Scripts de procesamiento
│   ├── transformers/             # Transformadores personalizados
│   ├── eda_clasificadores.ipynb  # Notebook principal del EDA
│   ├── example.csv
│   └── README.md
│
├── load/                         # Carga y validación inicial de los datos
│
├── unsupervised/                 # Clustering KMeans (aprendizaje no supervisado)
│   ├── clustering.py             # Módulo principal de clustering
│   └── README.md
│
├── modelos/                      # Entrenamiento supervisado (en desarrollo)
│   ├── random_forest/           # Modelos Random Forest por dominio
|   ├── svm/                      # Modelos SVM por dominio
│   ├── Regresion Logistica/        # Modelos Regresión logistica por dominio
│   └── Árbol de decisión/          # Modelos Árboles de decisión por dominio     
│
├── visualizacion/                # Gráficas y visualizaciones
│
├── metricas/                # Métricas y resultados
│
├── README.md                     # Este archivo
└── requirements.txt
```

---

## Pipeline completo

```
dataset.csv
    ↓
[ load/ ]
Carga y validación de datos crudos
    ↓
[ EDA/ ]
Análisis exploratorio, limpieza, escalado,
codificación y construcción de targets
    ↓
EDA/data/dataset_processed.csv
    ↓
[ unsupervised/clustering.py ]
4 KMeans independientes (K=5) por dominio
Validación de estructura de los datos
    ↓
[ modelos/ ]
4 modelos supervisados (Random Forest + SVM)
uno por cada target
    ↓
[ visualizacion/ ]
Gráficas, matrices de confusión, etc
    ↓
[ metricas/ ]
Métricas y resultados

```

---

## Módulos

### `EDA/`
Análisis exploratorio completo del dataset. Limpia los datos crudos, aplica escalado, genera dummies para variables categóricas y construye los 4 targets de recomendación a partir de los datasets originales. Entrega `dataset_processed.csv` como salida para el resto del pipeline.

### `load/`
Carga y validación inicial de los archivos de datos. Verifica integridad antes de pasar al EDA.

### `unsupervised/`
Ejecuta 4 KMeans independientes sobre el dataset procesado, uno por dominio. No reemplaza los targets del EDA sino que genera etiquetas adicionales para análisis exploratorio y valida la estructura de los datos. Retorna en memoria todo lo necesario para el entrenamiento supervisado. Ver `unsupervised/README.md` para detalle completo.

### `modelos/`
Entrena 4 modelos de clasificación supervisada usando los datos retornados por el módulo de clustering. Se implementan cuatro algoritmos por dominio:
- **Random Forest**: robusto ante variables categóricas codificadas y datasets de tamaño moderado
- **SVM**: efectivo en espacios de alta dimensión con clases bien separadas
- **Árbol de Decisión**: interpretable y eficiente en datasets pequeños, permite visualizar directamente las reglas de decisión aprendidas por el modelo
- **Regresión logística**: establece una línea base de rendimiento al ser el modelo más simple del conjunto, útil para comparar qué tanto aportan los modelos más complejos

Cada algoritmo genera 4 modelos independientes (uno por target), para un total de 8 modelos entrenados.

### `visualizacion/` 
Genera las métricas de evaluación y visualizaciones de resultados:
- Matrices de confusión por modelo y dominio
- Importancia de features (Random Forest)
- Curvas de aprendizaje
  
### `metricas/` 
- Comparación de accuracy entre Random Forest, SVM, Árboles de Decisión y Regresión logística
---

## Dataset procesado — columnas

**Features de entrada (15):**

| Columna | Descripción |
|---|---|
| `edad` | Edad del usuario (escalada) |
| `engagement_promedio` | Nivel de engagement general del usuario |
| `valence_musical_pref` | Preferencia de positividad musical (0–1) |
| `energia_musical_pref` | Preferencia de energía musical (0–1) |
| `hora_lectura_preferida_manana` | Dummy: consume contenido en la mañana |
| `hora_lectura_preferida_tarde` | Dummy: consume contenido en la tarde |
| `hora_lectura_preferida_noche` | Dummy: consume contenido en la noche |
| `velocidad_lectura_alta` | Dummy: lector rápido |
| `velocidad_lectura_media` | Dummy: lector de ritmo medio |
| `velocidad_lectura_baja` | Dummy: lector lento |
| `contenido_visual_pref_anime` | Dummy: preferencia por anime |
| `contenido_visual_pref_documentales` | Dummy: preferencia por documentales |
| `contenido_visual_pref_peliculas` | Dummy: preferencia por películas |
| `contenido_visual_pref_series cortas` | Dummy: preferencia por series cortas |
| `contenido_visual_pref_series largas` | Dummy: preferencia por series largas |

**Targets de salida (4 × 5 clases):**

| Columna | Descripción |
|---|---|
| `genero_libro_rec` | Género de libro recomendado (0–4) |
| `tipo_vino_rec` | Tipo de vino recomendado (0–4) |
| `genero_musical_rec` | Género musical recomendado (0–4) |
| `genero_serie_rec` | Género de serie/película recomendado (0–4) |

---

## Equipo

<!-- Reemplazar con nombres y roles reales -->
| Nombre | Rol |
|---|---|
| Agustin | EDA y preprocesamiento |
| Nicolas | Carga de datos (load) |
| Camilo | Clustering no supervisado |
| Luis / Alejandro | Modelos supervisados |
| Isabella | Visualización |
| Martin | Métricas |

---

## Instalación y uso

```bash
# Clonar el repositorio
git clone https://github.com/FundamentosDeAprendizajeSI3015/los-clasificados-recomendadores.git
cd los-clasificados-recomendadores
git checkout dev

# Instalar dependencias
pip install -r requirements.txt
```

---

## Dependencias principales

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Ver `requirements.txt` para la lista completa con versiones.

---

