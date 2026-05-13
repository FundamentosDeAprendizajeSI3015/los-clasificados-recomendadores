
# Módulo de Clustering — Unsupervised

## ¿Qué hace este módulo?

Toma el dataset de usuarios preprocesado por el EDA y ejecuta **4 KMeans independientes**, uno por cada dominio de recomendación. El resultado es un diccionario listo para ser consumido directamente por el módulo de entrenamiento supervisado.

No guarda nada en disco — todo opera en memoria y se pasa entre módulos por importación directa.

---

## Lugar en el pipeline

```
EDA/data/dataset_processed.csv
        ↓
unsupervised/clustering.py   ← este módulo
```

---

## Estructura de carpetas esperada

```
los-clasificados-recomendadores/
├── EDA/
│   └── data/
│       └── dataset_processed.csv    ← entrada
└── unsupervised/
    ├── clustering.py
    └── README.md
```

La ruta al CSV se resuelve automáticamente de forma relativa a `clustering.py`, así que no hay que configurar nada de rutas.

---

## ¿Qué hay en el dataset de entrada?

El CSV tiene **1000 filas y 19 columnas**: 15 features de usuario y 4 columnas target ya definidas por el EDA.

**Features de usuario (entrada del clustering y de los modelos):**

| Columna | Descripción |
|---|---|
| `edad` | Edad del usuario (escalada) |
| `engagement_promedio` | Nivel de engagement general |
| `valence_musical_pref` | Preferencia de positividad musical |
| `energia_musical_pref` | Preferencia de energía musical |
| `hora_lectura_preferida_*` | Dummies: mañana / tarde / noche |
| `velocidad_lectura_*` | Dummies: alta / media / baja |
| `contenido_visual_pref_*` | Dummies: anime / documentales / películas / series cortas / series largas |

**Targets (salida — ya vienen en el CSV desde el EDA):**

| Columna | Descripción | Clases |
|---|---|---|
| `genero_libro_rec` | Género de libro recomendado | 0–4 |
| `tipo_vino_rec` | Tipo de vino recomendado | 0–4 |
| `genero_musical_rec` | Género musical recomendado | 0–4 |
| `genero_serie_rec` | Género de serie recomendado | 0–4 |

---

## ¿Qué hace el clustering exactamente?

Los targets ya existen en el CSV — el EDA los construyó usando información de los 4 datasets originales (libros, vinos, música, Netflix). El clustering **no los reemplaza**, sino que corre en paralelo sobre subconjuntos de features para generar etiquetas KMeans adicionales (`kmeans_libro`, `kmeans_vino`, `kmeans_musica`, `kmeans_serie`) útiles para análisis exploratorio o como feature extra.

Cada KMeans usa las features más relevantes para su dominio:

| Dominio | Features usadas |
|---|---|
| Libro | edad, engagement, velocidad de lectura, hora de lectura |
| Vino | edad, engagement, valence musical, energía musical |
| Música | valence, energía, edad, hora de lectura |
| Serie | preferencias de contenido visual, edad, engagement |

**K=5** porque los targets originales tienen exactamente 5 clases. No se aplica un scaler adicional porque el CSV ya viene escalado desde el EDA.

---

## Lo que retorna `ejecutar_clustering()`

```python
resultado = ejecutar_clustering()
```

| Clave | Tipo | Contenido |
|---|---|---|
| `resultado["X"]` | `pd.DataFrame` (1000×15) | Features de usuario — entrada para los 4 modelos supervisados |
| `resultado["y_libro"]` | `pd.Series` | Target libro del EDA (0–4) |
| `resultado["y_vino"]` | `pd.Series` | Target vino del EDA (0–4) |
| `resultado["y_musica"]` | `pd.Series` | Target música del EDA (0–4) |
| `resultado["y_serie"]` | `pd.Series` | Target serie del EDA (0–4) |
| `resultado["modelos"]` | `dict` | KMeans entrenado por dominio — para predecir cluster en usuarios nuevos |
| `resultado["metricas"]` | `dict` | Silhouette score por dominio |
| `resultado["df_completo"]` | `pd.DataFrame` (1000×23) | df original + 4 columnas `kmeans_*` |
| `resultado["cluster_labels"]` | `pd.DataFrame` (1000×4) | Solo las 4 columnas `kmeans_*` |

---

## Cómo usarlo desde el módulo de entrenamiento

```python
# modelos/entrenamiento.py
from unsupervised.clustering import ejecutar_clustering
from sklearn.model_selection import train_test_split

resultado = ejecutar_clustering()

X        = resultado["X"]
y_libro  = resultado["y_libro"]
y_vino   = resultado["y_vino"]
y_musica = resultado["y_musica"]
y_serie  = resultado["y_serie"]

# Entrenar un modelo por cada target
X_train, X_test, y_train, y_test = train_test_split(X, y_libro,  test_size=0.2, random_state=42)
# modelo_libro.fit(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(X, y_vino,   test_size=0.2, random_state=42)
# modelo_vino.fit(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(X, y_musica, test_size=0.2, random_state=42)
# modelo_musica.fit(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(X, y_serie,  test_size=0.2, random_state=42)
# modelo_serie.fit(X_train, y_train)
```

El clustering corre cada vez que se importa — con 1000 filas y KMeans es inmediato. Si el dataset crece significativamente en el futuro, se puede considerar guardar `df_completo` en disco para no recalcular.

---

## Funciones disponibles

### `ejecutar_clustering(df=None)`
Función principal. Si `df` es `None`, carga el CSV automáticamente.

### `metodo_del_codo(df, dominio, max_k=12)`
Grafica la inercia para distintos valores de K en un dominio dado. Útil para validar o ajustar K. Guarda la imagen como `codo_{dominio}.png` en la carpeta `unsupervised/`.

```python
from unsupervised.clustering import metodo_del_codo, cargar_datos

df = cargar_datos()
metodo_del_codo(df, "libro")    # genera codo_libro.png
metodo_del_codo(df, "musica")   # genera codo_musica.png
```

### `cargar_datos()`
Carga y retorna el CSV desde `EDA/data/dataset_processed.csv`. Se puede usar de forma independiente si solo se necesitan los datos.

---

## Métricas de calidad del clustering

El silhouette score mide qué tan bien separados están los clusters (rango de -1 a 1):

| Rango | Interpretación |
|---|---|
| > 0.5 | Clusters bien definidos |
| 0.2 – 0.5 | Clusters aceptables |
| < 0.2 | Considerar ajustar K o features |

Con el dataset actual los scores están entre **0.18 y 0.21**, lo cual es esperado dado que el dataset es sintético y las features de usuario no tienen separación natural muy marcada. Esto no afecta el entrenamiento supervisado, que usa los targets del EDA directamente.

---

## Dependencias

```
pandas
numpy
scikit-learn
matplotlib
```
