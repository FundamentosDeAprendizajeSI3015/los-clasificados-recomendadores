# Análisis Exploratorio de Datos (EDA) y Preprocesamiento — Clasificación Multi-Output

Este repositorio contiene un análisis exploratorio de datos (EDA) profesional y un **pipeline de preprocesamiento estandarizado** para un problema de **clasificación multi-output**. El objetivo es entender las relaciones entre variables demográficas/preferencias (features) y recomendaciones de distintos dominios (targets), y dejar los datos listos para el entrenamiento de modelos de Machine Learning.

## Estructura del Proyecto

```
EDA-clasificadores/
├── data/
│   ├── dataset.csv              ← Ubicación esperada para tu dataset original
│   └── dataset_processed.csv    ← (Generado) Dataset preprocesado listo para modelos
├── src/
│   ├── __init__.py
│   ├── config.py                ← Variables de configuración (rutas, features, targets)
│   ├── eda_utils.py             ← Funciones auxiliares para el EDA
│   └── preprocess.py            ← Pipeline de estandarización y codificación
├── transformers/                ← (Generado) Carpeta con los objetos de transformación
│   ├── scaler.pkl               ← StandardScaler entrenado en features numéricos
│   ├── features_encoder.pkl     ← OneHotEncoder entrenado en features categóricos
│   └── target_encoders.dict.pkl ← Diccionario de LabelEncoders (uno por cada target)
├── eda_clasificadores.ipynb     ← Notebook interactivo principal con el análisis (EDA)
├── requirements.txt             ← Dependencias del proyecto
└── README.md                    ← Este archivo
```

## Requisitos Previos

Asegúrate de tener Python 3.9 o superior instalado. Se recomienda usar un entorno virtual (venv o conda).

### Instalación de Dependencias

Activa tu entorno virtual y ejecuta el siguiente comando en la raíz del proyecto:

```bash
pip install -r requirements.txt
```

---

## 🚀 Guía para el Siguiente Rol (Data Scientist / ML Engineer)

Se ha completado la etapa de Análisis Exploratorio y Preprocesamiento. Si tu trabajo es **entrenar los modelos de clasificación Multi-Output**, sigue estas instrucciones:

### 1. El Dataset Procesado
Ya no necesitas hacer limpieza ni codificación. Puedes usar directamente el archivo `data/dataset_processed.csv`.
- Todas las **variables numéricas** están estandarizadas (media 0, varianza 1).
- Todas las **variables categóricas de entrada (features)** han pasado por un One-Hot Encoding (OHE).
- Las **4 variables objetivo (targets)** se han codificado usando `LabelEncoder` (contienen enteros de `0` a `n-1`).

### 2. Uso de Transformadores (Inferencia y Nuevos Datos)
Para garantizar que los nuevos datos o datos en producción pasen exactamente por el mismo procesamiento, los transformadores han sido serializados usando `joblib`. 
Puedes cargarlos en tus scripts de modelado así:

```python
import joblib

# 1. Cargar el Scaler de numéricas
scaler = joblib.load('transformers/scaler.pkl')
# scaler.transform(nuevos_datos_numericos)

# 2. Cargar el Codificador de categóricas
ohe = joblib.load('transformers/features_encoder.pkl')
# ohe.transform(nuevos_datos_categoricos)

# 3. Cargar los codificadores de los Targets
target_encoders = joblib.load('transformers/target_encoders.dict.pkl')
# Para recuperar el nombre original de la predicción de género de libros:
# genero_libro_original = target_encoders['genero_libro_rec'].inverse_transform(predicciones_libro)
```

---

## Uso General (Re-ejecución)

Si hay un **nuevo dataset**, sigue estos pasos para re-ejecutar todo:

1. **Prepara tu dataset:**
   - Coloca tu nuevo archivo en `data/dataset.csv`.
   - Asegúrate de que las columnas coincidan con las descritas en `src/config.py`.

2. **Preprocesa los datos:**
   Ejecuta el pipeline de preprocesamiento para volver a generar los transformadores y el archivo procesado.
   ```bash
   python src/preprocess.py
   ```

3. **Ejecuta el Notebook de EDA:**
   Abre y corre todas las celdas de `eda_clasificadores.ipynb` en Jupyter Notebook o VSCode para actualizar el análisis interactivo.
   ```bash
   jupyter notebook
   ```

## Secciones del Análisis (EDA Notebook)

1.  **Configuración y carga de datos:** Validación automática de esquema y tipos.
2.  **Inspección inicial:** Estadísticas descriptivas y completitud.
3.  **Análisis de valores faltantes:** Detección de patrones y heatmaps de nulidad.
4.  **Análisis univariado:** Distribuciones de cada variable individual (histogramas y barplots).
5.  **Análisis bivariado:** Relaciones entre features y targets.
6.  **Análisis de correlación:** Matrices de correlación (numéricas) y asociación V de Cramér (categóricas).
7.  **Visualizaciones 3D:** Scatters interactivos usando Plotly.
8.  **PCA (Análisis de Componentes Principales):** Evaluación lineal de dimensionalidad.
9.  **UMAP:** Evaluación no lineal de dimensionalidad para clusters.
10. **Análisis multi-target:** Exploración de co-ocurrencia entre las variables objetivo.
11. **Conclusiones y recomendaciones:** Resumen ejecutivo de los hallazgos.

---

**Autor:** Agustín Figueroa Sierra
