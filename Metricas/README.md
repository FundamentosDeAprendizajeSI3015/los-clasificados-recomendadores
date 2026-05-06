# Métricas — Comparativa de Modelos Supervisados

Este módulo consolida y compara las métricas de evaluación de todos los modelos supervisados del proyecto. Los resultados se basan en el conjunto de prueba (20% del dataset, separado antes del entrenamiento).

## Contenido

```
Metricas/
├── collect_metrics.py          # Script de recopilación y comparativa
├── metricas_consolidadas.csv   # Tabla completa de métricas (generado)
├── metricas_consolidadas.json  # Misma tabla en formato JSON (generado)
└── README.md
```

## Uso

Ejecutar desde la raíz del proyecto:

```bash
python Metricas/collect_metrics.py
```

El script recorre automáticamente todas las carpetas de `Supervised/`, busca los archivos `evaluation.json` de cada modelo y target, y genera la comparativa. No requiere argumentos.

### Uso desde otro script

```python
from Metricas.collect_metrics import get_metrics_dataframe

df = get_metrics_dataframe()  # Lee metricas_consolidadas.csv si ya existe
```

## Modelos comparados

| Modelo | Descripción |
|---|---|
| Regresión Logística | Clasificador lineal con regularización L2 |
| Árbol de Decisión | Árbol CART con profundidad libre |
| RandomForest | Ensemble de 300 árboles |
| SVM (kernel_lineal) | SVC con kernel lineal |
| SVM (kernel_poly_d2) | SVC con kernel polinomial grado 2 |
| SVM (kernel_poly_d3) | SVC con kernel polinomial grado 3 |
| SVM (kernel_radial) | SVC con kernel RBF |

## Tareas de clasificación

Cada modelo predice 4 targets independientes:

| Target | Descripción |
|---|---|
| `genero_libro_rec` | Género literario recomendado |
| `genero_musical_rec` | Género musical recomendado |
| `genero_serie_rec` | Género de serie recomendado |
| `tipo_vino_rec` | Tipo de vino recomendado |

## Resultados (promedio sobre los 4 targets)

| Modelo | Accuracy | F1-Macro |
|---|---|---|
| **Árbol de Decisión** | **70.9%** | **70.7%** |
| SVM (kernel_radial) | 67.4% | 66.6% |
| SVM (kernel_poly_d2) | 67.1% | 66.7% |
| RandomForest | 66.4% | 65.0% |
| SVM (kernel_poly_d3) | 65.1% | 65.1% |
| Regresión Logística | 64.4% | 63.7% |
| SVM (kernel_lineal) | 64.4% | 63.5% |

### Por target (promedio sobre todos los modelos)

| Target | Accuracy | F1-Macro |
|---|---|---|
| `genero_libro_rec` | 70.4% | 70.2% |
| `genero_musical_rec` | 69.2% | 68.5% |
| `genero_serie_rec` | 68.4% | 67.7% |
| `tipo_vino_rec` | 58.1% | 57.0% |

`tipo_vino_rec` resulta el target más difícil de predecir en todos los modelos. `genero_libro_rec` es el más accesible.

## Métricas registradas

Por cada combinación modelo-target se registra:

- **Accuracy**: porcentaje de predicciones correctas sobre el test set
- **F1_Macro**: F1 promediado sin ponderar entre clases (penaliza desbalance)
- **Precision_Macro / Recall_Macro**: promedio macro de precisión y recall
- **Precision_Weighted / Recall_Weighted**: promedio ponderado por soporte de clase

## Cómo agregar un nuevo modelo

1. Crear la carpeta `Supervised/{NombreModelo}/reports/{target}/evaluation.json` con la estructura:

```json
{
  "target": "genero_libro_rec",
  "model": "NombreModelo",
  "accuracy": 0.72,
  "f1_macro": 0.71,
  "classification_report": { ... }
}
```

2. Volver a ejecutar `collect_metrics.py` — el nuevo modelo aparece automáticamente en la comparativa.

> Para modelos con variantes (como SVM con distintos kernels), colocar cada variante en `reports/{target}/{variante}/evaluation.json`. El colector las detecta automáticamente.
