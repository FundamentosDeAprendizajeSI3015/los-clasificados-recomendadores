# Supervised

Este módulo contiene la parte de aprendizaje supervisado del proyecto.

## Dataset final esperado

Ejemplo concreto de cómo quedaría una fila:

| Variable usuario | Valor |
| --- | --- |
| `edad` | 28 |
| `hora_lectura_preferida` | noche |
| `velocidad_lectura` | alta |
| `engagement_promedio` | 0.75 |
| `valence_musical_pref` | 0.8 |
| `energia_musical_pref` | 0.7 |
| `contenido_visual_pref` | series largas |
| `genero_libro_rec` (target) | thriller |
| `tipo_vino_rec` (target) | bajo en acidez, alta graduacion |
| `genero_musical_rec` (target) | pop energetico |
| `genero_serie_rec` (target) | drama |

### Variables de entrada (features)

- `edad`
- `hora_lectura_preferida`
- `velocidad_lectura`
- `engagement_promedio`
- `valence_musical_pref`
- `energia_musical_pref`
- `contenido_visual_pref`

### Targets que se predicen

- `genero_libro_rec`
- `tipo_vino_rec`
- `genero_musical_rec`
- `genero_serie_rec`

## Algoritmos implementados

### Árbol de Decisión

El script general de Árbol de Decisión está en `Árbol de Decisión/scr/script.py` y unifica entrenamiento + evaluación para los 4 targets en una sola corrida.

Detalles principales:

- Modelo: `DecisionTreeClassifier`
- Preprocesamiento:
  - variables numéricas con `SimpleImputer(median)`
  - variables categóricas con `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")`
- Entrenamiento por target con `train_test_split` estratificado
- Métricas: accuracy, f1 macro y classification report
- Salidas por target en `reports/<target>/`

### Regresión Logística

El script general de Regresión Logística está en `Regresión Logística/scr/script.py` y también ejecuta entrenamiento + evaluación para los 4 targets desde un solo archivo.

Detalles principales:

- Modelo: `LogisticRegression`
- Preprocesamiento:
  - variables numéricas con `SimpleImputer(median)` + `StandardScaler`
  - variables categóricas con `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")`
- Entrenamiento por target con `train_test_split` estratificado
- Métricas: accuracy, f1 macro y classification report
- Salidas por target en `reports/<target>/`

## Organización de carpetas

Se dejó un solo script por algoritmo en `scr/` (sin dividir en train/evaluate por target), y se mantiene `reports/` separado por target para conservar trazabilidad de resultados.

```text
Supervised/
	README.md
	Árbol de Decisión/
		scr/
			script.py
		reports/
			genero_libro_rec/
			genero_musical_rec/
			genero_serie_rec/
			tipo_vino_rec/
	Regresión Logística/
		scr/
			script.py
		reports/
			genero_libro_rec/
			genero_musical_rec/
			genero_serie_rec/
			tipo_vino_rec/
```

## Uso rápido

Ejecutar Árbol de Decisión (train + evaluate):

```bash
python "Árbol de Decisión/scr/script.py" --data <ruta_csv> --task both
```

Ejecutar Regresión Logística (train + evaluate):

```bash
python "Regresión Logística/scr/script.py" --data <ruta_csv> --task both
```

Opciones útiles:

- `--task train` solo entrena
- `--task evaluate` solo evalúa/predice usando modelos ya guardados
- `--eval-data <ruta_csv>` para evaluar/predicir con otro dataset