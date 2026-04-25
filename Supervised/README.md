# Supervised

Este modulo contiene la parte de aprendizaje supervisado del proyecto.

## Dataset final esperado

Ejemplo concreto de como quedaria una fila:

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

### RandomForest

El script general de RandomForest esta en `RandomForest/scr/script.py` y unifica entrenamiento + evaluacion para los 4 targets en una sola corrida.

Detalles principales:

- Modelo: `RandomForestClassifier`
- Preprocesamiento:
- variables numericas en passthrough
- variables categoricas con `OneHotEncoder(handle_unknown="ignore")`
- Entrenamiento por target con `train_test_split` estratificado
- Metricas: accuracy, f1 macro y classification report
- Salidas por target en `reports/<target>/`

### SVM

El script general de SVM esta en `SVM/scr/script.py` y tambien ejecuta entrenamiento + evaluacion para los 4 targets desde un solo archivo.

Detalles principales:

- Modelo: `SVC` con kernel RBF
- Preprocesamiento:
- variables numericas con `StandardScaler`
- variables categoricas con `OneHotEncoder(handle_unknown="ignore")`
- Entrenamiento por target con `train_test_split` estratificado
- Metricas: accuracy, f1 macro y classification report
- Salidas por target en `reports/<target>/`

## Organizacion de carpetas

Se dejo un solo script por algoritmo en `scr/` (sin dividir en train/evaluate por target), y se mantiene `reports/` separado por target para conservar trazabilidad de resultados.

```text
Supervised/
	README.md
	RandomForest/
		scr/
			script.py
		reports/
			genero_libro_rec/
			genero_musical_rec/
			genero_serie_rec/
			tipo_vino_rec/
	SVM/
		scr/
			script.py
		reports/
			genero_libro_rec/
			genero_musical_rec/
			genero_serie_rec/
			tipo_vino_rec/
```

## Uso rapido

Ejecutar RandomForest (train + evaluate):

```bash
python Supervised/RandomForest/scr/script.py --data <ruta_csv> --task both
```

Ejecutar SVM (train + evaluate):

```bash
python Supervised/SVM/scr/script.py --data <ruta_csv> --task both
```

Opciones utiles:

- `--task train` solo entrena
- `--task evaluate` solo evalua/predice usando modelos ya guardados
- `--eval-data <ruta_csv>` para evaluar/predicir con otro dataset
