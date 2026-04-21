# los-clasificados-recomendadores

## Objetivo

Este repositorio contiene la parte supervisada del proyecto de recomendadores, separada por carpetas de modelo:

- Regresión Logística
- Árbol de Decisión

Cada carpeta tiene 4 scripts, uno por target:

- genero_libro_rec
- tipo_vino_rec
- genero_musical_rec
- genero_serie_rec

Los targets se asumen generados por la etapa no supervisada (pseudo-etiquetas).

## Estructura actual

```text
los-clasificados-recomendadores/
├── README.md
├── Regresión Logística/
│   ├── 01_rl_genero_libro_rec.py
│   ├── 02_rl_tipo_vino_rec.py
│   ├── 03_rl_genero_musical_rec.py
│   ├── 04_rl_genero_serie_rec.py
│   └── resultados/
└── Árbol de Decisión/
	├── 01_arbol_genero_libro_rec.py
	├── 02_arbol_tipo_vino_rec.py
	├── 03_arbol_genero_musical_rec.py
	├── 04_arbol_genero_serie_rec.py
	└── resultados/
```

Nota: en este repo las carpetas estan nombradas con acentos.

## Requisitos

Python 3.10+ recomendado.

Instalacion de dependencias:

```bash
pip install pandas scikit-learn joblib
```

## Formato del dataset de entrada

Todos los scripts esperan un CSV con:

1. Features del usuario/item (numericas y/o categoricas)
2. La columna target correspondiente al script

Targets esperados en total:

- genero_libro_rec
- tipo_vino_rec
- genero_musical_rec
- genero_serie_rec

Puedes pasar la ruta del dataset con:

```bash
--data "ruta/al/dataset.csv"
```

Si no pasas --data, cada script intenta encontrar automaticamente un CSV que contenga su target.

## Ejecucion de scripts (Regresión Logística)

```bash
python "Regresión Logística/01_rl_genero_libro_rec.py" --data "data/processed/dataset_supervisado.csv"
python "Regresión Logística/02_rl_tipo_vino_rec.py" --data "data/processed/dataset_supervisado.csv"
python "Regresión Logística/03_rl_genero_musical_rec.py" --data "data/processed/dataset_supervisado.csv"
python "Regresión Logística/04_rl_genero_serie_rec.py" --data "data/processed/dataset_supervisado.csv"
```

## Ejecucion de scripts (Árbol de Decisión)

```bash
python "Árbol de Decisión/01_arbol_genero_libro_rec.py" --data "data/processed/dataset_supervisado.csv"
python "Árbol de Decisión/02_arbol_tipo_vino_rec.py" --data "data/processed/dataset_supervisado.csv"
python "Árbol de Decisión/03_arbol_genero_musical_rec.py" --data "data/processed/dataset_supervisado.csv"
python "Árbol de Decisión/04_arbol_genero_serie_rec.py" --data "data/processed/dataset_supervisado.csv"
```

## Que genera cada script

Cada script guarda 3 archivos en su carpeta resultados:

1. Modelo entrenado (.joblib)
2. Metricas (.json)
3. Predicciones de test (.csv)

Ejemplo de salida (nombre base por script):

- 01_rl_genero_libro_rec_model.joblib
- 01_rl_genero_libro_rec_metrics.json
- 01_rl_genero_libro_rec_predicciones.csv

## Metricas incluidas

En el archivo .json se guarda:

- accuracy
- precision_macro
- recall_macro
- f1_macro
- confusion_matrix
- labels_confusion_matrix

## Parametros disponibles por script

- --data: ruta al CSV
- --test-size: proporcion del set de prueba (default 0.2)
- --random-state: semilla (default 42)

Ejemplo:

```bash
python "Regresión Logística/01_rl_genero_libro_rec.py" --data "data/processed/dataset_supervisado.csv" --test-size 0.25 --random-state 7
```

## Flujo recomendado del proyecto

1. Ejecutar no supervisado para generar pseudo-labels (targets)
2. Exportar CSV final con features + targets
3. Ejecutar los 8 scripts supervisados
4. Comparar resultados entre Regresión Logística y Árbol de Decisión por target

## Notas

- Si una clase tiene muy pocos ejemplos, el script puede desactivar estratificacion automaticamente.
- Si no encuentra dataset, el script muestra error y pide usar --data.