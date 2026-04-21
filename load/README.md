# Data Loader

Módulo encargado de cargar el dataset generado para el sistema de recomendación.

---

### Estructura

```
load/
├── README.md           # Este archivo
├── create_dataset.py   # Script de generación del dataset sintético
├── load.py             # Script de carga del dataset
├── dataset.csv         # Dataset principal (1 000 filas, balanceado)
└── example.csv         # Ejemplo con una sola fila de referencia
```

### Prerequisitos

```bash
pip install pandas
```

---

### Funcionalidad

La función `load_data()` en `load.py` realiza lo siguiente:

1. Lee el archivo `dataset.csv` ubicado en esta misma carpeta.
2. Retorna un `pandas.DataFrame` con las 1 000 filas del dataset.

### Uso

```python
from load import load_data

# Cargar el dataset
df = load_data()

# Ver las primeras filas
print(df.head())
```

---

### Columnas del Dataset

| Columna | Tipo | Descripción |
|---|---|---|
| `edad` | int | Edad del individuo (18–65) |
| `hora_lectura_preferida` | str | Hora preferida de lectura (`manana`, `tarde`, `noche`) |
| `velocidad_lectura` | str | Velocidad de lectura (`baja`, `media`, `alta`) |
| `engagement_promedio` | float | Nivel promedio de engagement (0.0–1.0) |
| `valence_musical_pref` | float | Valencia musical preferida (0.0–1.0) |
| `energia_musical_pref` | float | Energía musical preferida (0.0–1.0) |
| `contenido_visual_pref` | str | Contenido visual preferido (`peliculas`, `series largas`, `series cortas`, `documentales`, `anime`) |
| **`genero_libro_rec`** | str | **Target** — Género de libro recomendado (5 clases) |
| **`tipo_vino_rec`** | str | **Target** — Tipo de vino recomendado (5 clases) |
| **`genero_musical_rec`** | str | **Target** — Género musical recomendado (5 clases) |
| **`genero_serie_rec`** | str | **Target** — Género de serie recomendado (5 clases) |

---

### Generación del Dataset

El archivo `create_dataset.py` genera el dataset sintético de 1 000 filas con clases balanceadas (200 filas por clase en cada variable objetivo). Para regenerar el dataset:

```bash
python create_dataset.py
```

Esto sobrescribirá `dataset.csv` en esta misma carpeta.

> **Nota:** El script usa `seed=42` para reproducibilidad. Cada ejecución produce exactamente el mismo dataset.