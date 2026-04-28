"""
Paquete src — Módulos reutilizables para el EDA de clasificación multi-output.

Uso desde cualquier directorio:
    import sys
    sys.path.insert(0, "/ruta/a/EDA-clasificadores")
    from src.config import *
    from src import eda_utils as eda
"""

from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMNS,
    ALL_FEATURES,
    ALL_COLUMNS,
    PROJECT_ROOT,
    DATA_DIR,
    DEFAULT_DATASET_PATH,
)
