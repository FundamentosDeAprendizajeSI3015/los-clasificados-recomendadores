"""
Módulo de utilidades para el Análisis Exploratorio de Datos (EDA).
Contiene funciones auxiliares para validación, visualización y análisis estadístico.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from .config import (
    EXPECTED_DTYPES,
    display_name,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMNS
)

def check_dataframe_schema(df: pd.DataFrame) -> bool:
    """
    Verifica que el DataFrame contenga las columnas esperadas y retorna advertencias
    si faltan columnas o los tipos son incorrectos.
    """
    expected_cols = set(EXPECTED_DTYPES.keys())
    actual_cols = set(df.columns)
    
    missing_cols = expected_cols - actual_cols
    extra_cols = actual_cols - expected_cols
    
    is_valid = True
    
    if missing_cols:
        print(f"⚠️ Faltan las siguientes columnas esperadas: {missing_cols}")
        is_valid = False
    else:
        print("✅ Todas las columnas esperadas están presentes.")
        
    if extra_cols:
        print(f"ℹ️ Se encontraron columnas adicionales que no serán procesadas por defecto: {extra_cols}")
        
    for col in expected_cols.intersection(actual_cols):
        expected_type = EXPECTED_DTYPES[col]
        actual_type = str(df[col].dtype)
        
        # Validación de tipos básica
        if expected_type == 'int' and not actual_type.startswith('int'):
            print(f"⚠️ La columna '{col}' debería ser de tipo entero, pero es '{actual_type}'")
            is_valid = False
        elif expected_type == 'float' and not actual_type.startswith('float'):
            print(f"⚠️ La columna '{col}' debería ser de tipo numérico flotante, pero es '{actual_type}'")
            is_valid = False
        elif expected_type == 'object' and actual_type not in ('object', 'category', 'string', 'str') and not actual_type.startswith('string'):
            print(f"⚠️ La columna '{col}' debería ser de tipo categórico/texto, pero es '{actual_type}'")
            is_valid = False
            
    if is_valid:
        print("✅ Los tipos de datos parecen correctos en base a las columnas requeridas.")
        
    return is_valid


def plot_missing_values(df: pd.DataFrame, figsize=(10, 6)):
    """
    Grafica el porcentaje de valores nulos por columna usando Seaborn.
    """
    missing_pct = df.isnull().mean() * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    
    if len(missing_pct) == 0:
        print("✅ No se encontraron valores faltantes en el dataset.")
        return
        
    plt.figure(figsize=figsize)
    sns.barplot(x=missing_pct.values, y=[display_name(c) for c in missing_pct.index], palette="viridis")
    plt.title("Porcentaje de valores faltantes por columna")
    plt.xlabel("Porcentaje (%)")
    plt.ylabel("Columnas")
    plt.xlim(0, 100)
    
    for i, v in enumerate(missing_pct.values):
        plt.text(v + 1, i, f"{v:.1f}%", va='center')
        
    plt.tight_layout()
    plt.show()


def cramers_v(confusion_matrix: pd.DataFrame) -> float:
    """
    Calcula el estadístico V de Cramér para medir la asociación entre dos variables categóricas.
    """
    chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    if n == 0 or min_dim == 0:
        return 0.0
    return np.sqrt((chi2 / n) / min_dim)
