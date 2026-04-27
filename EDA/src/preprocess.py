"""
Módulo de preprocesamiento de datos.

Este script carga el dataset original, estandariza las variables numéricas,
codifica las variables categóricas (features y targets), guarda los objetos
transformadores para su uso posterior y exporta el dataset preprocesado.
"""

import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

import sys
sys.path.insert(0, os.path.abspath('.'))

# Configuración centralizada
from src.config import (
    DEFAULT_DATASET_PATH,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMNS,
    PROJECT_ROOT
)

def preprocess_pipeline():
    print(f"Iniciando pipeline de preprocesamiento sobre {DEFAULT_DATASET_PATH}...")
    
    # 1. Cargar datos
    if not os.path.exists(DEFAULT_DATASET_PATH):
        raise FileNotFoundError(f"No se encontró el archivo: {DEFAULT_DATASET_PATH}")
        
    df = pd.read_csv(DEFAULT_DATASET_PATH)
    print(f"Dataset cargado exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas.")
    
    # 2. Crear directorio de transformadores
    transformers_dir = os.path.join(PROJECT_ROOT, "transformers")
    os.makedirs(transformers_dir, exist_ok=True)
    
    # Preparar DataFrames para resultados
    df_processed = pd.DataFrame(index=df.index)
    
    # 3. Procesar numéricas (StandardScaler)
    print("Escalando variables numéricas...")
    scaler = StandardScaler()
    scaled_nums = scaler.fit_transform(df[NUMERICAL_FEATURES])
    df_scaled_nums = pd.DataFrame(scaled_nums, columns=NUMERICAL_FEATURES, index=df.index)
    df_processed = pd.concat([df_processed, df_scaled_nums], axis=1)
    
    # Guardar scaler
    scaler_path = os.path.join(transformers_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler guardado en: {scaler_path}")
    
    # 4. Procesar categóricas - features (OneHotEncoder)
    print("Codificando variables categóricas (Features)...")
    # sparse_output=False en versiones modernas (sparse=False en versiones antiguas de scikit-learn)
    try:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
    encoded_cats = ohe.fit_transform(df[CATEGORICAL_FEATURES])
    feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
    df_encoded_cats = pd.DataFrame(encoded_cats, columns=feature_names, index=df.index)
    df_processed = pd.concat([df_processed, df_encoded_cats], axis=1)
    
    # Guardar OHE
    ohe_path = os.path.join(transformers_dir, "features_encoder.pkl")
    joblib.dump(ohe, ohe_path)
    print(f"OneHotEncoder guardado en: {ohe_path}")
    
    # 5. Procesar targets (LabelEncoder por cada target)
    print("Codificando variables objetivo (Targets)...")
    target_encoders = {}
    
    for target in TARGET_COLUMNS:
        le = LabelEncoder()
        df_processed[target] = le.fit_transform(df[target])
        target_encoders[target] = le
        print(f"   - {target}: mapeado a {len(le.classes_)} clases.")
        
    # Guardar diccionario de LabelEncoders
    le_path = os.path.join(transformers_dir, "target_encoders.dict.pkl")
    joblib.dump(target_encoders, le_path)
    print(f"Target Encoders guardados en: {le_path}")
    
    # 6. Exportar dataset procesado
    processed_dataset_path = os.path.join(PROJECT_ROOT, "data", "dataset_processed.csv")
    df_processed.to_csv(processed_dataset_path, index=False)
    print(f"Dataset procesado y listo para modelado exportado en: {processed_dataset_path}")
    print(f"   Dimensiones finales: {df_processed.shape[0]} filas, {df_processed.shape[1]} columnas.")

if __name__ == "__main__":
    preprocess_pipeline()
