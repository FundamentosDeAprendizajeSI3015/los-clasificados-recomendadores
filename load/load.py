import pandas as pd

def load_data():
    """
    Carga el conjunto de datos principal del proyecto.
    
    Retorna:
        pd.DataFrame: El DataFrame que contiene los datos del archivo 'dataset.csv'.
    """
    file_path = "dataset.csv"
    data = pd.read_csv(file_path)

    return data