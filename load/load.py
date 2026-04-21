import pandas as pd

def load_data():
    file_path = "example.csv"
    data = pd.read_csv(file_path)

    return data