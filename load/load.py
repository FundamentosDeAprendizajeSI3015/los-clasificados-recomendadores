import os
import pandas as pd
from dotenv import load_dotenv()

load_dotenv()
file_path = os.getenv("FILE_PATH")
file_type = os.getenv("FILE_TYPE")

def load_data():
    if file_type == 'csv':
        data = pd.read_csv(file_path)
    if file_type == 'json':
        data = pd.read_json(file_path)

    return data