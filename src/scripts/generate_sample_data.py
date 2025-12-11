import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_files():
    """Создание примеров данных"""
    
    # Чтение полного датасета
    df = pd.read_csv('data/raw/predictive_maintenance.csv')
    
    # Создание сэмпла сырых данных
    sample_raw = df.head(100)
    sample_raw.to_csv('data/samples/sample_raw.csv', index=False)
    
    # Создание сэмпла обработанных данных
    df_processed = preprocess_data(df.head(100))
    df_processed.to_csv('data/samples/sample_processed.csv', index=False)
    
    print("Созданы файлы сэмплов")

if __name__ == "__main__":
    create_sample_files()