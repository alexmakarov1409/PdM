import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Основная функция очистки данных"""
    # Удаление дубликатов
    df = df.drop_duplicates()
    
    # Обработка пропусков
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Удаление выбросов (метод IQR)
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
    
    return df

def normalize_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Нормализация признаков"""
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler