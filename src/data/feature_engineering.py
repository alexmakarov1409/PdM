import pandas as pd
import numpy as np

def create_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Создание временных признаков"""
    df['hour'] = pd.to_datetime(df[timestamp_col]).dt.hour
    df['day_of_week'] = pd.to_datetime(df[timestamp_col]).dt.dayofweek
    df['month'] = pd.to_datetime(df[timestamp_col]).dt.month
    return df

def create_rolling_features(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Создание скользящих статистик"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        df[f'{col}_rolling_mean'] = df[col].rolling(window=window).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=window).std()
        df[f'{col}_trend'] = df[col].diff(window)
    
    return df

def create_lag_features(df: pd.DataFrame, lags: list = [1, 3, 7]) -> pd.DataFrame:
    """Создание лаговых признаков"""
    for lag in lags:
        df[f'target_lag_{lag}'] = df['failure_target'].shift(lag)
    return df