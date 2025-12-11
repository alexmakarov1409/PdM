import pandas as pd
import numpy as np
import json
from typing import Any, Dict

def save_json(data: Dict[str, Any], path: str):
    """Сохранение данных в JSON"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_json(path: str) -> Dict[str, Any]:
    """Загрузка данных из JSON"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Расчет статистик по датасету"""
    stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
    }
    return stats