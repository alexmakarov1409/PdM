import joblib
import pandas as pd
import numpy as np
import time

def load_model(path: str):
    """Загрузка обученной модели"""
    return joblib.load(path)

def predict(model, features: pd.DataFrame) -> dict:
    """Предсказание вероятности отказа"""
    start_time = time.time()
    
    # Предсказание
    proba = model.predict_proba(features)[:, 1]
    prediction = (proba > 0.5).astype(int)
    
    # Время инференса
    inference_time = (time.time() - start_time) * 1000  # в миллисекундах
    
    return {
        'predictions': prediction,
        'probabilities': proba,
        'inference_time_ms': inference_time
    }

def batch_predict(model, batch_data: pd.DataFrame) -> pd.DataFrame:
    """Пакетное предсказание"""
    results = predict(model, batch_data)
    
    output_df = batch_data.copy()
    output_df['failure_probability'] = results['probabilities']
    output_df['predicted_failure'] = results['predictions']
    
    return output_df