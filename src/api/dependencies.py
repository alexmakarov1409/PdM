from fastapi import Depends, HTTPException, Header
import joblib
import numpy as np
from typing import Optional

# Глобальные переменные для модели
_model = None
_scaler = None

def get_model():
    """Зависимость для получения модели"""
    global _model, _scaler
    
    if _model is None:
        try:
            _model = joblib.load('models/final_model.pkl')
            _scaler = joblib.load('models/scaler.pkl')
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {str(e)}"
            )
    
    return _model, _scaler

def verify_api_key(api_key: Optional[str] = Header(None)):
    """Проверка API ключа (для production)"""
    # В реальном проекте здесь была бы проверка ключа
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API key is required"
        )
    
    # Пример проверки (в production - вынести в настройки)
    valid_keys = ["predictive-ai-2024", "test-key-123"]
    if api_key not in valid_keys:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    
    return api_key