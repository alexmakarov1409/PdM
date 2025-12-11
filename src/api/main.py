from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
import numpy as np
import joblib
import pandas as pd
from typing import List, Optional
import logging
import time
from datetime import datetime
import json

# Инициализация логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация FastAPI приложения
app = FastAPI(
    title="Predictive Maintenance API",
    description="API для предсказания вероятности выхода оборудования из строя",
    version="1.0.0"
)

# Модель загружается один раз при старте
MODEL = None
SCALER = None

# Pydantic схемы
class EquipmentData(BaseModel):
    """Схема для данных оборудования"""
    equipment_id: str = Field(..., example="EQ-001")
    timestamp: str = Field(..., example="2024-01-15T10:30:00")
    temperature: float = Field(..., ge=0, le=150, example=85.5)
    vibration: float = Field(..., ge=0, le=100, example=45.2)
    pressure: float = Field(..., ge=0, le=200, example=120.8)
    rotation_speed: float = Field(..., ge=0, le=5000, example=2450.0)
    power_consumption: float = Field(..., ge=0, example=1250.5)
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError('Invalid timestamp format. Use ISO 8601')

class BatchPredictionRequest(BaseModel):
    """Схема для пакетного предсказания"""
    data: List[EquipmentData]
    
    @validator('data')
    def validate_data_length(cls, v):
        if len(v) > 1000:
            raise ValueError('Batch size cannot exceed 1000 records')
        return v

class PredictionResponse(BaseModel):
    """Схема ответа предсказания"""
    equipment_id: str
    timestamp: str
    failure_probability: float = Field(..., ge=0, le=1)
    prediction: str
    confidence: float = Field(..., ge=0, le=1)
    inference_time_ms: float
    recommendation: str

class HealthResponse(BaseModel):
    """Схема для health check"""
    status: str
    model_loaded: bool
    api_version: str
    timestamp: str

class ModelMetrics(BaseModel):
    """Метрики модели"""
    recall: float
    fpr: float
    accuracy: float
    roc_auc: float

# Загрузка модели при старте
@app.on_event("startup")
async def load_model():
    """Загрузка модели и скалера при старте приложения"""
    global MODEL, SCALER
    
    try:
        MODEL = joblib.load('models/final_model.pkl')
        SCALER = joblib.load('models/scaler.pkl')
        logger.info("Model and scaler loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        MODEL = None
        SCALER = None

# Эндпоинты
@app.get("/", tags=["Root"])
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Predictive Maintenance API",
        "documentation": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Проверка здоровья API"""
    return HealthResponse(
        status="healthy" if MODEL is not None else "degraded",
        model_loaded=MODEL is not None,
        api_version="1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )

@app.get("/metrics", response_model=ModelMetrics, tags=["Monitoring"])
async def get_metrics():
    """Получение метрик модели"""
    # В реальном проекте эти метрики загружались бы из файла
    return ModelMetrics(
        recall=0.89,
        fpr=0.12,
        accuracy=0.91,
        roc_auc=0.94
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(equipment_data: EquipmentData):
    """Предсказание для одного оборудования"""
    start_time = time.time()
    
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    try:
        # Преобразование в DataFrame
        df = pd.DataFrame([equipment_data.dict()])
        
        # Преобразование временной метки
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Подготовка признаков
        feature_columns = ['temperature', 'vibration', 'pressure', 
                          'rotation_speed', 'power_consumption',
                          'hour', 'day_of_week', 'month']
        
        features = df[feature_columns].values
        
        # Масштабирование
        features_scaled = SCALER.transform(features)
        
        # Предсказание
        probability = MODEL.predict_proba(features_scaled)[0][1]
        
        # Определение класса
        prediction_class = "FAILURE" if probability > 0.5 else "NORMAL"
        confidence = probability if prediction_class == "FAILURE" else 1 - probability
        
        # Рекомендация
        if probability > 0.8:
            recommendation = "НЕМЕДЛЕННЫЙ РЕМОНТ"
        elif probability > 0.5:
            recommendation = "ПЛАНОВОЕ ОБСЛУЖИВАНИЕ"
        else:
            recommendation = "НОРМАЛЬНАЯ РАБОТА"
        
        inference_time = (time.time() - start_time) * 1000  # мс
        
        # Проверка времени инференса
        if inference_time > 50:
            logger.warning(f"Inference time exceeded 50ms: {inference_time:.2f}ms")
        
        return PredictionResponse(
            equipment_id=equipment_data.equipment_id,
            timestamp=equipment_data.timestamp,
            failure_probability=round(probability, 4),
            prediction=prediction_class,
            confidence=round(confidence, 4),
            inference_time_ms=round(inference_time, 2),
            recommendation=recommendation
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Prediction"])
async def predict_batch(batch_request: BatchPredictionRequest):
    """Пакетное предсказание для нескольких записей"""
    start_time = time.time()
    
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    try:
        results = []
        
        for item in batch_request.data:
            # Используем тот же код, что и для одиночного предсказания
            df = pd.DataFrame([item.dict()])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            
            feature_columns = ['temperature', 'vibration', 'pressure', 
                              'rotation_speed', 'power_consumption',
                              'hour', 'day_of_week', 'month']
            
            features = df[feature_columns].values
            features_scaled = SCALER.transform(features)
            probability = MODEL.predict_proba(features_scaled)[0][1]
            
            prediction_class = "FAILURE" if probability > 0.5 else "NORMAL"
            confidence = probability if prediction_class == "FAILURE" else 1 - probability
            
            if probability > 0.8:
                recommendation = "НЕМЕДЛЕННЫЙ РЕМОНТ"
            elif probability > 0.5:
                recommendation = "ПЛАНОВОЕ ОБСЛУЖИВАНИЕ"
            else:
                recommendation = "НОРМАЛЬНАЯ РАБОТА"
            
            results.append(PredictionResponse(
                equipment_id=item.equipment_id,
                timestamp=item.timestamp,
                failure_probability=round(probability, 4),
                prediction=prediction_class,
                confidence=round(confidence, 4),
                inference_time_ms=0,  # Для пакета считаем общее время
                recommendation=recommendation
            ))
        
        batch_time = (time.time() - start_time) * 1000
        avg_time = batch_time / len(results)
        logger.info(f"Batch prediction completed: {len(results)} items, "
                   f"total: {batch_time:.2f}ms, avg: {avg_time:.2f}ms")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/equipment/{equipment_id}/history", tags=["Equipment"])
async def get_equipment_history(equipment_id: str, limit: int = 100):
    """Получение истории предсказаний для оборудования"""
    # В реальном проекте здесь был бы запрос к БД
    return {
        "equipment_id": equipment_id,
        "history": [],
        "message": "В реальной реализации здесь была бы история из базы данных"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)