"""
Pydantic модели для валидации данных API
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum


class EquipmentType(str, Enum):
    """Типы оборудования"""
    PUMP = "pump"
    COMPRESSOR = "compressor"
    TURBINE = "turbine"
    GENERATOR = "generator"
    CONVEYOR = "conveyor"


class AlertLevel(str, Enum):
    """Уровни алертов"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


class SensorData(BaseModel):
    """Данные с датчиков оборудования"""
    temperature: float = Field(
        ...,
        ge=0,
        le=150,
        description="Температура оборудования (°C)",
        example=85.5
    )
    vibration: float = Field(
        ...,
        ge=0,
        le=100,
        description="Уровень вибрации (mm/s)",
        example=42.3
    )
    pressure: float = Field(
        ...,
        ge=0,
        le=200,
        description="Давление (psi)",
        example=120.8
    )
    rotation_speed: float = Field(
        ...,
        ge=0,
        le=5000,
        description="Скорость вращения (RPM)",
        example=2450.0
    )
    power_consumption: float = Field(
        ...,
        ge=0,
        description="Потребление энергии (kW)",
        example=1250.5
    )
    
    @root_validator
    def validate_sensor_relationships(cls, values):
        """Валидация взаимосвязей между датчиками"""
        temperature = values.get('temperature')
        pressure = values.get('pressure')
        vibration = values.get('vibration')
        
        # Температура и давление должны быть согласованы
        if temperature > 100 and pressure < 50:
            raise ValueError(
                "High temperature with low pressure indicates sensor error"
            )
        
        # Высокая вибрация с нормальной температурой может быть проблемой
        if vibration > 70 and temperature < 60:
            raise ValueError(
                "High vibration with low temperature requires inspection"
            )
        
        return values


class EquipmentRequest(BaseModel):
    """Запрос для предсказания состояния оборудования"""
    equipment_id: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Уникальный идентификатор оборудования",
        example="EQ-001"
    )
    equipment_type: EquipmentType = Field(
        ...,
        description="Тип оборудования",
        example=EquipmentType.PUMP
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Временная метка данных",
        example="2024-01-15T10:30:00Z"
    )
    sensor_data: SensorData = Field(
        ...,
        description="Данные с датчиков"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Дополнительные метаданные",
        example={"location": "factory-a", "operator": "john.doe"}
    )
    
    @validator('equipment_id')
    def validate_equipment_id(cls, v):
        """Валидация ID оборудования"""
        if not v.startswith(('EQ-', 'PUMP-', 'COMP-', 'TURB-')):
            raise ValueError("Equipment ID must start with valid prefix")
        return v
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Валидация временной метки"""
        if v > datetime.utcnow():
            raise ValueError("Timestamp cannot be in the future")
        
        # Не слишком старые данные (максимум 7 дней)
        max_age = datetime.utcnow().timestamp() - (7 * 24 * 3600)
        if v.timestamp() < max_age:
            raise ValueError("Data is too old (more than 7 days)")
        
        return v


class BatchPredictionRequest(BaseModel):
    """Пакетный запрос для предсказаний"""
    requests: List[EquipmentRequest] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="Список запросов для предсказания"
    )
    
    @validator('requests')
    def validate_batch_size(cls, v):
        """Валидация размера пакета"""
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000 requests")
        return v


class PredictionResponse(BaseModel):
    """Ответ с предсказанием"""
    request_id: str = Field(
        ...,
        description="Идентификатор запроса",
        example="req_1234567890"
    )
    equipment_id: str = Field(
        ...,
        description="Идентификатор оборудования"
    )
    timestamp: datetime = Field(
        ...,
        description="Временная метка предсказания"
    )
    failure_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Вероятность выхода из строя (0-1)",
        example=0.8723
    )
    prediction: bool = Field(
        ...,
        description="Бинарное предсказание (True - отказ ожидается)",
        example=True
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Уверенность модели в предсказании",
        example=0.89
    )
    alert_level: AlertLevel = Field(
        ...,
        description="Уровень алерта"
    )
    recommendation: str = Field(
        ...,
        description="Рекомендация для обслуживания",
        example="Требуется немедленный ремонт"
    )
    inference_time_ms: float = Field(
        ...,
        ge=0,
        description="Время выполнения инференса (мс)",
        example=12.45
    )
    features_used: List[str] = Field(
        ...,
        description="Список использованных признаков",
        example=["temperature", "vibration", "pressure"]
    )
    model_version: str = Field(
        ...,
        description="Версия модели",
        example="1.0.0"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BatchPredictionResponse(BaseModel):
    """Пакетный ответ с предсказаниями"""
    predictions: List[PredictionResponse] = Field(
        ...,
        description="Список предсказаний"
    )
    batch_id: str = Field(
        ...,
        description="Идентификатор пакета",
        example="batch_1234567890"
    )
    total_requests: int = Field(
        ...,
        description="Общее количество запросов в пакете"
    )
    successful_predictions: int = Field(
        ...,
        description="Количество успешных предсказаний"
    )
    failed_predictions: int = Field(
        ...,
        description="Количество неудачных предсказаний"
    )
    total_inference_time_ms: float = Field(
        ...,
        description="Общее время инференса для пакета (мс)"
    )
    avg_inference_time_ms: float = Field(
        ...,
        description="Среднее время инференса на запрос (мс)"
    )


class HealthResponse(BaseModel):
    """Ответ health check"""
    status: str = Field(
        ...,
        description="Статус сервиса",
        example="healthy"
    )
    timestamp: datetime = Field(
        ...,
        description="Время проверки"
    )
    service: str = Field(
        ...,
        description="Название сервиса",
        example="Predictive Maintenance API"
    )
    version: str = Field(
        ...,
        description="Версия API",
        example="1.0.0"
    )
    model_loaded: bool = Field(
        ...,
        description="Загружена ли модель",
        example=True
    )
    model_version: Optional[str] = Field(
        None,
        description="Версия модели"
    )
    dependencies: Dict[str, bool] = Field(
        ...,
        description="Статус зависимостей",
        example={
            "database": True,
            "redis": True,
            "feature_store": True
        }
    )
    uptime_seconds: float = Field(
        ...,
        description="Время работы сервиса (секунды)"
    )
    memory_usage_mb: Optional[float] = Field(
        None,
        description="Использование памяти (МБ)"
    )


class ErrorResponse(BaseModel):
    """Стандартный ответ об ошибке"""
    error: str = Field(
        ...,
        description="Текст ошибки",
        example="Model not loaded"
    )
    error_code: str = Field(
        ...,
        description="Код ошибки",
        example="MODEL_NOT_LOADED"
    )
    detail: Optional[str] = Field(
        None,
        description="Детали ошибки"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Время возникновения ошибки"
    )
    request_id: Optional[str] = Field(
        None,
        description="Идентификатор запроса"
    )


class ModelInfoResponse(BaseModel):
    """Информация о модели"""
    model_name: str = Field(
        ...,
        description="Название модели"
    )
    model_version: str = Field(
        ...,
        description="Версия модели"
    )
    model_type: str = Field(
        ...,
        description="Тип модели"
    )
    created_at: datetime = Field(
        ...,
        description="Дата создания модели"
    )
    metrics: Dict[str, float] = Field(
        ...,
        description="Метрики модели",
        example={
            "accuracy": 0.92,
            "recall": 0.89,
            "precision": 0.91
        }
    )
    feature_count: int = Field(
        ...,
        description="Количество признаков"
    )
    threshold: float = Field(
        ...,
        description="Порог классификации"
    )
    requirements: Dict[str, Any] = Field(
        ...,
        description="Требования модели"
    )


class FeatureImportanceResponse(BaseModel):
    """Важность признаков модели"""
    feature_name: str = Field(
        ...,
        description="Название признака"
    )
    importance: float = Field(
        ...,
        description="Важность признака",
        example=0.234
    )
    rank: int = Field(
        ...,
        description="Ранг важности"
    )
    feature_type: str = Field(
        ...,
        description="Тип признака"
    )
    description: Optional[str] = Field(
        None,
        description="Описание признака"
    )