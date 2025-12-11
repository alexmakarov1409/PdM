"""
Конфигурация приложения с поддержкой переменных окружения
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field, validator
from pydantic.networks import AnyHttpUrl, RedisDsn, PostgresDsn
import secrets


class Settings(BaseSettings):
    """Настройки приложения"""
    
    # Основные настройки
    APP_NAME: str = "Predictive Maintenance API"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field("development", env="ENVIRONMENT")
    DEBUG: bool = Field(False, env="DEBUG")
    
    # Настройки сервера
    HOST: str = Field("0.0.0.0", env="HOST")
    PORT: int = Field(8000, env="PORT")
    WORKERS: int = Field(4, env="WORKERS")
    
    # Безопасность
    SECRET_KEY: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        env="SECRET_KEY"
    )
    API_KEY_HEADER: str = Field("X-API-Key", env="API_KEY_HEADER")
    API_KEYS: List[str] = Field(["test-api-key-123"], env="API_KEYS")
    
    # CORS
    CORS_ORIGINS: List[str] = Field(["*"], env="CORS_ORIGINS")
    
    # Базы данных
    POSTGRES_URL: PostgresDsn = Field(
        "postgresql://postgres:password@localhost:5432/predictive_maintenance",
        env="POSTGRES_URL"
    )
    REDIS_URL: RedisDsn = Field("redis://localhost:6379/0", env="REDIS_URL")
    
    # Feature Store
    FEATURE_STORE_URL: str = Field(
        "postgresql://postgres:password@localhost:5432/feature_store",
        env="FEATURE_STORE_URL"
    )
    
    # Модели
    MODEL_PATH: str = Field("models/final_model.pkl", env="MODEL_PATH")
    SCALER_PATH: str = Field("models/scaler.pkl", env="SCALER_PATH")
    THRESHOLD: float = Field(0.5, env="THRESHOLD")
    
    # Метрики
    ENABLE_METRICS: bool = Field(True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(9090, env="METRICS_PORT")
    
    # Логирование
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field("logs/api.log", env="LOG_FILE")
    
    # Лимиты
    MAX_REQUEST_SIZE: int = Field(1024 * 1024, env="MAX_REQUEST_SIZE")  # 1MB
    RATE_LIMIT_REQUESTS: int = Field(100, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_PERIOD: int = Field(60, env="RATE_LIMIT_PERIOD")  # seconds
    
    # Валидация
    MIN_TEMPERATURE: float = Field(0.0, env="MIN_TEMPERATURE")
    MAX_TEMPERATURE: float = Field(150.0, env="MAX_TEMPERATURE")
    MIN_VIBRATION: float = Field(0.0, env="MIN_VIBRATION")
    MAX_VIBRATION: float = Field(100.0, env="MAX_VIBRATION")
    
    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = Field(
        "localhost:9092", env="KAFKA_BOOTSTRAP_SERVERS"
    )
    KAFKA_TOPIC_PREDICTIONS: str = Field(
        "predictions", env="KAFKA_TOPIC_PREDICTIONS"
    )
    
    # Тестирование
    TESTING: bool = Field(False, env="TESTING")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        allowed = ["development", "testing", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of: {allowed}")
        return v.upper()
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("API_KEYS", pre=True)
    def parse_api_keys(cls, v):
        if isinstance(v, str):
            return [key.strip() for key in v.split(",")]
        return v


# Глобальный экземпляр настроек
settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Получение экземпляра настроек (синглтон)
    """
    global settings_instance
    
    if settings_instance is None:
        settings_instance = Settings()
    
    return settings_instance