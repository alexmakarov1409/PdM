"""
FastAPI приложение для Predictive Maintenance API
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn

from api.core.config import Settings, get_settings
from api.core.exceptions import (
    ModelNotLoadedError,
    ValidationError,
    ServiceUnavailableError
)
from api.routers import predictions, health, data
from api.utils.logger import setup_logger
from api.utils.metrics import setup_metrics
from api.services.model_service import ModelService

# Настройка логгера
logger = setup_logger(__name__)

# Глобальные переменные для состояния приложения
model_service: ModelService = None
settings: Settings = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекстный менеджер для управления жизненным циклом приложения
    """
    global model_service, settings
    
    # Startup
    logger.info("Starting Predictive Maintenance API...")
    
    # Загрузка конфигурации
    settings = get_settings()
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    
    # Инициализация сервиса модели
    try:
        model_service = ModelService(
            model_path=settings.MODEL_PATH,
            scaler_path=settings.SCALER_PATH,
            feature_store_url=settings.FEATURE_STORE_URL,
            redis_url=settings.REDIS_URL
        )
        await model_service.initialize()
        logger.info("Model service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model service: {str(e)}")
        raise
    
    # Настройка метрик Prometheus
    setup_metrics()
    
    logger.info("API startup completed")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Predictive Maintenance API...")
    
    if model_service:
        await model_service.cleanup()
    
    logger.info("API shutdown completed")


# Создание FastAPI приложения
app = FastAPI(
    title="Predictive Maintenance API",
    description="""
    REST API для предсказания выхода оборудования из строя.
    
    ## Возможности:
    * 📊 Предсказание вероятности отказа оборудования
    * 🔄 Пакетная обработка данных
    * 📈 Мониторинг и метрики
    * 🔐 Аутентификация и авторизация
    * 📝 Автоматическая документация OpenAPI
    """,
    version="1.0.0",
    contact={
        "name": "AI Architecture Team",
        "email": "ai-team@company.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В production заменить на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,
)


# Обработчики исключений
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Обработчик ошибок валидации"""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.errors(),
            "body": exc.body
        },
    )


@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    """Обработчик ошибки загрузки модели"""
    logger.error(f"Model not loaded: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": "Model is not loaded",
            "error": str(exc)
        },
    )


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Обработчик ошибок валидации бизнес-логики"""
    logger.warning(f"Business validation error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": str(exc),
            "error_code": exc.error_code
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Обработчик общих исключений"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        },
    )


# Подключение роутеров
app.include_router(
    predictions.router,
    prefix="/api/v1",
    tags=["predictions"]
)

app.include_router(
    health.router,
    prefix="/api/v1",
    tags=["health"]
)

app.include_router(
    data.router,
    prefix="/api/v1",
    tags=["data"]
)


# Корневой эндпоинт
@app.get("/", tags=["root"])
async def root():
    """Корневой эндпоинт API"""
    return {
        "message": "Predictive Maintenance API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "environment": settings.ENVIRONMENT if settings else "unknown"
    }


@app.get("/api/v1/info")
async def api_info():
    """Информация о API"""
    model_info = await model_service.get_model_info() if model_service else {}
    
    return {
        "name": "Predictive Maintenance API",
        "version": "1.0.0",
        "status": "operational" if model_service else "degraded",
        "model_loaded": model_service is not None,
        "model_info": model_info,
        "endpoints": {
            "predict": "/api/v1/predict",
            "batch_predict": "/api/v1/predict/batch",
            "health": "/api/v1/health",
            "metrics": "/metrics"
        }
    }


# Настройка инструментации для Prometheus
@app.on_event("startup")
async def startup_event():
    """Настройка метрик при старте"""
    Instrumentator().instrument(app).expose(app)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )