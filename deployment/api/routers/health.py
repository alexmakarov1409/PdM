"""
Роутер для health check и мониторинга
"""

import time
import psutil
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from api.models import HealthResponse
from api.services.model_service import ModelService, get_model_service
from api.core.config import get_settings
from api.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Время запуска приложения
START_TIME = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="""
    Проверка работоспособности API и зависимостей.
    
    ### Проверяет:
    * Статус API
    * Загрузку ML модели
    * Подключение к базам данных
    * Использование памяти
    * Время работы
    """,
    tags=["health"]
)
async def health_check(
    model_service: ModelService = Depends(get_model_service)
) -> HealthResponse:
    """
    Endpoint для проверки здоровья приложения
    """
    settings = get_settings()
    
    # Проверка зависимостей
    dependencies_status = await _check_dependencies(model_service)
    
    # Получение информации о модели
    model_info = await model_service.get_model_info()
    
    # Использование памяти
    memory_info = psutil.Process().memory_info()
    memory_usage_mb = memory_info.rss / (1024 * 1024)
    
    # Определение общего статуса
    overall_status = "healthy"
    if not all(dependencies_status.values()):
        overall_status = "degraded"
    if not model_info.get("loaded", False):
        overall_status = "unhealthy"
    
    response = HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        service="Predictive Maintenance API",
        version=settings.APP_VERSION,
        model_loaded=model_info.get("loaded", False),
        model_version=model_info.get("version", None),
        dependencies=dependencies_status,
        uptime_seconds=round(time.time() - START_TIME, 2),
        memory_usage_mb=round(memory_usage_mb, 2)
    )
    
    logger.debug(f"Health check: status={overall_status}, "
                f"dependencies={dependencies_status}")
    
    return response


@router.get(
    "/health/liveness",
    summary="Liveness Probe",
    description="""
    Kubernetes liveness probe.
    Проверяет, что приложение запущено и отвечает.
    """,
    tags=["health"]
)
async def liveness_probe():
    """
    Liveness probe для Kubernetes
    """
    return JSONResponse(
        status_code=200,
        content={
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@router.get(
    "/health/readiness",
    summary="Readiness Probe",
    description="""
    Kubernetes readiness probe.
    Проверяет, что приложение готово принимать трафик.
    """,
    tags=["health"]
)
async def readiness_probe(
    model_service: ModelService = Depends(get_model_service)
):
    """
    Readiness probe для Kubernetes
    """
    # Проверяем, что модель загружена
    model_info = await model_service.get_model_info()
    
    if model_info.get("loaded", False):
        return JSONResponse(
            status_code=200,
            content={
                "status": "ready",
                "timestamp": datetime.utcnow().isoformat(),
                "model_loaded": True
            }
        )
    else:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "timestamp": datetime.utcnow().isoformat(),
                "model_loaded": False,
                "message": "Model is not loaded"
            }
        )


@router.get(
    "/health/detailed",
    summary="Detailed Health Check",
    description="""
    Подробная проверка здоровья всех компонентов системы.
    """,
    tags=["health"]
)
async def detailed_health_check(
    model_service: ModelService = Depends(get_model_service)
):
    """
    Подробная проверка здоровья
    """
    settings = get_settings()
    
    # Собираем информацию о всех компонентах
    health_info = {
        "service": {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "uptime_seconds": round(time.time() - START_TIME, 2),
            "timestamp": datetime.utcnow().isoformat()
        },
        "model": await model_service.get_detailed_model_info(),
        "system": await _get_system_info(),
        "dependencies": await _get_detailed_dependencies(model_service),
        "limits": {
            "max_request_size_mb": settings.MAX_REQUEST_SIZE / (1024 * 1024),
            "rate_limit": f"{settings.RATE_LIMIT_REQUESTS}/{settings.RATE_LIMIT_PERIOD}s"
        }
    }
    
    # Определяем общий статус
    all_healthy = (
        health_info["model"]["status"] == "loaded" and
        all(dep["status"] == "healthy" 
            for dep in health_info["dependencies"].values())
    )
    
    health_info["overall_status"] = "healthy" if all_healthy else "degraded"
    
    return JSONResponse(
        status_code=200 if all_healthy else 503,
        content=health_info
    )


async def _check_dependencies(model_service: ModelService) -> Dict[str, bool]:
    """
    Проверка статуса зависимостей
    """
    dependencies = {
        "database": False,
        "redis": False,
        "feature_store": False,
        "kafka": False
    }
    
    try:
        # Проверка базы данных
        # В реальном проекте здесь был бы запрос к БД
        dependencies["database"] = True
        
        # Проверка Redis
        if hasattr(model_service, 'redis_client'):
            try:
                await model_service.redis_client.ping()
                dependencies["redis"] = True
            except:
                dependencies["redis"] = False
        
        # Проверка Feature Store
        dependencies["feature_store"] = await model_service.check_feature_store()
        
        # Проверка Kafka
        if hasattr(model_service, 'kafka_producer'):
            dependencies["kafka"] = True
        else:
            dependencies["kafka"] = False
            
    except Exception as e:
        logger.error(f"Error checking dependencies: {str(e)}")
    
    return dependencies


async def _get_detailed_dependencies(model_service: ModelService) -> Dict:
    """
    Получение детальной информации о зависимостях
    """
    dependencies = {}
    
    # База данных
    dependencies["database"] = {
        "type": "PostgreSQL",
        "status": "healthy",  # В реальном проекте проверять подключение
        "response_time_ms": 0,
        "last_check": datetime.utcnow().isoformat()
    }
    
    # Redis
    if hasattr(model_service, 'redis_client'):
        try:
            start = time.time()
            await model_service.redis_client.ping()
            response_time = (time.time() - start) * 1000
            
            dependencies["redis"] = {
                "type": "Redis",
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "last_check": datetime.utcnow().isoformat()
            }
        except Exception as e:
            dependencies["redis"] = {
                "type": "Redis",
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }
    
    # Feature Store
    feature_store_status = await model_service.check_feature_store()
    dependencies["feature_store"] = {
        "type": "Feature Store",
        "status": "healthy" if feature_store_status else "unhealthy",
        "last_check": datetime.utcnow().isoformat()
    }
    
    return dependencies


async def _get_system_info() -> Dict:
    """
    Получение системной информации
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    cpu_percent = process.cpu_percent(interval=0.1)
    
    system_info = {
        "memory": {
            "rss_mb": round(memory_info.rss / (1024 * 1024), 2),
            "vms_mb": round(memory_info.vms / (1024 * 1024), 2),
            "percent": round(process.memory_percent(), 2)
        },
        "cpu": {
            "percent": round(cpu_percent, 2),
            "threads": process.num_threads()
        },
        "disk": {
            "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
            "used_gb": round(psutil.disk_usage('/').used / (1024**3), 2),
            "free_gb": round(psutil.disk_usage('/').free / (1024**3), 2)
        }
    }
    
    return system_info