"""
Роутер для эндпоинтов предсказаний
"""

import time
import uuid
from typing import List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
import numpy as np

from api.models import (
    EquipmentRequest,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    AlertLevel,
    ErrorResponse
)
from api.services.model_service import ModelService, get_model_service
from api.services.validation_service import ValidationService
from api.utils.logger import get_logger
from api.utils.metrics import (
    record_prediction_time,
    record_batch_prediction_time,
    increment_prediction_counter,
    increment_error_counter
)

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    },
    summary="Предсказание для одного оборудования",
    description="""
    Предсказывает вероятность выхода оборудования из строя
    на основе данных с датчиков.
    
    ### Требования к данным:
    * Все датчики должны быть в допустимых диапазонах
    * Временная метка не может быть в будущем
    * Данные не старше 7 дней
    
    ### Возвращает:
    * Вероятность отказа (0-1)
    * Бинарное предсказание
    * Уровень алерта
    * Рекомендацию по обслуживанию
    """,
    tags=["predictions"]
)
async def predict(
    request: EquipmentRequest,
    background_tasks: BackgroundTasks,
    model_service: ModelService = Depends(get_model_service)
) -> PredictionResponse:
    """
    Предсказание для одного оборудования
    """
    request_id = f"req_{str(uuid.uuid4())[:10]}"
    start_time = time.time()
    
    try:
        logger.info(f"Prediction request {request_id} for {request.equipment_id}")
        
        # Валидация запроса
        validation_service = ValidationService()
        validation_result = await validation_service.validate_request(request)
        
        if not validation_result.is_valid:
            logger.warning(f"Validation failed for {request_id}: {validation_result.errors}")
            increment_error_counter("validation_error")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Validation failed",
                    "errors": validation_result.errors
                }
            )
        
        # Выполнение предсказания
        prediction_start = time.time()
        prediction_result = await model_service.predict(request)
        inference_time_ms = (time.time() - prediction_start) * 1000
        
        # Запись метрик
        record_prediction_time(inference_time_ms)
        increment_prediction_counter("single")
        
        # Определение уровня алерта и рекомендации
        alert_level, recommendation = _get_alert_and_recommendation(
            prediction_result.failure_probability
        )
        
        # Создание ответа
        response = PredictionResponse(
            request_id=request_id,
            equipment_id=request.equipment_id,
            timestamp=datetime.utcnow(),
            failure_probability=round(prediction_result.failure_probability, 4),
            prediction=prediction_result.prediction,
            confidence=round(prediction_result.confidence, 4),
            alert_level=alert_level,
            recommendation=recommendation,
            inference_time_ms=round(inference_time_ms, 2),
            features_used=prediction_result.features_used,
            model_version=prediction_result.model_version
        )
        
        # Асинхронная отправка в Kafka (если настроено)
        background_tasks.add_task(
            _send_prediction_to_kafka,
            request,
            response,
            model_service
        )
        
        logger.info(f"Prediction completed for {request_id}: "
                   f"prob={response.failure_probability}, "
                   f"time={response.inference_time_ms}ms")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for {request_id}: {str(e)}", exc_info=True)
        increment_error_counter("prediction_error")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal prediction error",
                "request_id": request_id,
                "error_message": str(e)
            }
        )
    
    finally:
        total_time = (time.time() - start_time) * 1000
        logger.debug(f"Total request time for {request_id}: {total_time:.2f}ms")


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    },
    summary="Пакетное предсказание",
    description="""
    Пакетное предсказание для нескольких единиц оборудования.
    
    ### Ограничения:
    * Максимум 1000 запросов в пакете
    * Общий размер данных не более 1MB
    
    ### Возвращает:
    * Список предсказаний для каждого оборудования
    * Статистику по пакету
    * Время выполнения
    """,
    tags=["predictions"]
)
async def batch_predict(
    batch_request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    model_service: ModelService = Depends(get_model_service)
) -> BatchPredictionResponse:
    """
    Пакетное предсказание для нескольких записей
    """
    batch_id = f"batch_{str(uuid.uuid4())[:10]}"
    start_time = time.time()
    
    try:
        logger.info(f"Batch prediction request {batch_id} with {len(batch_request.requests)} items")
        
        # Валидация размера пакета
        if len(batch_request.requests) > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch size cannot exceed 1000 requests"
            )
        
        # Выполнение предсказаний
        batch_start = time.time()
        predictions = []
        successful = 0
        failed = 0
        
        for i, request in enumerate(batch_request.requests):
            request_id = f"{batch_id}_item_{i}"
            
            try:
                # Предсказание для каждого запроса
                prediction_result = await model_service.predict(request)
                
                # Определение уровня алерта и рекомендации
                alert_level, recommendation = _get_alert_and_recommendation(
                    prediction_result.failure_probability
                )
                
                # Создание ответа
                prediction_response = PredictionResponse(
                    request_id=request_id,
                    equipment_id=request.equipment_id,
                    timestamp=datetime.utcnow(),
                    failure_probability=round(prediction_result.failure_probability, 4),
                    prediction=prediction_result.prediction,
                    confidence=round(prediction_result.confidence, 4),
                    alert_level=alert_level,
                    recommendation=recommendation,
                    inference_time_ms=0,  # Заполняется позже
                    features_used=prediction_result.features_used,
                    model_version=prediction_result.model_version
                )
                
                predictions.append(prediction_response)
                successful += 1
                
                # Асинхронная отправка в Kafka
                background_tasks.add_task(
                    _send_prediction_to_kafka,
                    request,
                    prediction_response,
                    model_service
                )
                
            except Exception as e:
                logger.error(f"Failed prediction for {request_id}: {str(e)}")
                failed += 1
                increment_error_counter("batch_item_error")
        
        # Расчет общего времени
        total_inference_time = (time.time() - batch_start) * 1000
        
        # Запись метрик
        record_batch_prediction_time(total_inference_time)
        increment_prediction_counter("batch", successful)
        
        # Обновление времени инференса для каждого предсказания
        avg_time = total_inference_time / max(successful, 1)
        for pred in predictions:
            pred.inference_time_ms = round(avg_time, 2)
        
        # Создание пакетного ответа
        response = BatchPredictionResponse(
            predictions=predictions,
            batch_id=batch_id,
            total_requests=len(batch_request.requests),
            successful_predictions=successful,
            failed_predictions=failed,
            total_inference_time_ms=round(total_inference_time, 2),
            avg_inference_time_ms=round(avg_time, 2)
        )
        
        logger.info(f"Batch prediction completed for {batch_id}: "
                   f"successful={successful}, failed={failed}, "
                   f"total_time={response.total_inference_time_ms}ms")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error for {batch_id}: {str(e)}", exc_info=True)
        increment_error_counter("batch_error")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal batch prediction error",
                "batch_id": batch_id,
                "error_message": str(e)
            }
        )
    
    finally:
        total_time = (time.time() - start_time) * 1000
        logger.debug(f"Total batch request time for {batch_id}: {total_time:.2f}ms")


def _get_alert_and_recommendation(failure_probability: float):
    """
    Определение уровня алерта и рекомендации на основе вероятности отказа
    """
    if failure_probability >= 0.8:
        return "critical", "НЕМЕДЛЕННЫЙ РЕМОНТ - высокая вероятность отказа"
    elif failure_probability >= 0.6:
        return "warning", "ПЛАНОВОЕ ОБСЛУЖИВАНИЕ - рекомендуется проверить оборудование"
    elif failure_probability >= 0.4:
        return "warning", "МОНИТОРИНГ - увеличен риск отказа"
    else:
        return "normal", "НОРМАЛЬНАЯ РАБОТА - оборудование в хорошем состоянии"


async def _send_prediction_to_kafka(request, response, model_service):
    """
    Асинхронная отправка предсказания в Kafka
    """
    try:
        # Проверяем, настроен ли Kafka
        if hasattr(model_service, 'kafka_producer') and model_service.kafka_producer:
            message = {
                "request_id": response.request_id,
                "equipment_id": request.equipment_id,
                "timestamp": response.timestamp.isoformat(),
                "failure_probability": response.failure_probability,
                "prediction": response.prediction,
                "alert_level": response.alert_level,
                "inference_time_ms": response.inference_time_ms
            }
            
            await model_service.send_to_kafka(message)
            logger.debug(f"Prediction sent to Kafka: {response.request_id}")
            
    except Exception as e:
        logger.warning(f"Failed to send prediction to Kafka: {str(e)}")
        # Не бросаем исключение, так как это background task