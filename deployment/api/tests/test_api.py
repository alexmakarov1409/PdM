"""
Тесты для API эндпоинтов
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import json

from fastapi.testclient import TestClient
from fastapi import status

from api.main import app
from api.models import (
    EquipmentRequest,
    SensorData,
    EquipmentType,
    PredictionResponse,
    BatchPredictionResponse
)
from api.services.model_service import ModelService


@pytest.fixture
def client():
    """Фикстура для тестового клиента"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_model_service():
    """Мок сервиса модели"""
    service = AsyncMock(spec=ModelService)
    service.initialize = AsyncMock()
    service.cleanup = AsyncMock()
    
    # Мок для предсказания
    prediction_result = MagicMock()
    prediction_result.failure_probability = 0.85
    prediction_result.prediction = True
    prediction_result.confidence = 0.89
    prediction_result.features_used = ["temperature", "vibration", "pressure"]
    prediction_result.model_version = "1.0.0"
    
    service.predict = AsyncMock(return_value=prediction_result)
    service.get_model_info = AsyncMock(return_value={
        "loaded": True,
        "version": "1.0.0",
        "name": "predictive_model"
    })
    service.check_feature_store = AsyncMock(return_value=True)
    
    return service


@pytest.fixture
def sample_request_data():
    """Фикстура с примером данных запроса"""
    return {
        "equipment_id": "EQ-001",
        "equipment_type": "pump",
        "timestamp": datetime.utcnow().isoformat(),
        "sensor_data": {
            "temperature": 85.5,
            "vibration": 42.3,
            "pressure": 120.8,
            "rotation_speed": 2450.0,
            "power_consumption": 1250.5
        },
        "metadata": {
            "location": "factory-a",
            "operator": "john.doe"
        }
    }


@pytest.fixture
def sample_batch_request_data():
    """Фикстура с примером пакетного запроса"""
    return {
        "requests": [
            {
                "equipment_id": "EQ-001",
                "equipment_type": "pump",
                "timestamp": datetime.utcnow().isoformat(),
                "sensor_data": {
                    "temperature": 85.5,
                    "vibration": 42.3,
                    "pressure": 120.8,
                    "rotation_speed": 2450.0,
                    "power_consumption": 1250.5
                }
            },
            {
                "equipment_id": "EQ-002",
                "equipment_type": "compressor",
                "timestamp": datetime.utcnow().isoformat(),
                "sensor_data": {
                    "temperature": 75.0,
                    "vibration": 35.0,
                    "pressure": 110.5,
                    "rotation_speed": 2300.0,
                    "power_consumption": 1150.0
                }
            }
        ]
    }


class TestRootEndpoint:
    """Тесты корневого эндпоинта"""
    
    def test_root_endpoint(self, client):
        """Тест корневого эндпоинта"""
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert data["message"] == "Predictive Maintenance API"
        assert data["version"] == "1.0.0"


class TestHealthEndpoints:
    """Тесты эндпоинтов здоровья"""
    
    @patch('api.routers.health.get_model_service')
    def test_health_check(self, mock_get_service, client, mock_model_service):
        """Тест health check эндпоинта"""
        mock_get_service.return_value = mock_model_service
        
        response = client.get("/api/v1/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "timestamp" in data
        assert "service" in data
        assert "dependencies" in data
    
    def test_liveness_probe(self, client):
        """Тест liveness probe"""
        response = client.get("/api/v1/health/liveness")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["status"] == "alive"
        assert "timestamp" in data
    
    @patch('api.routers.health.get_model_service')
    def test_readiness_probe_ready(self, mock_get_service, client, mock_model_service):
        """Тест readiness probe когда модель загружена"""
        mock_get_service.return_value = mock_model_service
        
        response = client.get("/api/v1/health/readiness")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["status"] == "ready"
        assert data["model_loaded"] is True
    
    @patch('api.routers.health.get_model_service')
    def test_readiness_probe_not_ready(self, mock_get_service, client, mock_model_service):
        """Тест readiness probe когда модель не загружена"""
        mock_model_service.get_model_info.return_value = {"loaded": False}
        mock_get_service.return_value = mock_model_service
        
        response = client.get("/api/v1/health/readiness")
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        
        assert data["status"] == "not_ready"
        assert data["model_loaded"] is False


class TestPredictionEndpoints:
    """Тесты эндпоинтов предсказаний"""
    
    @patch('api.routers.predictions.get_model_service')
    def test_predict_success(self, mock_get_service, client, mock_model_service, sample_request_data):
        """Тест успешного предсказания"""
        mock_get_service.return_value = mock_model_service
        
        response = client.post("/api/v1/predict", json=sample_request_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Проверка структуры ответа
        assert "request_id" in data
        assert "equipment_id" in data
        assert data["equipment_id"] == sample_request_data["equipment_id"]
        assert "failure_probability" in data
        assert 0 <= data["failure_probability"] <= 1
        assert "prediction" in data
        assert isinstance(data["prediction"], bool)
        assert "alert_level" in data
        assert "recommendation" in data
        assert "inference_time_ms" in data
        assert data["inference_time_ms"] > 0
        assert "model_version" in data
        
        # Проверка, что модель была вызвана с правильными данными
        mock_model_service.predict.assert_called_once()
        call_args = mock_model_service.predict.call_args[0][0]
        assert isinstance(call_args, EquipmentRequest)
        assert call_args.equipment_id == sample_request_data["equipment_id"]
    
    @patch('api.routers.predictions.get_model_service')
    def test_predict_invalid_data(self, mock_get_service, client, mock_model_service):
        """Тест предсказания с невалидными данными"""
        mock_get_service.return_value = mock_model_service
        
        invalid_data = {
            "equipment_id": "EQ-001",
            "equipment_type": "pump",
            "timestamp": datetime.utcnow().isoformat(),
            "sensor_data": {
                "temperature": 200,  # Превышение допустимого значения
                "vibration": 42.3,
                "pressure": 120.8,
                "rotation_speed": 2450.0,
                "power_consumption": 1250.5
            }
        }
        
        response = client.post("/api/v1/predict", json=invalid_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "detail" in response.json()
    
    @patch('api.routers.predictions.get_model_service')
    def test_predict_future_timestamp(self, mock_get_service, client, mock_model_service):
        """Тест предсказания с временной меткой в будущем"""
        mock_get_service.return_value = mock_model_service
        
        future_time = (datetime.utcnow() + timedelta(days=1)).isoformat()
        
        invalid_data = {
            "equipment_id": "EQ-001",
            "equipment_type": "pump",
            "timestamp": future_time,
            "sensor_data": {
                "temperature": 85.5,
                "vibration": 42.3,
                "pressure": 120.8,
                "rotation_speed": 2450.0,
                "power_consumption": 1250.5
            }
        }
        
        response = client.post("/api/v1/predict", json=invalid_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @patch('api.routers.predictions.get_model_service')
    def test_batch_predict_success(self, mock_get_service, client, mock_model_service, sample_batch_request_data):
        """Тест успешного пакетного предсказания"""
        mock_get_service.return_value = mock_model_service
        
        response = client.post("/api/v1/predict/batch", json=sample_batch_request_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Проверка структуры ответа
        assert "batch_id" in data
        assert "predictions" in data
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) == 2
        assert "total_requests" in data
        assert data["total_requests"] == 2
        assert "successful_predictions" in data
        assert "failed_predictions" in data
        assert "total_inference_time_ms" in data
        assert "avg_inference_time_ms" in data
        
        # Проверка каждого предсказания
        for prediction in data["predictions"]:
            assert "request_id" in prediction
            assert "equipment_id" in prediction
            assert "failure_probability" in prediction
            assert "prediction" in prediction
    
    @patch('api.routers.predictions.get_model_service')
    def test_batch_predict_too_large(self, mock_get_service, client, mock_model_service):
        """Тест пакетного предсказания с слишком большим пакетом"""
        mock_get_service.return_value = mock_model_service
        
        # Создаем слишком большой пакет (1001 запрос)
        large_batch = {
            "requests": [
                {
                    "equipment_id": f"EQ-{i:03d}",
                    "equipment_type": "pump",
                    "timestamp": datetime.utcnow().isoformat(),
                    "sensor_data": {
                        "temperature": 85.5,
                        "vibration": 42.3,
                        "pressure": 120.8,
                        "rotation_speed": 2450.0,
                        "power_consumption": 1250.5
                    }
                }
                for i in range(1001)
            ]
        }
        
        response = client.post("/api/v1/predict/batch", json=large_batch)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "detail" in response.json()
        assert "cannot exceed" in response.json()["detail"].lower()
    
    @patch('api.routers.predictions.get_model_service')
    def test_predict_model_not_loaded(self, mock_get_service, client):
        """Тест предсказания когда модель не загружена"""
        # Мок сервиса, который бросает исключение
        mock_service = AsyncMock(spec=ModelService)
        mock_service.predict = AsyncMock(
            side_effect=Exception("Model not loaded")
        )
        mock_get_service.return_value = mock_service
        
        request_data = {
            "equipment_id": "EQ-001",
            "equipment_type": "pump",
            "timestamp": datetime.utcnow().isoformat(),
            "sensor_data": {
                "temperature": 85.5,
                "vibration": 42.3,
                "pressure": 120.8,
                "rotation_speed": 2450.0,
                "power_consumption": 1250.5
            }
        }
        
        response = client.post("/api/v1/predict", json=request_data)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "error" in response.json()
    
    @patch('api.routers.predictions.get_model_service')
    def test_predict_response_time(self, mock_get_service, client, mock_model_service, sample_request_data):
        """Тест времени ответа предсказания"""
        mock_get_service.return_value = mock_model_service
        
        import time
        start_time = time.time()
        response = client.post("/api/v1/predict", json=sample_request_data)
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000
        
        assert response.status_code == status.HTTP_200_OK
        assert response_time < 1000  # Должно быть менее 1 секунды
        
        # Проверяем, что время инференса в ответе разумное
        data = response.json()
        assert data["inference_time_ms"] < 100  # Должно быть менее 100ms


class TestValidation:
    """Тесты валидации данных"""
    
    def test_sensor_data_validation(self):
        """Тест валидации данных датчиков"""
        # Корректные данные
        valid_data = {
            "temperature": 85.5,
            "vibration": 42.3,
            "pressure": 120.8,
            "rotation_speed": 2450.0,
            "power_consumption": 1250.5
        }
        
        sensor_data = SensorData(**valid_data)
        assert sensor_data.temperature == 85.5
        assert sensor_data.vibration == 42.3
        
        # Невалидные данные - температура вне диапазона
        invalid_data = valid_data.copy()
        invalid_data["temperature"] = 200.0  # Превышает 150
        
        with pytest.raises(ValueError):
            SensorData(**invalid_data)
        
        # Невалидные данные - отрицательная вибрация
        invalid_data = valid_data.copy()
        invalid_data["vibration"] = -10.0
        
        with pytest.raises(ValueError):
            SensorData(**invalid_data)
    
    def test_equipment_request_validation(self):
        """Тест валидации запроса оборудования"""
        # Корректный запрос
        valid_request = {
            "equipment_id": "EQ-001",
            "equipment_type": EquipmentType.PUMP,
            "timestamp": datetime.utcnow(),
            "sensor_data": SensorData(
                temperature=85.5,
                vibration=42.3,
                pressure=120.8,
                rotation_speed=2450.0,
                power_consumption=1250.5
            )
        }
        
        request = EquipmentRequest(**valid_request)
        assert request.equipment_id == "EQ-001"
        assert request.equipment_type == EquipmentType.PUMP
        
        # Невалидный ID оборудования
        invalid_request = valid_request.copy()
        invalid_request["equipment_id"] = "INVALID-ID"  # Не начинается с правильного префикса
        
        with pytest.raises(ValueError):
            EquipmentRequest(**invalid_request)
        
        # Будущая временная метка
        invalid_request = valid_request.copy()
        invalid_request["timestamp"] = datetime.utcnow() + timedelta(days=1)
        
        with pytest.raises(ValueError):
            EquipmentRequest(**invalid_request)


class TestErrorHandling:
    """Тесты обработки ошибок"""
    
    @patch('api.routers.predictions.get_model_service')
    def test_validation_error_response(self, mock_get_service, client):
        """Тест ответа на ошибку валидации"""
        mock_service = AsyncMock(spec=ModelService)
        mock_get_service.return_value = mock_service
        
        # Неполные данные (отсутствует sensor_data)
        invalid_data = {
            "equipment_id": "EQ-001",
            "equipment_type": "pump",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        response = client.post("/api/v1/predict", json=invalid_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        
        assert "detail" in data
        assert isinstance(data["detail"], list)
        assert len(data["detail"]) > 0
        
        # Проверяем, что ошибка содержит информацию о поле
        error_detail = data["detail"][0]
        assert "loc" in error_detail
        assert "msg" in error_detail
        assert "type" in error_detail
    
    def test_nonexistent_endpoint(self, client):
        """Тест обращения к несуществующему эндпоинту"""
        response = client.get("/api/v1/nonexistent")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        
        assert "detail" in data
        assert data["detail"] == "Not Found"


@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Асинхронные тесты эндпоинтов"""
    
    async def test_concurrent_predictions(self, mock_model_service):
        """Тест конкурентных запросов предсказаний"""
        from api.routers.predictions import predict
        from api.models import EquipmentRequest, SensorData, EquipmentType
        
        # Создаем несколько запросов
        requests = []
        for i in range(5):
            request = EquipmentRequest(
                equipment_id=f"EQ-{i:03d}",
                equipment_type=EquipmentType.PUMP,
                timestamp=datetime.utcnow(),
                sensor_data=SensorData(
                    temperature=80 + i,
                    vibration=40 + i,
                    pressure=120 + i,
                    rotation_speed=2400 + i * 10,
                    power_consumption=1200 + i * 10
                )
            )
            requests.append(request)
        
        # Запускаем конкурентно
        tasks = []
        for request in requests:
            task = asyncio.create_task(
                predict(request, BackgroundTasks(), mock_model_service)
            )
            tasks.append(task)
        
        # Ожидаем завершения всех задач
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Проверяем результаты
        assert len(results) == 5
        for result in results:
            assert isinstance(result, PredictionResponse)
            assert 0 <= result.failure_probability <= 1
            assert isinstance(result.prediction, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])