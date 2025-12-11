"""
Интеграционные тесты API
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.main import app
from api.core.config import Settings
from api.models import EquipmentRequest, SensorData, EquipmentType
from api.database.models import Base, Equipment, SensorData as DBSensorData


class TestAPIIntegration:
    """Интеграционные тесты API"""
    
    @pytest.fixture(autouse=True)
    def setup_database(self):
        """Настройка тестовой базы данных"""
        # Создаем in-memory базу данных для тестов
        self.engine = create_engine("sqlite:///:memory:")
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Создаем таблицы
        Base.metadata.create_all(bind=self.engine)
        
        # Создаем тестовые данные
        self.session = self.SessionLocal()
        
        # Добавляем тестовое оборудование
        equipment = Equipment(
            equipment_id="EQ-001",
            equipment_type="pump",
            manufacturer="Test Manufacturer",
            model="Test Model",
            installation_date=datetime.utcnow().date(),
            location="Test Location"
        )
        self.session.add(equipment)
        
        # Добавляем тестовые данные датчиков
        for i in range(5):
            sensor_data = DBSensorData(
                equipment_id="EQ-001",
                timestamp=datetime.utcnow() - timedelta(hours=i),
                temperature=80.0 + i,
                vibration=40.0 + i,
                pressure=120.0 + i,
                rotation_speed=2400.0 + i * 10,
                power_consumption=1200.0 + i * 10
            )
            self.session.add(sensor_data)
        
        self.session.commit()
        
        yield
        
        # Очистка после тестов
        self.session.close()
        Base.metadata.drop_all(bind=self.engine)
    
    @pytest.fixture
    def client(self):
        """Фикстура тестового клиента"""
        with TestClient(app) as test_client:
            yield test_client
    
    @pytest.fixture
    def mock_model_service(self):
        """Мок сервиса модели"""
        with patch('api.main.ModelService') as MockService:
            mock_service = AsyncMock()
            mock_service.initialize = AsyncMock()
            mock_service.cleanup = AsyncMock()
            
            # Мок предсказания
            mock_prediction = AsyncMock()
            mock_prediction.failure_probability = 0.85
            mock_prediction.prediction = True
            mock_prediction.confidence = 0.89
            mock_prediction.features_used = ["temperature", "vibration", "pressure"]
            mock_prediction.model_version = "1.0.0"
            
            mock_service.predict = AsyncMock(return_value=mock_prediction)
            mock_service.get_model_info = AsyncMock(return_value={
                "loaded": True,
                "version": "1.0.0"
            })
            
            MockService.return_value = mock_service
            yield mock_service
    
    def test_health_check_integration(self, client, mock_model_service):
        """Интеграционный тест health check"""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "service" in data
        assert data["service"] == "Predictive Maintenance API"
        
        # Проверяем, что модель загружена
        assert data["model_loaded"] is True
    
    def test_prediction_integration(self, client, mock_model_service):
        """Интеграционный тест предсказания"""
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
            },
            "metadata": {
                "location": "factory-a",
                "operator": "john.doe"
            }
        }
        
        response = client.post("/api/v1/predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Проверяем структуру ответа
        assert "request_id" in data
        assert "equipment_id" in data
        assert data["equipment_id"] == "EQ-001"
        assert "failure_probability" in data
        assert 0 <= data["failure_probability"] <= 1
        assert "prediction" in data
        assert isinstance(data["prediction"], bool)
        assert "alert_level" in data
        assert "recommendation" in data
        assert "inference_time_ms" in data
        assert data["inference_time_ms"] > 0
        assert "model_version" in data
        
        # Проверяем, что сервис был вызван
        mock_model_service.predict.assert_called_once()
    
    def test_batch_prediction_integration(self, client, mock_model_service):
        """Интеграционный тест пакетного предсказания"""
        batch_request = {
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
        
        response = client.post("/api/v1/predict/batch", json=batch_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Проверяем структуру ответа
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
        
        # Проверяем каждое предсказание
        for prediction in data["predictions"]:
            assert "request_id" in prediction
            assert "equipment_id" in prediction
            assert "failure_probability" in prediction
            assert "prediction" in prediction
        
        # Проверяем, что сервис был вызван для каждого запроса
        assert mock_model_service.predict.call_count == 2
    
    def test_prediction_with_invalid_data(self, client):
        """Тест предсказания с невалидными данными"""
        # Температура вне диапазона
        invalid_request = {
            "equipment_id": "EQ-001",
            "equipment_type": "pump",
            "timestamp": datetime.utcnow().isoformat(),
            "sensor_data": {
                "temperature": 200.0,  # > 150
                "vibration": 42.3,
                "pressure": 120.8,
                "rotation_speed": 2450.0,
                "power_consumption": 1250.5
            }
        }
        
        response = client.post("/api/v1/predict", json=invalid_request)
        
        # Должен вернуть 422 (Validation Error)
        assert response.status_code == 422
        data = response.json()
        
        assert "detail" in data
        # Проверяем, что ошибка содержит информацию о поле
        error_detail = data["detail"][0]
        assert "loc" in error_detail
        assert "temperature" in str(error_detail["loc"]).lower()
    
    def test_prediction_with_future_timestamp(self, client):
        """Тест предсказания с временной меткой в будущем"""
        future_time = (datetime.utcnow() + timedelta(days=1)).isoformat()
        
        invalid_request = {
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
        
        response = client.post("/api/v1/predict", json=invalid_request)
        
        assert response.status_code == 422
        data = response.json()
        
        assert "detail" in data
        assert any("future" in str(error).lower() for error in data["detail"])
    
    def test_prediction_with_nonexistent_equipment(self, client, mock_model_service):
        """Тест предсказания для несуществующего оборудования"""
        request_data = {
            "equipment_id": "EQ-NONEXISTENT",
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
        
        # Настраиваем мок, чтобы он симулировал проверку оборудования
        from api.core.exceptions import ValidationError
        mock_model_service.predict.side_effect = ValidationError(
            "Equipment not found", "EQUIPMENT_NOT_FOUND"
        )
        
        response = client.post("/api/v1/predict", json=request_data)
        
        # Должен вернуть 400 (Bad Request)
        assert response.status_code == 400
        data = response.json()
        
        assert "detail" in data
        assert "error" in data["detail"]
        assert "not found" in data["detail"]["error"].lower()
    
    def test_api_rate_limiting(self, client, mock_model_service):
        """Тест ограничения скорости запросов"""
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
        
        # Мокаем rate limiter
        with patch('api.main.rate_limiter') as mock_limiter:
            mock_limiter.is_allowed.return_value = False  # Лимит превышен
            
            response = client.post("/api/v1/predict", json=request_data)
            
            assert response.status_code == 429  # Too Many Requests
            data = response.json()
            
            assert "detail" in data
            assert "rate limit" in data["detail"].lower()
    
    def test_concurrent_predictions(self, client, mock_model_service):
        """Тест конкурентных запросов предсказаний"""
        import concurrent.futures
        
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
        
        # Функция для выполнения запроса
        def make_request():
            return client.post("/api/v1/predict", json=request_data)
        
        # Выполняем 10 конкурентных запросов
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # Проверяем, что все запросы успешны
        successful = [r for r in results if r.status_code == 200]
        assert len(successful) == 10
        
        # Проверяем, что сервис был вызван 10 раз
        assert mock_model_service.predict.call_count == 10
    
    def test_prediction_performance(self, client, mock_model_service):
        """Тест производительности предсказаний"""
        import time
        
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
        
        # Измеряем время выполнения
        start_time = time.time()
        response = client.post("/api/v1/predict", json=request_data)
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000  # в миллисекундах
        
        assert response.status_code == 200
        
        # Проверяем время ответа (должно быть < 100ms согласно требованиям)
        data = response.json()
        assert data["inference_time_ms"] < 100
        
        # Общее время запроса должно быть разумным
        assert response_time < 500  # < 500ms
    
    def test_error_handling(self, client, mock_model_service):
        """Тест обработки ошибок"""
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
        
        # Настраиваем мок, чтобы он бросал исключение
        mock_model_service.predict.side_effect = Exception("Internal server error")
        
        response = client.post("/api/v1/predict", json=request_data)
        
        # Должен вернуть 500 (Internal Server Error)
        assert response.status_code == 500
        data = response.json()
        
        assert "detail" in data
        assert "internal server error" in data["detail"].lower()
    
    def test_api_documentation(self, client):
        """Тест доступности документации API"""
        # Проверяем Swagger UI
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger-ui" in response.text.lower()
        
        # Проверяем OpenAPI спецификацию
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        
        # Проверяем основные поля OpenAPI
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
        assert "/api/v1/predict" in data["paths"]
        assert "/api/v1/health" in data["paths"]
        
        # Проверяем ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "redoc" in response.text.lower()


class TestDatabaseIntegration:
    """Интеграционные тесты базы данных"""
    
    def test_equipment_crud(self):
        """Тест CRUD операций для оборудования"""
        from api.database.models import Equipment
        from api.database.session import SessionLocal
        
        session = SessionLocal()
        
        try:
            # Create
            equipment = Equipment(
                equipment_id="TEST-EQ-001",
                equipment_type="test_pump",
                manufacturer="Test Manufacturer",
                model="Test Model",
                installation_date=datetime.utcnow().date(),
                location="Test Location"
            )
            session.add(equipment)
            session.commit()
            
            # Read
            retrieved = session.query(Equipment).filter_by(equipment_id="TEST-EQ-001").first()
            assert retrieved is not None
            assert retrieved.equipment_type == "test_pump"
            assert retrieved.manufacturer == "Test Manufacturer"
            
            # Update
            retrieved.location = "Updated Location"
            session.commit()
            
            updated = session.query(Equipment).filter_by(equipment_id="TEST-EQ-001").first()
            assert updated.location == "Updated Location"
            
            # Delete
            session.delete(updated)
            session.commit()
            
            deleted = session.query(Equipment).filter_by(equipment_id="TEST-EQ-001").first()
            assert deleted is None
            
        finally:
            session.close()
    
    def test_sensor_data_relationships(self):
        """Тест связей между таблицами"""
        from api.database.models import Equipment, SensorData
        from api.database.session import SessionLocal
        
        session = SessionLocal()
        
        try:
            # Создаем оборудование
            equipment = Equipment(
                equipment_id="TEST-EQ-002",
                equipment_type="test_compressor",
                manufacturer="Test Manufacturer"
            )
            session.add(equipment)
            session.commit()
            
            # Создаем данные датчиков
            sensor_data = SensorData(
                equipment_id="TEST-EQ-002",
                timestamp=datetime.utcnow(),
                temperature=85.5,
                vibration=42.3,
                pressure=120.8
            )
            session.add(sensor_data)
            session.commit()
            
            # Проверяем связь
            retrieved_equipment = session.query(Equipment).filter_by(
                equipment_id="TEST-EQ-002"
            ).first()
            
            assert retrieved_equipment is not None
            # Проверяем, что можно получить данные датчиков через связь
            assert len(retrieved_equipment.sensor_data) == 1
            assert retrieved_equipment.sensor_data[0].temperature == 85.5
            
            # Проверяем обратную связь
            retrieved_sensor = session.query(SensorData).filter_by(
                equipment_id="TEST-EQ-002"
            ).first()
            
            assert retrieved_sensor is not None
            assert retrieved_sensor.equipment.equipment_id == "TEST-EQ-002"
            assert retrieved_sensor.equipment.equipment_type == "test_compressor"
            
        finally:
            # Очистка
            session.query(SensorData).filter_by(equipment_id="TEST-EQ-002").delete()
            session.query(Equipment).filter_by(equipment_id="TEST-EQ-002").delete()
            session.commit()
            session.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])