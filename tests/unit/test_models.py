"""
Unit тесты для Pydantic моделей данных
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from api.models import (
    SensorData,
    EquipmentRequest,
    BatchPredictionRequest,
    PredictionResponse,
    EquipmentType,
    AlertLevel
)


class TestSensorData:
    """Тесты модели SensorData"""
    
    def test_valid_sensor_data(self):
        """Тест валидных данных датчиков"""
        data = {
            "temperature": 85.5,
            "vibration": 42.3,
            "pressure": 120.8,
            "rotation_speed": 2450.0,
            "power_consumption": 1250.5
        }
        
        sensor_data = SensorData(**data)
        
        assert sensor_data.temperature == 85.5
        assert sensor_data.vibration == 42.3
        assert sensor_data.pressure == 120.8
        assert sensor_data.rotation_speed == 2450.0
        assert sensor_data.power_consumption == 1250.5
    
    def test_temperature_out_of_range(self):
        """Тест температуры вне допустимого диапазона"""
        data = {
            "temperature": 200.0,  # Максимум 150
            "vibration": 42.3,
            "pressure": 120.8,
            "rotation_speed": 2450.0,
            "power_consumption": 1250.5
        }
        
        with pytest.raises(ValueError) as exc_info:
            SensorData(**data)
        
        assert "less than or equal to 150" in str(exc_info.value)
    
    def test_negative_vibration(self):
        """Тест отрицательной вибрации"""
        data = {
            "temperature": 85.5,
            "vibration": -10.0,  # Не может быть отрицательной
            "pressure": 120.8,
            "rotation_speed": 2450.0,
            "power_consumption": 1250.5
        }
        
        with pytest.raises(ValueError) as exc_info:
            SensorData(**data)
        
        assert "greater than or equal to 0" in str(exc_info.value)
    
    def test_missing_field(self):
        """Тест отсутствующего поля"""
        data = {
            "temperature": 85.5,
            "vibration": 42.3,
            "pressure": 120.8,
            "rotation_speed": 2450.0
            # Отсутствует power_consumption
        }
        
        with pytest.raises(ValueError) as exc_info:
            SensorData(**data)
        
        assert "field required" in str(exc_info.value)
    
    def test_sensor_relationship_validation(self):
        """Тест валидации взаимосвязей датчиков"""
        # Высокая температура с низким давлением должно вызывать ошибку
        data = {
            "temperature": 120.0,  # Высокая
            "vibration": 42.3,
            "pressure": 30.0,  # Низкое
            "rotation_speed": 2450.0,
            "power_consumption": 1250.5
        }
        
        with pytest.raises(ValueError) as exc_info:
            SensorData(**data)
        
        assert "sensor error" in str(exc_info.value).lower()
    
    def test_edge_cases(self):
        """Тест граничных значений"""
        # Минимальные значения
        min_data = {
            "temperature": 0.0,
            "vibration": 0.0,
            "pressure": 0.0,
            "rotation_speed": 0.0,
            "power_consumption": 0.0
        }
        
        sensor_min = SensorData(**min_data)
        assert sensor_min.temperature == 0.0
        assert sensor_min.vibration == 0.0
        
        # Максимальные значения
        max_data = {
            "temperature": 150.0,
            "vibration": 100.0,
            "pressure": 200.0,
            "rotation_speed": 5000.0,
            "power_consumption": 10000.0
        }
        
        sensor_max = SensorData(**max_data)
        assert sensor_max.temperature == 150.0
        assert sensor_max.vibration == 100.0
    
    def test_data_types(self):
        """Тест типов данных"""
        data = {
            "temperature": 85,  # int вместо float
            "vibration": 42.3,
            "pressure": 120.8,
            "rotation_speed": 2450,
            "power_consumption": 1250.5
        }
        
        sensor_data = SensorData(**data)
        assert isinstance(sensor_data.temperature, float)
        assert isinstance(sensor_data.rotation_speed, float)


class TestEquipmentRequest:
    """Тесты модели EquipmentRequest"""
    
    def test_valid_equipment_request(self):
        """Тест валидного запроса оборудования"""
        sensor_data = SensorData(
            temperature=85.5,
            vibration=42.3,
            pressure=120.8,
            rotation_speed=2450.0,
            power_consumption=1250.5
        )
        
        timestamp = datetime.utcnow()
        
        request = EquipmentRequest(
            equipment_id="EQ-001",
            equipment_type=EquipmentType.PUMP,
            timestamp=timestamp,
            sensor_data=sensor_data,
            metadata={"location": "factory-a"}
        )
        
        assert request.equipment_id == "EQ-001"
        assert request.equipment_type == EquipmentType.PUMP
        assert request.timestamp == timestamp
        assert request.metadata["location"] == "factory-a"
    
    def test_invalid_equipment_id_format(self):
        """Тест невалидного формата ID оборудования"""
        sensor_data = SensorData(
            temperature=85.5,
            vibration=42.3,
            pressure=120.8,
            rotation_speed=2450.0,
            power_consumption=1250.5
        )
        
        with pytest.raises(ValueError) as exc_info:
            EquipmentRequest(
                equipment_id="INVALID_ID",  # Не начинается с правильного префикса
                equipment_type=EquipmentType.PUMP,
                timestamp=datetime.utcnow(),
                sensor_data=sensor_data
            )
        
        assert "must start with valid prefix" in str(exc_info.value)
    
    def test_valid_equipment_id_prefixes(self):
        """Тест различных валидных префиксов ID оборудования"""
        sensor_data = SensorData(
            temperature=85.5,
            vibration=42.3,
            pressure=120.8,
            rotation_speed=2450.0,
            power_consumption=1250.5
        )
        
        valid_prefixes = ["EQ-", "PUMP-", "COMP-", "TURB-"]
        
        for prefix in valid_prefixes:
            equipment_id = f"{prefix}001"
            request = EquipmentRequest(
                equipment_id=equipment_id,
                equipment_type=EquipmentType.PUMP,
                timestamp=datetime.utcnow(),
                sensor_data=sensor_data
            )
            assert request.equipment_id == equipment_id
    
    def test_future_timestamp(self):
        """Тест временной метки в будущем"""
        sensor_data = SensorData(
            temperature=85.5,
            vibration=42.3,
            pressure=120.8,
            rotation_speed=2450.0,
            power_consumption=1250.5
        )
        
        future_time = datetime.utcnow() + timedelta(days=1)
        
        with pytest.raises(ValueError) as exc_info:
            EquipmentRequest(
                equipment_id="EQ-001",
                equipment_type=EquipmentType.PUMP,
                timestamp=future_time,
                sensor_data=sensor_data
            )
        
        assert "cannot be in the future" in str(exc_info.value)
    
    def test_old_timestamp(self):
        """Тест слишком старой временной метки"""
        sensor_data = SensorData(
            temperature=85.5,
            vibration=42.3,
            pressure=120.8,
            rotation_speed=2450.0,
            power_consumption=1250.5
        )
        
        old_time = datetime.utcnow() - timedelta(days=8)  # Более 7 дней
        
        with pytest.raises(ValueError) as exc_info:
            EquipmentRequest(
                equipment_id="EQ-001",
                equipment_type=EquipmentType.PUMP,
                timestamp=old_time,
                sensor_data=sensor_data
            )
        
        assert "too old" in str(exc_info.value).lower()
    
    def test_default_timestamp(self):
        """Тест значения по умолчанию для timestamp"""
        sensor_data = SensorData(
            temperature=85.5,
            vibration=42.3,
            pressure=120.8,
            rotation_speed=2450.0,
            power_consumption=1250.5
        )
        
        request = EquipmentRequest(
            equipment_id="EQ-001",
            equipment_type=EquipmentType.PUMP,
            sensor_data=sensor_data
        )
        
        # Проверяем, что timestamp установлен (значение по умолчанию)
        assert request.timestamp is not None
        # Проверяем, что timestamp не в будущем
        assert request.timestamp <= datetime.utcnow()
    
    def test_metadata_optional(self):
        """Тест опциональности metadata"""
        sensor_data = SensorData(
            temperature=85.5,
            vibration=42.3,
            pressure=120.8,
            rotation_speed=2450.0,
            power_consumption=1250.5
        )
        
        # Без metadata
        request_without_metadata = EquipmentRequest(
            equipment_id="EQ-001",
            equipment_type=EquipmentType.PUMP,
            timestamp=datetime.utcnow(),
            sensor_data=sensor_data
        )
        
        assert request_without_metadata.metadata is None
        
        # С metadata
        request_with_metadata = EquipmentRequest(
            equipment_id="EQ-001",
            equipment_type=EquipmentType.PUMP,
            timestamp=datetime.utcnow(),
            sensor_data=sensor_data,
            metadata={"key": "value"}
        )
        
        assert request_with_metadata.metadata == {"key": "value"}


class TestBatchPredictionRequest:
    """Тесты модели BatchPredictionRequest"""
    
    def test_valid_batch_request(self):
        """Тест валидного пакетного запроса"""
        sensor_data = SensorData(
            temperature=85.5,
            vibration=42.3,
            pressure=120.8,
            rotation_speed=2450.0,
            power_consumption=1250.5
        )
        
        requests = [
            EquipmentRequest(
                equipment_id=f"EQ-{i:03d}",
                equipment_type=EquipmentType.PUMP,
                timestamp=datetime.utcnow(),
                sensor_data=sensor_data
            )
            for i in range(3)
        ]
        
        batch_request = BatchPredictionRequest(requests=requests)
        
        assert len(batch_request.requests) == 3
        assert batch_request.requests[0].equipment_id == "EQ-000"
        assert batch_request.requests[2].equipment_id == "EQ-002"
    
    def test_empty_batch(self):
        """Тест пустого пакета"""
        with pytest.raises(ValueError) as exc_info:
            BatchPredictionRequest(requests=[])
        
        assert "at least 1 item" in str(exc_info.value)
    
    def test_batch_size_limit(self):
        """Тест ограничения размера пакета"""
        sensor_data = SensorData(
            temperature=85.5,
            vibration=42.3,
            pressure=120.8,
            rotation_speed=2450.0,
            power_consumption=1250.5
        )
        
        # Создаем 1001 запрос (превышение лимита)
        requests = [
            EquipmentRequest(
                equipment_id=f"EQ-{i:06d}",
                equipment_type=EquipmentType.PUMP,
                timestamp=datetime.utcnow(),
                sensor_data=sensor_data
            )
            for i in range(1001)
        ]
        
        with pytest.raises(ValueError) as exc_info:
            BatchPredictionRequest(requests=requests)
        
        assert "cannot exceed 1000" in str(exc_info.value)
    
    def test_batch_with_different_equipment_types(self):
        """Тест пакета с разными типами оборудования"""
        sensor_data = SensorData(
            temperature=85.5,
            vibration=42.3,
            pressure=120.8,
            rotation_speed=2450.0,
            power_consumption=1250.5
        )
        
        requests = [
            EquipmentRequest(
                equipment_id="EQ-001",
                equipment_type=EquipmentType.PUMP,
                timestamp=datetime.utcnow(),
                sensor_data=sensor_data
            ),
            EquipmentRequest(
                equipment_id="COMP-001",
                equipment_type=EquipmentType.COMPRESSOR,
                timestamp=datetime.utcnow(),
                sensor_data=sensor_data
            ),
            EquipmentRequest(
                equipment_id="TURB-001",
                equipment_type=EquipmentType.TURBINE,
                timestamp=datetime.utcnow(),
                sensor_data=sensor_data
            )
        ]
        
        batch_request = BatchPredictionRequest(requests=requests)
        
        assert len(batch_request.requests) == 3
        assert batch_request.requests[0].equipment_type == EquipmentType.PUMP
        assert batch_request.requests[1].equipment_type == EquipmentType.COMPRESSOR
        assert batch_request.requests[2].equipment_type == EquipmentType.TURBINE


class TestPredictionResponse:
    """Тесты модели PredictionResponse"""
    
    def test_valid_prediction_response(self):
        """Тест валидного ответа предсказания"""
        response = PredictionResponse(
            request_id="req_1234567890",
            equipment_id="EQ-001",
            timestamp=datetime.utcnow(),
            failure_probability=0.8723,
            prediction=True,
            confidence=0.89,
            alert_level=AlertLevel.CRITICAL,
            recommendation="НЕМЕДЛЕННЫЙ РЕМОНТ",
            inference_time_ms=12.45,
            features_used=["temperature", "vibration", "pressure"],
            model_version="1.0.0"
        )
        
        assert response.request_id == "req_1234567890"
        assert response.equipment_id == "EQ-001"
        assert response.failure_probability == 0.8723
        assert response.prediction is True
        assert response.confidence == 0.89
        assert response.alert_level == AlertLevel.CRITICAL
        assert response.recommendation == "НЕМЕДЛЕННЫЙ РЕМОНТ"
        assert response.inference_time_ms == 12.45
        assert response.features_used == ["temperature", "vibration", "pressure"]
        assert response.model_version == "1.0.0"
    
    def test_failure_probability_range(self):
        """Тест диапазона вероятности отказа"""
        # Ниже 0
        with pytest.raises(ValueError) as exc_info:
            PredictionResponse(
                request_id="req_123",
                equipment_id="EQ-001",
                timestamp=datetime.utcnow(),
                failure_probability=-0.1,
                prediction=False,
                confidence=0.5,
                alert_level=AlertLevel.NORMAL,
                recommendation="OK",
                inference_time_ms=10.0,
                features_used=["temp"],
                model_version="1.0"
            )
        
        assert "greater than or equal to 0" in str(exc_info.value)
        
        # Выше 1
        with pytest.raises(ValueError) as exc_info:
            PredictionResponse(
                request_id="req_123",
                equipment_id="EQ-001",
                timestamp=datetime.utcnow(),
                failure_probability=1.1,
                prediction=True,
                confidence=0.5,
                alert_level=AlertLevel.CRITICAL,
                recommendation="REPAIR",
                inference_time_ms=10.0,
                features_used=["temp"],
                model_version="1.0"
            )
        
        assert "less than or equal to 1" in str(exc_info.value)
    
    def test_confidence_range(self):
        """Тест диапазона уверенности"""
        with pytest.raises(ValueError) as exc_info:
            PredictionResponse(
                request_id="req_123",
                equipment_id="EQ-001",
                timestamp=datetime.utcnow(),
                failure_probability=0.5,
                prediction=True,
                confidence=1.5,  # > 1
                alert_level=AlertLevel.NORMAL,
                recommendation="OK",
                inference_time_ms=10.0,
                features_used=["temp"],
                model_version="1.0"
            )
        
        assert "less than or equal to 1" in str(exc_info.value)
    
    def test_inference_time_positive(self):
        """Тест положительного времени инференса"""
        with pytest.raises(ValueError) as exc_info:
            PredictionResponse(
                request_id="req_123",
                equipment_id="EQ-001",
                timestamp=datetime.utcnow(),
                failure_probability=0.5,
                prediction=True,
                confidence=0.8,
                alert_level=AlertLevel.NORMAL,
                recommendation="OK",
                inference_time_ms=-5.0,  # Отрицательное
                features_used=["temp"],
                model_version="1.0"
            )
        
        assert "greater than or equal to 0" in str(exc_info.value)
    
    def test_alert_levels(self):
        """Тест всех уровней алертов"""
        for alert_level in AlertLevel:
            response = PredictionResponse(
                request_id="req_123",
                equipment_id="EQ-001",
                timestamp=datetime.utcnow(),
                failure_probability=0.5,
                prediction=False,
                confidence=0.8,
                alert_level=alert_level,
                recommendation="OK",
                inference_time_ms=10.0,
                features_used=["temp"],
                model_version="1.0"
            )
            
            assert response.alert_level == alert_level
    
    def test_json_serialization(self):
        """Тест сериализации в JSON"""
        timestamp = datetime.utcnow()
        response = PredictionResponse(
            request_id="req_1234567890",
            equipment_id="EQ-001",
            timestamp=timestamp,
            failure_probability=0.8723,
            prediction=True,
            confidence=0.89,
            alert_level=AlertLevel.CRITICAL,
            recommendation="НЕМЕДЛЕННЫЙ РЕМОНТ",
            inference_time_ms=12.45,
            features_used=["temperature", "vibration", "pressure"],
            model_version="1.0.0"
        )
        
        # Конвертация в dict
        response_dict = response.dict()
        
        assert response_dict["request_id"] == "req_1234567890"
        assert response_dict["equipment_id"] == "EQ-001"
        # Проверяем, что timestamp сериализован в ISO формат
        assert "T" in response_dict["timestamp"]  # ISO формат содержит 'T'
        
        # Конвертация в JSON
        response_json = response.json()
        import json
        parsed = json.loads(response_json)
        
        assert parsed["request_id"] == "req_1234567890"
        assert parsed["failure_probability"] == 0.8723
        assert parsed["prediction"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])