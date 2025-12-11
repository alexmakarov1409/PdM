"""
Unit тесты для сервисного слоя
"""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from api.services.model_service import ModelService
from api.services.validation_service import ValidationService
from api.services.feature_service import FeatureService
from api.models import EquipmentRequest, SensorData, EquipmentType
from api.core.exceptions import ModelNotLoadedError, ValidationError


class TestModelService:
    """Тесты сервиса модели"""
    
    @pytest.fixture
    def mock_model(self):
        """Фикстура мок модели"""
        model = Mock()
        model.predict_proba.return_value = np.array([[0.2, 0.8]])  # [non-failure, failure]
        return model
    
    @pytest.fixture
    def mock_scaler(self):
        """Фикстура мок скалера"""
        scaler = Mock()
        scaler.transform.return_value = np.array([[0.5, 0.3, 0.7]])
        return scaler
    
    @pytest.fixture
    def mock_redis(self):
        """Фикстура мок Redis"""
        redis = AsyncMock()
        redis.get.return_value = None
        redis.setex.return_value = True
        return redis
    
    @pytest.fixture
    def model_service(self, mock_model, mock_scaler, mock_redis):
        """Фикстура сервиса модели с моками"""
        service = ModelService(
            model_path="test_model.pkl",
            scaler_path="test_scaler.pkl",
            feature_store_url="test_url",
            redis_url="redis://test"
        )
        service.model = mock_model
        service.scaler = mock_scaler
        service.redis_client = mock_redis
        return service
    
    def test_initialization_without_model(self):
        """Тест инициализации без модели"""
        service = ModelService(
            model_path="nonexistent.pkl",
            scaler_path="nonexistent.pkl"
        )
        
        assert service.model is None
        assert service.scaler is None
        assert not service.is_loaded
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, model_service, mock_model, mock_scaler):
        """Тест успешной инициализации"""
        with patch('joblib.load') as mock_load:
            mock_load.side_effect = [mock_model, mock_scaler]
            
            await model_service.initialize()
            
            assert model_service.model is not None
            assert model_service.scaler is not None
            assert model_service.is_loaded
            mock_load.assert_any_call("test_model.pkl")
            mock_load.assert_any_call("test_scaler.pkl")
    
    @pytest.mark.asyncio
    async def test_initialize_failure(self, model_service):
        """Тест неудачной инициализации"""
        with patch('joblib.load', side_effect=Exception("File not found")):
            await model_service.initialize()
            
            assert model_service.model is None
            assert model_service.scaler is None
            assert not model_service.is_loaded
    
    @pytest.mark.asyncio
    async def test_predict_success(self, model_service):
        """Тест успешного предсказания"""
        # Подготовка тестового запроса
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
            timestamp=datetime.utcnow(),
            sensor_data=sensor_data
        )
        
        # Мок feature service
        mock_feature_service = AsyncMock()
        mock_feature_service.get_features_for_inference.return_value = {
            'features': {'feature1': 0.5, 'feature2': 0.3, 'feature3': 0.7},
            'source': 'test'
        }
        
        model_service.feature_service = mock_feature_service
        
        # Выполнение предсказания
        result = await model_service.predict(request)
        
        # Проверки
        assert result.failure_probability == 0.8  # Из predict_proba
        assert result.prediction is True  # Порог по умолчанию 0.5
        assert result.confidence == 0.8
        assert result.model_version == "1.0.0"
        
        # Проверка вызовов
        mock_feature_service.get_features_for_inference.assert_called_once()
        model_service.model.predict_proba.assert_called_once()
        model_service.scaler.transform.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_predict_model_not_loaded(self, model_service):
        """Тест предсказания когда модель не загружена"""
        model_service.model = None
        model_service.is_loaded = False
        
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
            timestamp=datetime.utcnow(),
            sensor_data=sensor_data
        )
        
        with pytest.raises(ModelNotLoadedError):
            await model_service.predict(request)
    
    @pytest.mark.asyncio
    async def test_predict_with_custom_threshold(self, model_service):
        """Тест предсказания с кастомным порогом"""
        # Устанавливаем порог 0.9
        model_service.threshold = 0.9
        
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
            timestamp=datetime.utcnow(),
            sensor_data=sensor_data
        )
        
        mock_feature_service = AsyncMock()
        mock_feature_service.get_features_for_inference.return_value = {
            'features': {'feature1': 0.5, 'feature2': 0.3, 'feature3': 0.7},
            'source': 'test'
        }
        
        model_service.feature_service = mock_feature_service
        model_service.model.predict_proba.return_value = np.array([[0.2, 0.8]])  # 0.8 < 0.9
        
        result = await model_service.predict(request)
        
        # С порогом 0.9, вероятность 0.8 должна давать prediction=False
        assert result.failure_probability == 0.8
        assert result.prediction is False
        assert result.confidence == 0.2  # 1 - 0.8
    
    @pytest.mark.asyncio
    async def test_batch_predict(self, model_service):
        """Тест пакетного предсказания"""
        # Подготовка тестовых запросов
        requests = []
        for i in range(3):
            sensor_data = SensorData(
                temperature=85.5 + i,
                vibration=42.3 + i,
                pressure=120.8 + i,
                rotation_speed=2450.0 + i * 10,
                power_consumption=1250.5 + i * 10
            )
            
            request = EquipmentRequest(
                equipment_id=f"EQ-{i:03d}",
                equipment_type=EquipmentType.PUMP,
                timestamp=datetime.utcnow(),
                sensor_data=sensor_data
            )
            requests.append(request)
        
        # Моки
        mock_feature_service = AsyncMock()
        mock_feature_service.get_features_for_inference.return_value = {
            'features': {'feature1': 0.5, 'feature2': 0.3, 'feature3': 0.7},
            'source': 'test'
        }
        
        model_service.feature_service = mock_feature_service
        
        # Разные вероятности для разных запросов
        probabilities = [np.array([[0.3, 0.7]]), np.array([[0.1, 0.9]]), np.array([[0.5, 0.5]])]
        model_service.model.predict_proba.side_effect = probabilities
        
        # Выполнение пакетного предсказания
        results = await model_service.batch_predict(requests)
        
        # Проверки
        assert len(results) == 3
        assert results[0].failure_probability == 0.7
        assert results[0].prediction is True
        assert results[1].failure_probability == 0.9
        assert results[1].prediction is True
        assert results[2].failure_probability == 0.5
        assert results[2].prediction is True  # Порог 0.5, равно порогу = True
        
        # Проверка количества вызовов
        assert mock_feature_service.get_features_for_inference.call_count == 3
        assert model_service.model.predict_proba.call_count == 3
    
    def test_get_model_info_loaded(self, model_service):
        """Тест получения информации о загруженной модели"""
        model_service.model = Mock()
        model_service.model.__class__.__name__ = "RandomForestClassifier"
        model_service.is_loaded = True
        
        info = model_service.get_model_info()
        
        assert info["loaded"] is True
        assert info["model_type"] == "RandomForestClassifier"
        assert info["version"] == "1.0.0"
    
    def test_get_model_info_not_loaded(self, model_service):
        """Тест получения информации о незагруженной модели"""
        model_service.model = None
        model_service.is_loaded = False
        
        info = model_service.get_model_info()
        
        assert info["loaded"] is False
        assert "error" in info
    
    @pytest.mark.asyncio
    async def test_cache_prediction(self, model_service, mock_redis):
        """Тест кэширования предсказания"""
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
            timestamp=datetime.utcnow(),
            sensor_data=sensor_data
        )
        
        # Мок feature service
        mock_feature_service = AsyncMock()
        mock_feature_service.get_features_for_inference.return_value = {
            'features': {'feature1': 0.5, 'feature2': 0.3, 'feature3': 0.7},
            'source': 'test'
        }
        
        model_service.feature_service = mock_feature_service
        
        # Первый вызов - кэш пустой
        mock_redis.get.return_value = None
        await model_service.predict(request, use_cache=True)
        
        # Должен быть запрос в Redis
        mock_redis.get.assert_called_once()
        # Должен быть вызов модели
        model_service.model.predict_proba.assert_called_once()
        # Должен быть сохранен в кэш
        mock_redis.setex.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_use_cached_prediction(self, model_service, mock_redis):
        """Тест использования кэшированного предсказания"""
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
            timestamp=datetime.utcnow(),
            sensor_data=sensor_data
        )
        
        # Мок кэшированного значения
        cached_data = {
            'failure_probability': 0.75,
            'prediction': True,
            'confidence': 0.75,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        mock_redis.get.return_value = json.dumps(cached_data)  # Нужно импортировать json
        
        result = await model_service.predict(request, use_cache=True)
        
        # Должен вернуть кэшированное значение
        assert result.failure_probability == 0.75
        assert result.prediction is True
        # Модель не должна вызываться
        model_service.model.predict_proba.assert_not_called()


class TestValidationService:
    """Тесты сервиса валидации"""
    
    @pytest.fixture
    def validation_service(self):
        """Фикстура сервиса валидации"""
        return ValidationService()
    
    @pytest.mark.asyncio
    async def test_validate_request_valid(self, validation_service):
        """Тест валидации валидного запроса"""
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
            timestamp=datetime.utcnow(),
            sensor_data=sensor_data
        )
        
        result = await validation_service.validate_request(request)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.warnings == []
    
    @pytest.mark.asyncio
    async def test_validate_request_invalid_sensor_data(self, validation_service):
        """Тест валидации запроса с невалидными данными датчиков"""
        # Датчики вне допустимых диапазонов
        sensor_data = SensorData(
            temperature=200.0,  # > 150
            vibration=120.0,    # > 100
            pressure=250.0,     # > 200
            rotation_speed=2450.0,
            power_consumption=1250.5
        )
        
        request = EquipmentRequest(
            equipment_id="EQ-001",
            equipment_type=EquipmentType.PUMP,
            timestamp=datetime.utcnow(),
            sensor_data=sensor_data
        )
        
        result = await validation_service.validate_request(request)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        # Проверяем, что ошибки содержат информацию о полях
        error_messages = [e["message"] for e in result.errors]
        assert any("temperature" in msg.lower() for msg in error_messages)
        assert any("vibration" in msg.lower() for msg in error_messages)
        assert any("pressure" in msg.lower() for msg in error_messages)
    
    @pytest.mark.asyncio
    async def test_validate_equipment_exists(self, validation_service):
        """Тест проверки существования оборудования"""
        request = EquipmentRequest(
            equipment_id="EQ-NONEXISTENT",
            equipment_type=EquipmentType.PUMP,
            timestamp=datetime.utcnow(),
            sensor_data=SensorData(
                temperature=85.5,
                vibration=42.3,
                pressure=120.8,
                rotation_speed=2450.0,
                power_consumption=1250.5
            )
        )
        
        # Мок базы данных, которая возвращает False (оборудование не найдено)
        with patch.object(validation_service, '_check_equipment_in_db', return_value=False):
            result = await validation_service.validate_request(request)
            
            assert result.is_valid is False
            assert any("not found" in e["message"].lower() for e in result.errors)
    
    @pytest.mark.asyncio
    async def test_validate_historical_consistency(self, validation_service):
        """Тест проверки согласованности с историческими данными"""
        request = EquipmentRequest(
            equipment_id="EQ-001",
            equipment_type=EquipmentType.PUMP,
            timestamp=datetime.utcnow(),
            sensor_data=SensorData(
                temperature=85.5,
                vibration=42.3,
                pressure=120.8,
                rotation_speed=2450.0,
                power_consumption=1250.5
            )
        )
        
        # Мок исторических данных с аномальным значением
        historical_data = {
            'temperature': [80.0, 82.0, 81.5, 150.0],  # Резкий скачок
            'vibration': [40.0, 41.0, 39.5, 42.0],
            'pressure': [120.0, 121.0, 119.5, 120.5]
        }
        
        with patch.object(validation_service, '_get_historical_data', return_value=historical_data):
            result = await validation_service.validate_request(request)
            
            # Должно быть предупреждение об аномалии
            assert result.is_valid is True  # Все еще валидно
            assert len(result.warnings) > 0
            assert any("anomal" in w.lower() for w in result.warnings)
    
    @pytest.mark.asyncio
    async def test_validate_sensor_relationships(self, validation_service):
        """Тест проверки взаимосвязей датчиков"""
        # Нормальная температура с очень низким давлением
        request = EquipmentRequest(
            equipment_id="EQ-001",
            equipment_type=EquipmentType.PUMP,
            timestamp=datetime.utcnow(),
            sensor_data=SensorData(
                temperature=85.5,
                vibration=42.3,
                pressure=30.0,  # Подозрительно низкое
                rotation_speed=2450.0,
                power_consumption=1250.5
            )
        )
        
        result = await validation_service.validate_request(request)
        
        # Должно быть предупреждение о возможной проблеме с датчиком
        assert result.is_valid is True  # Все еще валидно
        assert len(result.warnings) > 0
        assert any("sensor" in w.lower() for w in result.warnings)
        assert any("pressure" in w.lower() for w in result.warnings)
    
    def test_validate_numeric_range(self, validation_service):
        """Тест валидации числового диапазона"""
        # Валидное значение
        assert validation_service._validate_numeric_range(50, 0, 100, "test") is None
        
        # Ниже минимума
        error = validation_service._validate_numeric_range(-10, 0, 100, "test")
        assert error is not None
        assert "less than minimum" in error.lower()
        
        # Выше максимума
        error = validation_service._validate_numeric_range(150, 0, 100, "test")
        assert error is not None
        assert "greater than maximum" in error.lower()
        
        # Граничные значения
        assert validation_service._validate_numeric_range(0, 0, 100, "test") is None
        assert validation_service._validate_numeric_range(100, 0, 100, "test") is None
    
    def test_calculate_statistics(self, validation_service):
        """Тест расчета статистик"""
        data = [10, 20, 30, 40, 50]
        
        stats = validation_service._calculate_statistics(data, "test")
        
        assert stats["mean"] == 30.0
        assert stats["std"] == pytest.approx(14.142, rel=1e-3)
        assert stats["min"] == 10
        assert stats["max"] == 50
        assert stats["count"] == 5
    
    def test_detect_anomalies(self, validation_service):
        """Тест обнаружения аномалий"""
        # Нормальные данные
        normal_data = [10, 11, 10.5, 11.5, 10.8]
        anomalies = validation_service._detect_anomalies(normal_data, "test")
        assert len(anomalies) == 0
        
        # С аномалией
        data_with_anomaly = [10, 11, 10.5, 11.5, 50.0]  # 50.0 - аномалия
        anomalies = validation_service._detect_anomalies(data_with_anomaly, "test")
        assert len(anomalies) == 1
        assert anomalies[0] == 50.0
        
        # Мало данных
        few_data = [10, 11]
        anomalies = validation_service._detect_anomalies(few_data, "test")
        assert len(anomalies) == 0  # Не должно обнаруживать при малом количестве данных


class TestFeatureService:
    """Тесты сервиса признаков"""
    
    @pytest.fixture
    def feature_service(self):
        """Фикстура сервиса признаков"""
        service = FeatureService(
            feature_store_url="test_url",
            redis_url="redis://test"
        )
        return service
    
    @pytest.mark.asyncio
    async def test_get_features_for_inference_success(self, feature_service):
        """Тест успешного получения признаков для инференса"""
        equipment_id = "EQ-001"
        feature_names = ["temperature", "vibration", "pressure"]
        
        # Моки
        mock_online_store = AsyncMock()
        mock_online_store.get_features.return_value = {
            'features': {
                'temperature': 85.5,
                'vibration': 42.3,
                'pressure': 120.8
            },
            'timestamp': datetime.utcnow().isoformat(),
            'fresh': True
        }
        
        mock_offline_store = Mock()
        
        feature_service.online_store = mock_online_store
        feature_service.offline_store = mock_offline_store
        
        result = await feature_service.get_features_for_inference(
            equipment_id, feature_names
        )
        
        assert result['features']['temperature'] == 85.5
        assert result['features']['vibration'] == 42.3
        assert result['features']['pressure'] == 120.8
        assert result['source'] == 'online_store'
        assert result['fresh'] is True
        
        mock_online_store.get_features.assert_called_once_with(
            equipment_id, feature_names
        )
        # При наличии свежих данных в online store, offline store не должен вызываться
        mock_offline_store.get_features_for_training.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_features_stale_online_data(self, feature_service):
        """Тест получения признаков с устаревшими онлайн данными"""
        equipment_id = "EQ-001"
        feature_names = ["temperature", "vibration", "pressure"]
        
        # Моки online store с устаревшими данными
        mock_online_store = AsyncMock()
        mock_online_store.get_features.return_value = {
            'features': {
                'temperature': 85.5,
                'vibration': 42.3,
                'pressure': 120.8
            },
            'timestamp': (datetime.utcnow() - timedelta(hours=2)).isoformat(),
            'fresh': False  # Устарели
        }
        
        # Моки offline store с свежими данными
        mock_offline_store = Mock()
        mock_df = Mock()
        mock_df.empty = False
        mock_df.iloc = Mock()
        mock_df.iloc.__getitem__ = Mock(return_value=Mock(
            __getitem__=Mock(side_effect=lambda x: {
                'temperature': 86.0,
                'vibration': 43.0,
                'pressure': 121.0,
                'timestamp': datetime.utcnow()
            }[x] if x in ['temperature', 'vibration', 'pressure', 'timestamp'] else None)
        ))
        mock_offline_store.get_features_for_training.return_value = mock_df
        
        feature_service.online_store = mock_online_store
        feature_service.offline_store = mock_offline_store
        
        result = await feature_service.get_features_for_inference(
            equipment_id, feature_names
        )
        
        # Должны получить данные из offline store
        assert result['features']['temperature'] == 86.0
        assert result['source'] == 'offline_store'
        
        # Проверяем, что данные были обновлены в online store
        mock_online_store.store_features.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_features_no_data(self, feature_service):
        """Тест получения признаков когда данных нет"""
        equipment_id = "EQ-001"
        feature_names = ["temperature", "vibration", "pressure"]
        
        # Моки без данных
        mock_online_store = AsyncMock()
        mock_online_store.get_features.return_value = {}
        
        mock_offline_store = Mock()
        mock_offline_store.get_features_for_training.return_value = Mock(empty=True)
        
        feature_service.online_store = mock_online_store
        feature_service.offline_store = mock_offline_store
        
        result = await feature_service.get_features_for_inference(
            equipment_id, feature_names
        )
        
        assert result['features'] == {}
        assert result['source'] == 'none'
        assert result['fresh'] is False
    
    @pytest.mark.asyncio
    async def test_get_feature_vector(self, feature_service):
        """Тест получения вектора признаков"""
        equipment_id = "EQ-001"
        feature_names = ["temperature", "vibration", "pressure"]
        
        # Мок получения признаков
        mock_features = {
            'features': {
                'temperature': 85.5,
                'vibration': 42.3,
                'pressure': 120.8
            },
            'source': 'test'
        }
        
        with patch.object(feature_service, 'get_features_for_inference', 
                         return_value=mock_features):
            vector = await feature_service.get_feature_vector(
                equipment_id, feature_names
            )
            
            assert len(vector) == 3
            assert vector == [85.5, 42.3, 120.8]
    
    @pytest.mark.asyncio
    async def test_get_feature_vector_missing_features(self, feature_service):
        """Тест получения вектора признаков с отсутствующими признаками"""
        equipment_id = "EQ-001"
        feature_names = ["temperature", "vibration", "pressure", "missing"]
        
        # Мок с отсутствующим признаком
        mock_features = {
            'features': {
                'temperature': 85.5,
                'vibration': 42.3,
                'pressure': 120.8
                # missing отсутствует
            },
            'source': 'test'
        }
        
        with patch.object(feature_service, 'get_features_for_inference', 
                         return_value=mock_features):
            vector = await feature_service.get_feature_vector(
                equipment_id, feature_names
            )
            
            # Отсутствующий признак должен быть заменен на 0.0
            assert len(vector) == 4
            assert vector == [85.5, 42.3, 120.8, 0.0]
    
    @pytest.mark.asyncio
    async def test_batch_get_features(self, feature_service):
        """Тест пакетного получения признаков"""
        equipment_ids = ["EQ-001", "EQ-002", "EQ-003"]
        feature_names = ["temperature", "vibration"]
        
        # Мок online store для пакетного получения
        mock_online_store = AsyncMock()
        mock_online_store.get_multiple_features.return_value = {
            "EQ-001": {
                'features': {'temperature': 85.5, 'vibration': 42.3},
                'timestamp': datetime.utcnow().isoformat()
            },
            "EQ-002": {
                'features': {'temperature': 75.0, 'vibration': 35.0},
                'timestamp': datetime.utcnow().isoformat()
            }
            # EQ-003 нет в online store
        }
        
        feature_service.online_store = mock_online_store
        
        # Мок для индивидуального получения (для EQ-003)
        with patch.object(feature_service, 'get_features_for_inference', 
                         side_effect=lambda eq_id, features: {
                             'features': {'temperature': 65.0, 'vibration': 25.0},
                             'source': 'offline_store'
                         }):
            results = await feature_service.batch_get_features(
                equipment_ids, feature_names
            )
            
            assert len(results) == 3
            assert "EQ-001" in results
            assert "EQ-002" in results
            assert "EQ-003" in results
            
            # Проверяем источники
            assert results["EQ-001"]["source"] == "online_store"
            assert results["EQ-003"]["source"] == "offline_store"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])