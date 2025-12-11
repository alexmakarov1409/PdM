"""
Конфигурация pytest для всех тестов
"""

import pytest
import asyncio
import tempfile
import os
from typing import Generator, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.main import app
from api.core.config import Settings
from api.database.models import Base
from api.database.session import SessionLocal


def pytest_configure(config):
    """Конфигурация pytest при старте"""
    config.addinivalue_line(
        "markers", "integration: маркер для интеграционных тестов"
    )
    config.addinivalue_line(
        "markers", "slow: маркер для медленных тестов"
    )
    config.addinivalue_line(
        "markers", "database: маркер для тестов базы данных"
    )


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Фикстура для event loop"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Фикстура тестовых настроек"""
    return Settings(
        ENVIRONMENT="testing",
        DEBUG=True,
        MODEL_PATH="tests/test_model.pkl",
        SCALER_PATH="tests/test_scaler.pkl",
        TESTING=True,
        API_KEYS=["test-api-key-123"],
        POSTGRES_URL="sqlite:///:memory:",
        REDIS_URL="redis://localhost:6379/1"
    )


@pytest.fixture
def override_settings(test_settings):
    """Переопределение настроек для тестов"""
    from api.core.config import get_settings
    
    original_settings = get_settings()
    
    # Временно заменяем настройки
    import api.main
    api.main.settings = test_settings
    
    yield test_settings
    
    # Восстанавливаем оригинальные настройки
    api.main.settings = original_settings


@pytest.fixture
def client(override_settings) -> TestClient:
    """Фикстура тестового клиента"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_model_service() -> AsyncMock:
    """Мок сервиса модели"""
    service = AsyncMock()
    
    # Настройка методов
    service.initialize = AsyncMock()
    service.cleanup = AsyncMock()
    
    # Мок предсказания
    mock_prediction = AsyncMock()
    mock_prediction.failure_probability = 0.85
    mock_prediction.prediction = True
    mock_prediction.confidence = 0.89
    mock_prediction.features_used = ["temperature", "vibration", "pressure"]
    mock_prediction.model_version = "1.0.0"
    
    service.predict = AsyncMock(return_value=mock_prediction)
    service.batch_predict = AsyncMock(return_value=[mock_prediction, mock_prediction])
    service.get_model_info = AsyncMock(return_value={
        "loaded": True,
        "version": "1.0.0",
        "model_type": "RandomForestClassifier",
        "metrics": {"accuracy": 0.92, "recall": 0.89}
    })
    service.check_feature_store = AsyncMock(return_value=True)
    
    return service


@pytest.fixture
def patch_model_service(mock_model_service):
    """Патчинг сервиса модели"""
    with patch('api.main.ModelService', return_value=mock_model_service):
        with patch('api.routers.predictions.get_model_service', 
                  return_value=mock_model_service):
            with patch('api.routers.health.get_model_service',
                      return_value=mock_model_service):
                yield mock_model_service


@pytest.fixture
def sample_sensor_data() -> dict:
    """Пример данных датчиков"""
    return {
        "temperature": 85.5,
        "vibration": 42.3,
        "pressure": 120.8,
        "rotation_speed": 2450.0,
        "power_consumption": 1250.5
    }


@pytest.fixture
def sample_prediction_request(sample_sensor_data) -> dict:
    """Пример запроса предсказания"""
    return {
        "equipment_id": "EQ-001",
        "equipment_type": "pump",
        "timestamp": datetime.utcnow().isoformat(),
        "sensor_data": sample_sensor_data,
        "metadata": {
            "location": "test-factory",
            "operator": "test-operator"
        }
    }


@pytest.fixture
def sample_batch_prediction_request(sample_sensor_data) -> dict:
    """Пример пакетного запроса предсказания"""
    return {
        "requests": [
            {
                "equipment_id": "EQ-001",
                "equipment_type": "pump",
                "timestamp": datetime.utcnow().isoformat(),
                "sensor_data": sample_sensor_data
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


@pytest.fixture
def test_database():
    """Фикстура тестовой базы данных"""
    # Создаем временную базу данных в памяти
    engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Создаем таблицы
    Base.metadata.create_all(bind=engine)
    
    yield TestingSessionLocal
    
    # Очищаем
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(test_database):
    """Фикстура сессии базы данных"""
    session = test_database()
    
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def test_model_file():
    """Фикстура тестового файла модели"""
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    
    # Создаем простую тестовую модель
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Сохраняем во временный файл
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        joblib.dump(model, tmp.name)
        model_path = tmp.name
    
    yield model_path
    
    # Удаляем временный файл
    try:
        os.unlink(model_path)
    except:
        pass


@pytest.fixture
def test_scaler_file():
    """Фикстура тестового файла скалера"""
    import joblib
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # Создаем тестовый скалер
    scaler = StandardScaler()
    scaler.fit(np.random.rand(100, 5))
    
    # Сохраняем во временный файл
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        joblib.dump(scaler, tmp.name)
        scaler_path = tmp.name
    
    yield scaler_path
    
    # Удаляем временный файл
    try:
        os.unlink(scaler_path)
    except:
        pass


@pytest.fixture
def mock_redis():
    """Мок Redis клиента"""
    redis_mock = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.setex = AsyncMock(return_value=True)
    redis_mock.ping = AsyncMock(return_value=True)
    redis_mock.delete = AsyncMock(return_value=1)
    
    return redis_mock


@pytest.fixture
def mock_kafka():
    """Мок Kafka продюсера"""
    kafka_mock = AsyncMock()
    kafka_mock.send = AsyncMock()
    kafka_mock.flush = AsyncMock()
    kafka_mock.close = AsyncMock()
    
    return kafka_mock


@pytest.fixture
def mock_feature_store():
    """Мок Feature Store"""
    feature_store_mock = AsyncMock()
    feature_store_mock.get_features_for_inference = AsyncMock(return_value={
        'features': {
            'temperature': 85.5,
            'vibration': 42.3,
            'pressure': 120.8,
            'rotation_speed': 2450.0,
            'power_consumption': 1250.5,
            'hour': 10,
            'day_of_week': 1,
            'month': 1
        },
        'source': 'online_store',
        'fresh': True
    })
    
    return feature_store_mock


@pytest.fixture
def valid_api_key(test_settings) -> str:
    """Валидный API ключ для тестов"""
    return test_settings.API_KEYS[0]


@pytest.fixture
def auth_headers(valid_api_key) -> dict:
    """Заголовки авторизации для тестов"""
    return {"X-API-Key": valid_api_key}


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Очистка тестовых файлов после тестов"""
    yield
    
    import os
    import glob
    
    # Удаляем тестовые файлы моделей
    test_model_files = glob.glob("tests/test_*.pkl")
    for file in test_model_files:
        try:
            os.remove(file)
        except:
            pass
    
    # Удаляем тестовые логи
    test_log_files = glob.glob("tests/test_*.log")
    for file in test_log_files:
        try:
            os.remove(file)
        except:
            pass
    
    # Удаляем временные файлы coverage
    coverage_files = glob.glob(".coverage*")
    for file in coverage_files:
        try:
            os.remove(file)
        except:
            pass


@pytest.fixture
def mock_prometheus_metrics():
    """Мок метрик Prometheus"""
    with patch('api.utils.metrics.record_prediction_time') as mock_record_time:
        with patch('api.utils.metrics.increment_prediction_counter') as mock_increment:
            with patch('api.utils.metrics.increment_error_counter') as mock_error:
                yield {
                    'record_time': mock_record_time,
                    'increment': mock_increment,
                    'error': mock_error
                }


# Настройка отображения прогресса тестов
def pytest_runtest_logreport(report):
    """Логирование прогресса тестов"""
    if report.when == 'call':
        if report.passed:
            print(f"✓ {report.nodeid}")
        elif report.failed:
            print(f"✗ {report.nodeid}")
            if report.longrepr:
                # Ограничиваем вывод ошибки
                error_lines = str(report.longreprtext).split('\n')
                for line in error_lines[:10]:  # Первые 10 строк
                    print(f"  {line}")


# Настройка маркеров
def pytest_collection_modifyitems(config, items):
    """Модификация тестов при сборе"""
    # Помечаем медленные тесты
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "performance" in item.nodeid or "load" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        if "database" in item.nodeid:
            item.add_marker(pytest.mark.database)


# Параметры для параметризованных тестов
def pytest_generate_tests(metafunc):
    """Генерация параметров для тестов"""
    if "alert_level" in metafunc.fixturenames:
        from api.models import AlertLevel
        metafunc.parametrize("alert_level", list(AlertLevel))
    
    if "equipment_type" in metafunc.fixturenames:
        from api.models import EquipmentType
        metafunc.parametrize("equipment_type", list(EquipmentType))