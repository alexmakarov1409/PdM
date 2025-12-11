"""
Конфигурация тестов pytest
"""

import pytest
import asyncio
from typing import Generator, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from api.main import app
from api.core.config import Settings
from api.services.model_service import ModelService


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
        API_KEYS=["test-api-key"]
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
def mock_model_service() -> ModelService:
    """Мок сервиса модели"""
    service = AsyncMock(spec=ModelService)
    
    # Настройка возвращаемых значений
    service.initialize = AsyncMock()
    service.cleanup = AsyncMock()
    service.get_model_info = AsyncMock(return_value={
        "loaded": True,
        "version": "1.0.0",
        "name": "test_model",
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
    from datetime import datetime
    
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


@pytest.fixture
def test_database():
    """Фикстура тестовой базы данных"""
    import tempfile
    import os
    
    # Создаем временный файл базы данных
    db_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    db_path = db_file.name
    db_file.close()
    
    # URL для тестовой БД
    db_url = f"sqlite:///{db_path}"
    
    yield db_url
    
    # Удаляем временный файл
    try:
        os.unlink(db_path)
    except:
        pass


# Настройка отображения прогресса тестов
def pytest_runtest_logreport(report):
    """Логирование прогресса тестов"""
    if report.when == 'call':
        if report.passed:
            print(f"✓ {report.nodeid}")
        elif report.failed:
            print(f"✗ {report.nodeid}")
            if report.longrepr:
                print(report.longreprtext)