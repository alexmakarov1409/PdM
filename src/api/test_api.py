import pytest
from fastapi.testclient import TestClient
import numpy as np
from datetime import datetime
import json
from main import app

client = TestClient(app)

def test_health_check():
    """Тест health check эндпоинта"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_root_endpoint():
    """Тест корневого эндпоинта"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Predictive Maintenance API" in data

def test_single_prediction():
    """Тест одиночного предсказания"""
    test_data = {
        "equipment_id": "EQ-TEST-001",
        "timestamp": "2024-01-15T10:30:00",
        "temperature": 85.5,
        "vibration": 45.2,
        "pressure": 120.8,
        "rotation_speed": 2450.0,
        "power_consumption": 1250.5
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    
    # Проверяем структуру ответа
    assert "failure_probability" in data
    assert "prediction" in data
    assert "recommendation" in data
    assert "inference_time_ms" in data
    
    # Проверяем типы данных
    assert isinstance(data["failure_probability"], float)
    assert data["failure_probability"] >= 0 and data["failure_probability"] <= 1
    assert data["inference_time_ms"] > 0

def test_batch_prediction():
    """Тест пакетного предсказания"""
    test_data = {
        "data": [
            {
                "equipment_id": "EQ-TEST-001",
                "timestamp": "2024-01-15T10:30:00",
                "temperature": 85.5,
                "vibration": 45.2,
                "pressure": 120.8,
                "rotation_speed": 2450.0,
                "power_consumption": 1250.5
            },
            {
                "equipment_id": "EQ-TEST-002",
                "timestamp": "2024-01-15T11:30:00",
                "temperature": 75.0,
                "vibration": 35.0,
                "pressure": 110.5,
                "rotation_speed": 2300.0,
                "power_consumption": 1150.0
            }
        ]
    }
    
    response = client.post("/predict/batch", json=test_data)
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["equipment_id"] == "EQ-TEST-001"
    assert data[1]["equipment_id"] == "EQ-TEST-002"

def test_invalid_data():
    """Тест на невалидные данные"""
    invalid_data = {
        "equipment_id": "EQ-TEST-001",
        "timestamp": "invalid-date",
        "temperature": 1000,  # Превышение допустимого значения
        "vibration": 45.2,
        "pressure": 120.8,
        "rotation_speed": 2450.0,
        "power_consumption": 1250.5
    }
    
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error

def test_metrics_endpoint():
    """Тест эндпоинта метрик"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    
    # Проверяем наличие всех метрик
    assert "recall" in data
    assert "fpr" in data
    assert "accuracy" in data
    assert "roc_auc" in data
    
    # Проверяем диапазоны значений
    assert 0 <= data["recall"] <= 1
    assert 0 <= data["fpr"] <= 1

if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v"])