#!/bin/bash
set -e

echo "Starting Predictive Maintenance API..."

# Ожидание доступности базы данных
if [ -n "${POSTGRES_HOST}" ]; then
    echo "Waiting for PostgreSQL at ${POSTGRES_HOST}:${POSTGRES_PORT}..."
    while ! nc -z ${POSTGRES_HOST} ${POSTGRES_PORT}; do
        sleep 1
    done
    echo "PostgreSQL is available!"
fi

# Ожидание доступности Redis
if [ -n "${REDIS_HOST}" ]; then
    echo "Waiting for Redis at ${REDIS_HOST}:${REDIS_PORT}..."
    while ! nc -z ${REDIS_HOST} ${REDIS_PORT}; do
        sleep 1
    done
    echo "Redis is available!"
fi

# Ожидание доступности Kafka
if [ -n "${KAFKA_BOOTSTRAP_SERVERS}" ]; then
    echo "Waiting for Kafka..."
    # Упрощенная проверка Kafka
    sleep 10
    echo "Kafka is presumed available (simplified check)"
fi

# Создание директорий если не существуют
mkdir -p /app/logs
mkdir -p /app/models

# Загрузка модели если указан URL
if [ -n "${MODEL_DOWNLOAD_URL}" ]; then
    echo "Downloading model from ${MODEL_DOWNLOAD_URL}..."
    curl -L "${MODEL_DOWNLOAD_URL}" -o /app/models/final_model.pkl
    echo "Model downloaded successfully"
fi

if [ -n "${SCALER_DOWNLOAD_URL}" ]; then
    echo "Downloading scaler from ${SCALER_DOWNLOAD_URL}..."
    curl -L "${SCALER_DOWNLOAD_URL}" -o /app/models/scaler.pkl
    echo "Scaler downloaded successfully"
fi

# Проверка наличия моделей
if [ ! -f "/app/models/final_model.pkl" ]; then
    echo "WARNING: Model file not found at /app/models/final_model.pkl"
    echo "The API will start but predictions will fail until model is loaded"
fi

# Запуск миграций базы данных если необходимо
if [ "${RUN_MIGRATIONS}" = "true" ]; then
    echo "Running database migrations..."
    python -c "
from alembic.config import Config
from alembic import command
alembic_cfg = Config('alembic.ini')
command.upgrade(alembic_cfg, 'head')
"
    echo "Migrations completed"
fi

# Запуск приложения
echo "Starting API server..."
exec "$@"