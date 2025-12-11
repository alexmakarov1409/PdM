# Многостадийный Dockerfile для FastAPI приложения
# Stage 1: Сборка зависимостей
FROM python:3.11-slim AS builder

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей
COPY requirements/prod.txt requirements.txt

# Создание виртуального окружения и установка зависимостей
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Stage 2: Сборка приложения
FROM python:3.11-slim AS app-builder

WORKDIR /app

# Копирование виртуального окружения из builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Копирование исходного кода
COPY api/ ./api/
COPY models/ ./models/
COPY deployment/ ./deployment/

# Создание пользователя для безопасности
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app


# Stage 3: Production образ
FROM python:3.11-slim

# Метаданные образа
LABEL maintainer="ai-team@company.com"
LABEL version="1.0.0"
LABEL description="Predictive Maintenance API"
LABEL org.opencontainers.image.source="https://github.com/company/predictive-maintenance"

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Копирование виртуального окружения из builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Копирование приложения из app-builder
COPY --from=app-builder --chown=appuser:appuser /app /app

# Копирование скриптов
COPY docker/api/entrypoint.sh /entrypoint.sh
COPY docker/api/gunicorn_conf.py /gunicorn_conf.py

# Настройка прав
RUN chmod +x /entrypoint.sh

# Создание директорий для логов
RUN mkdir -p /app/logs && chown -R appuser:appuser /app/logs

# Создание директории для моделей если не существует
RUN mkdir -p /app/models

# Переключение на непривилегированного пользователя
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Открытие порта
EXPOSE 8000

# Точка входа
ENTRYPOINT ["/entrypoint.sh"]

# Команда запуска
CMD ["gunicorn", "--config", "/gunicorn_conf.py", "api.main:app"]