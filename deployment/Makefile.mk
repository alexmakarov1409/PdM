# Makefile для управления Predictive Maintenance системой

.PHONY: help build up down logs restart clean test migrate backup

# Цвета для вывода
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
WHITE  := $(shell tput -Txterm setaf 7)
RESET  := $(shell tput -Txterm sgr0)

# Переменные
COMPOSE := docker-compose
COMPOSE_FILE := docker-compose.yml
ENV_FILE := .env

help: ## Показать эту справку
	@echo '${YELLOW}Predictive Maintenance System Management${RESET}'
	@echo '${GREEN}Usage:${RESET}'
	@echo '  make ${YELLOW}<target>${RESET}'
	@echo ''
	@echo '${GREEN}Targets:${RESET}'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  ${YELLOW}%-20s${RESET} %s\n", $$1, $$2}' $(MAKEFILE_LIST)

init: ## Инициализация проекта (копирование .env файла)
	@if [ ! -f .env ]; then \
		echo "Copying .env.example to .env..."; \
		cp .env.example .env; \
		echo "${GREEN}Please edit .env file with your configuration${RESET}"; \
	else \
		echo "${YELLOW}.env file already exists${RESET}"; \
	fi

build: ## Собрать Docker образы
	@echo "${GREEN}Building Docker images...${RESET}"
	@$(COMPOSE) -f $(COMPOSE_FILE) build --parallel

up: ## Запустить все сервисы в фоновом режиме
	@echo "${GREEN}Starting all services...${RESET}"
	@$(COMPOSE) -f $(COMPOSE_FILE) up -d

up-dev: ## Запустить сервисы для разработки
	@echo "${GREEN}Starting development services...${RESET}"
	@$(COMPOSE) -f $(COMPOSE_FILE) -f docker-compose.override.yml up -d

down: ## Остановить все сервисы
	@echo "${YELLOW}Stopping all services...${RESET}"
	@$(COMPOSE) -f $(COMPOSE_FILE) down

down-clean: ## Остановить и удалить volumes
	@echo "${YELLOW}Stopping services and removing volumes...${RESET}"
	@$(COMPOSE) -f $(COMPOSE_FILE) down -v

logs: ## Показать логи всех сервисов
	@$(COMPOSE) -f $(COMPOSE_FILE) logs -f

logs-api: ## Показать логи API
	@$(COMPOSE) -f $(COMPOSE_FILE) logs -f api

logs-db: ## Показать логи базы данных
	@$(COMPOSE) -f $(COMPOSE_FILE) logs -f postgres

restart: ## Перезапустить все сервисы
	@echo "${YELLOW}Restarting all services...${RESET}"
	@$(COMPOSE) -f $(COMPOSE_FILE) restart

restart-api: ## Перезапустить API сервис
	@echo "${YELLOW}Restarting API service...${RESET}"
	@$(COMPOSE) -f $(COMPOSE_FILE) restart api

clean: ## Очистить ненужные Docker ресурсы
	@echo "${YELLOW}Cleaning up Docker resources...${RESET}"
	@docker system prune -f

test: ## Запустить тесты
	@echo "${GREEN}Running tests...${RESET}"
	@$(COMPOSE) -f $(COMPOSE_FILE) exec api pytest /app/tests -v

test-coverage: ## Запустить тесты с покрытием
	@echo "${GREEN}Running tests with coverage...${RESET}"
	@$(COMPOSE) -f $(COMPOSE_FILE) exec api pytest /app/tests -v --cov=/app/api --cov-report=html

migrate: ## Выполнить миграции базы данных
	@echo "${GREEN}Running database migrations...${RESET}"
	@$(COMPOSE) -f $(COMPOSE_FILE) exec api python -c "
from alembic.config import Config
from alembic import command
alembic_cfg = Config('alembic.ini')
command.upgrade(alembic_cfg, 'head')
"

backup: ## Создать резервную копию базы данных
	@echo "${GREEN}Creating database backup...${RESET}"
	@mkdir -p backups
	@$(COMPOSE) -f $(COMPOSE_FILE) exec postgres pg_dump -U postgres predictive_maintenance > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "${GREEN}Backup created in backups/ directory${RESET}"

restore: ## Восстановить базу данных из резервной копии
	@if [ -z "$(FILE)" ]; then \
		echo "${RED}Usage: make restore FILE=backups/backup_YYYYMMDD_HHMMSS.sql${RESET}"; \
		exit 1; \
	fi
	@echo "${YELLOW}Restoring database from $(FILE)...${RESET}"
	@$(COMPOSE) -f $(COMPOSE_FILE) exec -T postgres psql -U postgres -d predictive_maintenance < $(FILE)

status: ## Показать статус всех сервисов
	@echo "${GREEN}Service status:${RESET}"
	@$(COMPOSE) -f $(COMPOSE_FILE) ps

shell-api: ## Открыть shell в API контейнере
	@$(COMPOSE) -f $(COMPOSE_FILE) exec api /bin/bash

shell-db: ## Открыть shell в PostgreSQL контейнере
	@$(COMPOSE) -f $(COMPOSE_FILE) exec postgres psql -U postgres -d predictive_maintenance

monitor: ## Открыть мониторинг в браузере
	@echo "${GREEN}Opening monitoring dashboards...${RESET}"
	@echo "API Docs: http://localhost:8000/docs"
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"
	@echo "Kafka UI: http://localhost:8080"
	@echo "pgAdmin: http://localhost:5050"

deploy: build up migrate ## Полное развертывание (build + up + migrate)
	@echo "${GREEN}Deployment completed!${RESET}"
	@make monitor

production: ## Развертывание в production режиме
	@echo "${GREEN}Deploying in production mode...${RESET}"
	@$(COMPOSE) -f $(COMPOSE_FILE) -f docker-compose.prod.yml up -d --build

scale-api: ## Масштабировать API сервис
	@if [ -z "$(REPLICAS)" ]; then \
		echo "${RED}Usage: make scale-api REPLICAS=3${RESET}"; \
		exit 1; \
	fi
	@echo "${GREEN}Scaling API to $(REPLICAS) replicas...${RESET}"
	@$(COMPOSE) -f $(COMPOSE_FILE) up -d --scale api=$(REPLICAS) --no-recreate api

# Специальные цели для разработки
dev: up-dev migrate ## Запустить среду разработки
	@echo "${GREEN}Development environment ready!${RESET}"
	@make monitor

format: ## Форматировать код
	@echo "${GREEN}Formatting code...${RESET}"
	@$(COMPOSE) -f $(COMPOSE_FILE) exec api black /app/api
	@$(COMPOSE) -f $(COMPOSE_FILE) exec api isort /app/api

lint: ## Проверить код линтером
	@echo "${GREEN}Linting code...${RESET}"
	@$(COMPOSE) -f $(COMPOSE_FILE) exec api flake8 /app/api
	@$(COMPOSE) -f $(COMPOSE_FILE) exec api mypy /app/api

security: ## Проверить безопасность зависимостей
	@echo "${GREEN}Checking dependencies security...${RESET}"
	@$(COMPOSE) -f $(COMPOSE_FILE) exec api safety check

benchmark: ## Запустить нагрузочное тестирование
	@echo "${GREEN}Running benchmark...${RESET}"
	@$(COMPOSE) -f $(COMPOSE_FILE) exec api python -c "
import asyncio
import httpx
import time

async def make_request(client):
    try:
        response = await client.post(
            'http://api:8000/api/v1/predict',
            json={
                'equipment_id': 'EQ-001',
                'equipment_type': 'pump',
                'timestamp': '2024-01-15T10:30:00',
                'sensor_data': {
                    'temperature': 85.5,
                    'vibration': 42.3,
                    'pressure': 120.8,
                    'rotation_speed': 2450.0,
                    'power_consumption': 1250.5
                }
            },
            headers={'X-API-Key': 'test-api-key-123'}
        )
        return response.status_code == 200
    except:
        return False

async def main():
    start_time = time.time()
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [make_request(client) for _ in range(100)]
        results = await asyncio.gather(*tasks)
    
    success = sum(results)
    total_time = time.time() - start_time
    print(f'Requests: 100')
    print(f'Successful: {success}')
    print(f'Failed: {100 - success}')
    print(f'Total time: {total_time:.2f}s')
    print(f'Requests per second: {100/total_time:.2f}')
    print(f'Success rate: {success}%')

asyncio.run(main())
"

# Алиасы
start: up
stop: down
update: build restart
ps: status
db: shell-db
api: shell-api