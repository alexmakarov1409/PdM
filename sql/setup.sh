#!/bin/bash
# setup.sh - Скрипт для настройки базы данных

echo "=== Настройка базы данных Predictive Maintenance ==="

# Переменные окружения
DB_NAME="predictive_maintenance"
DB_USER="postgres"
DB_PASSWORD="your_password"
DB_HOST="localhost"
DB_PORT="5432"

# Проверка наличия psql
if ! command -v psql &> /dev/null; then
    echo "Ошибка: psql не найден. Установите PostgreSQL."
    exit 1
fi

# Экспорт переменных окружения
export PGPASSWORD=$DB_PASSWORD

echo "1. Создание базы данных..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -c "CREATE DATABASE $DB_NAME;"

echo "2. Запуск schema.sql..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f sql/schema.sql

echo "3. Запуск layers.sql..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f sql/layers.sql

echo "4. Запуск functions.sql..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f sql/functions.sql

echo "5. Создание тестовых данных..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "
-- Вставка тестовых данных
INSERT INTO raw.equipment_data (udi, product_id, equipment_type, air_temperature_k, process_temperature_k, rotational_speed_rpm, torque_nm, tool_wear_min, target, failure_type) VALUES
('test_001', 'M14860', 'M', 298.1, 308.6, 1551, 42.8, 0, 0, 'No Failure'),
('test_002', 'L47181', 'L', 298.2, 308.7, 1408, 46.3, 3, 0, 'No Failure'),
('test_003', 'H29643', 'H', 298.3, 308.8, 1498, 49.4, 5, 1, 'Heat Dissipation Failure');

-- Запуск процедур очистки и генерации признаков
CALL cleaned.clean_and_validate_data();
CALL features.generate_features();
"

echo "6. Проверка установки..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "
SELECT 'Слои данных:' as check_name, COUNT(*) as count FROM information_schema.schemata WHERE schema_name IN ('raw', 'cleaned', 'features', 'models')
UNION ALL
SELECT 'Таблицы в raw', COUNT(*) FROM information_schema.tables WHERE table_schema = 'raw'
UNION ALL
SELECT 'Таблицы в cleaned', COUNT(*) FROM information_schema.tables WHERE table_schema = 'cleaned';
"

echo "=== Настройка завершена ==="
echo "База данных: $DB_NAME"
echo "Хост: $DB_HOST:$DB_PORT"
echo "Пользователь: $DB_USER"