-- schema.sql
-- Создание базы данных и пользователя для проекта предиктивного обслуживания

-- =============================================
-- 1. Создание базы данных (если не существует)
-- =============================================
CREATE DATABASE predictive_maintenance
    WITH 
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1
    IS_TEMPLATE = False;

COMMENT ON DATABASE predictive_maintenance IS 'База данных для системы предиктивного обслуживания оборудования';

-- =============================================
-- 2. Создание схем (schemas) для логического разделения
-- =============================================
CREATE SCHEMA IF NOT EXISTS raw;
COMMENT ON SCHEMA raw IS 'Слой для сырых, неизмененных данных';

CREATE SCHEMA IF NOT EXISTS cleaned;
COMMENT ON SCHEMA cleaned IS 'Слой для очищенных и валидированных данных';

CREATE SCHEMA IF NOT EXISTS features;
COMMENT ON SCHEMA features IS 'Слой с подготовленными признаками для ML';

CREATE SCHEMA IF NOT EXISTS models;
COMMENT ON SCHEMA models IS 'Слой для хранения метаданных моделей и результатов';

CREATE SCHEMA IF NOT EXISTS monitoring;
COMMENT ON SCHEMA monitoring IS 'Слой для мониторинга и метрик системы';

-- =============================================
-- 3. Raw Layer - таблицы для сырых данных
-- =============================================

-- Основная таблица сырых данных
CREATE TABLE raw.equipment_data (
    id BIGSERIAL PRIMARY KEY,
    udi VARCHAR(50) NOT NULL,
    product_id VARCHAR(50) NOT NULL,
    equipment_type VARCHAR(10) NOT NULL,
    air_temperature_k NUMERIC(8, 2) NOT NULL,
    process_temperature_k NUMERIC(8, 2) NOT NULL,
    rotational_speed_rpm INTEGER NOT NULL,
    torque_nm NUMERIC(8, 2) NOT NULL,
    tool_wear_min INTEGER NOT NULL,
    target INTEGER NOT NULL,
    failure_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_system VARCHAR(50),
    batch_id UUID,
    raw_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE raw.equipment_data IS 'Сырые данные с датчиков оборудования';
COMMENT ON COLUMN raw.equipment_data.udi IS 'Уникальный идентификатор записи';
COMMENT ON COLUMN raw.equipment_data.product_id IS 'Идентификатор продукта';
COMMENT ON COLUMN raw.equipment_data.equipment_type IS 'Тип оборудования (L, M, H)';
COMMENT ON COLUMN raw.equipment_data.batch_id IS 'Идентификатор пакетной загрузки';

-- Индексы для ускорения запросов
CREATE INDEX idx_raw_equipment_udi ON raw.equipment_data(udi);
CREATE INDEX idx_raw_equipment_timestamp ON raw.equipment_data(timestamp);
CREATE INDEX idx_raw_equipment_batch_id ON raw.equipment_data(batch_id);

-- Таблица метаданных загрузок
CREATE TABLE raw.data_imports (
    id BIGSERIAL PRIMARY KEY,
    file_name VARCHAR(255),
    file_size BIGINT,
    records_count INTEGER,
    status VARCHAR(20) DEFAULT 'pending',
    error_message TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    created_by VARCHAR(50)
);

-- =============================================
-- 4. Cleaned Layer - таблицы для очищенных данных
-- =============================================

-- Основная таблица очищенных данных
CREATE TABLE cleaned.equipment_cleaned (
    equipment_id VARCHAR(50) PRIMARY KEY,
    udi VARCHAR(50) NOT NULL,
    product_id VARCHAR(50) NOT NULL,
    equipment_type VARCHAR(10) NOT NULL,
    air_temperature_k NUMERIC(8, 2) NOT NULL,
    process_temperature_k NUMERIC(8, 2) NOT NULL,
    rotational_speed_rpm INTEGER NOT NULL,
    torque_nm NUMERIC(8, 2) NOT NULL,
    tool_wear_min INTEGER NOT NULL,
    target INTEGER NOT NULL,
    failure_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    -- Валидационные флаги
    is_valid BOOLEAN DEFAULT TRUE,
    validation_errors JSONB,
    
    -- Статистические метрики
    z_score_air_temp NUMERIC(8, 2),
    z_score_process_temp NUMERIC(8, 2),
    z_score_torque NUMERIC(8, 2),
    
    -- Метаданные
    cleaning_version INTEGER DEFAULT 1,
    cleaned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT chk_air_temp_range CHECK (air_temperature_k BETWEEN 290 AND 310),
    CONSTRAINT chk_process_temp_range CHECK (process_temperature_k BETWEEN 305 AND 320),
    CONSTRAINT chk_rotational_speed_range CHECK (rotational_speed_rpm BETWEEN 1000 AND 3000),
    CONSTRAINT chk_torque_range CHECK (torque_nm BETWEEN 0 AND 100),
    CONSTRAINT chk_tool_wear_range CHECK (tool_wear_min BETWEEN 0 AND 300),
    CONSTRAINT chk_target_range CHECK (target IN (0, 1))
);

COMMENT ON TABLE cleaned.equipment_cleaned IS 'Очищенные и валидированные данные оборудования';

-- Индексы
CREATE INDEX idx_cleaned_equipment_type ON cleaned.equipment_cleaned(equipment_type);
CREATE INDEX idx_cleaned_timestamp ON cleaned.equipment_cleaned(timestamp);
CREATE INDEX idx_cleaned_target ON cleaned.equipment_cleaned(target);

-- Таблица истории очистки
CREATE TABLE cleaned.cleaning_history (
    id BIGSERIAL PRIMARY KEY,
    cleaning_date DATE DEFAULT CURRENT_DATE,
    records_processed INTEGER,
    records_validated INTEGER,
    records_invalidated INTEGER,
    cleaning_rules JSONB,
    cleaning_duration INTERVAL,
    created_by VARCHAR(50)
);

-- =============================================
-- 5. Features Layer - таблицы с признаками для ML
-- =============================================

-- Основная таблица признаков
CREATE TABLE features.equipment_features (
    features_id BIGSERIAL PRIMARY KEY,
    equipment_id VARCHAR(50) NOT NULL,
    feature_date DATE NOT NULL,
    
    -- Базовые признаки
    equipment_type_encoded INTEGER,
    air_temp_normalized NUMERIC(8, 4),
    process_temp_normalized NUMERIC(8, 4),
    rotational_speed_normalized NUMERIC(8, 4),
    torque_normalized NUMERIC(8, 4),
    tool_wear_normalized NUMERIC(8, 4),
    
    -- Инженерные признаки
    temperature_difference NUMERIC(8, 4),
    torque_speed_ratio NUMERIC(8, 4),
    power_estimate NUMERIC(10, 4),
    wear_category VARCHAR(20),
    
    -- Скользящие статистики (за последние N записей)
    rolling_mean_torque_10 NUMERIC(8, 4),
    rolling_std_torque_10 NUMERIC(8, 4),
    rolling_mean_temp_diff_10 NUMERIC(8, 4),
    rolling_std_temp_diff_10 NUMERIC(8, 4),
    
    -- Целевая переменная
    failure_target INTEGER,
    failure_probability NUMERIC(8, 4),
    
    -- Метаданные
    feature_version INTEGER DEFAULT 1,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Constraints
    CONSTRAINT fk_equipment_features FOREIGN KEY (equipment_id) 
        REFERENCES cleaned.equipment_cleaned(equipment_id)
);

COMMENT ON TABLE features.equipment_features IS 'Таблица с подготовленными признаками для ML моделей';

-- Индексы
CREATE INDEX idx_features_equipment_id ON features.equipment_features(equipment_id);
CREATE INDEX idx_features_date ON features.equipment_features(feature_date);
CREATE INDEX idx_features_target ON features.equipment_features(failure_target);
CREATE INDEX idx_features_active ON features.equipment_features(is_active);

-- Таблица метаданных признаков
CREATE TABLE features.feature_metadata (
    feature_id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100) UNIQUE NOT NULL,
    feature_description TEXT,
    feature_type VARCHAR(50), -- 'numeric', 'categorical', 'engineered'
    source_table VARCHAR(100),
    calculation_logic TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- =============================================
-- 6. Models Layer - таблицы для хранения моделей и результатов
-- =============================================

-- Таблица метаданных моделей
CREATE TABLE models.model_metadata (
    model_id VARCHAR(100) PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'lightgbm', 'xgboost', 'random_forest'
    model_path VARCHAR(500) NOT NULL,
    training_date DATE NOT NULL,
    features_used JSONB NOT NULL,
    hyperparameters JSONB NOT NULL,
    performance_metrics JSONB NOT NULL,
    training_data_range DATE[],
    is_production BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE models.model_metadata IS 'Метаданные обученных ML моделей';

-- Таблица результатов предсказаний
CREATE TABLE models.predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    equipment_id VARCHAR(50) NOT NULL,
    prediction_timestamp TIMESTAMP NOT NULL,
    model_id VARCHAR(100) NOT NULL,
    failure_probability NUMERIC(8, 4) NOT NULL,
    predicted_class INTEGER NOT NULL, -- 0 или 1
    prediction_threshold NUMERIC(8, 4) DEFAULT 0.35,
    actual_class INTEGER, -- заполняется позже
    is_correct BOOLEAN, -- заполняется позже
    shap_values JSONB,
    inference_time_ms NUMERIC(8, 2),
    batch_id UUID,
    
    -- Constraints
    CONSTRAINT fk_predictions_model FOREIGN KEY (model_id) 
        REFERENCES models.model_metadata(model_id),
    CONSTRAINT chk_probability_range CHECK (failure_probability BETWEEN 0 AND 1)
);

COMMENT ON TABLE models.predictions IS 'История предсказаний модели';

-- Индексы
CREATE INDEX idx_predictions_equipment ON models.predictions(equipment_id);
CREATE INDEX idx_predictions_timestamp ON models.predictions(prediction_timestamp);
CREATE INDEX idx_predictions_model ON models.predictions(model_id);
CREATE INDEX idx_predictions_batch ON models.predictions(batch_id);

-- Таблица для A/B тестирования моделей
CREATE TABLE models.ab_testing (
    test_id VARCHAR(100) PRIMARY KEY,
    test_name VARCHAR(200) NOT NULL,
    control_model_id VARCHAR(100) NOT NULL,
    treatment_model_id VARCHAR(100) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE,
    sample_size INTEGER,
    success_metric VARCHAR(100),
    results JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- 7. Monitoring Layer - таблицы для мониторинга
-- =============================================

-- Таблица метрик модели
CREATE TABLE monitoring.model_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    metric_date DATE NOT NULL,
    model_id VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC(10, 4) NOT NULL,
    metric_type VARCHAR(50), -- 'accuracy', 'recall', 'fpr', 'precision'
    calculation_window VARCHAR(50), -- 'daily', 'weekly', 'monthly'
    records_processed INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE monitoring.model_metrics IS 'Исторические метрики качества моделей';

-- Таблица дрейфа данных
CREATE TABLE monitoring.data_drift (
    drift_id BIGSERIAL PRIMARY KEY,
    check_date DATE NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    drift_score NUMERIC(10, 4) NOT NULL,
    drift_threshold NUMERIC(10, 4) NOT NULL,
    is_drifted BOOLEAN DEFAULT FALSE,
    reference_distribution JSONB,
    current_distribution JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Таблица алертов
CREATE TABLE monitoring.alerts (
    alert_id BIGSERIAL PRIMARY KEY,
    alert_type VARCHAR(50) NOT NULL, -- 'data_quality', 'model_performance', 'system'
    alert_severity VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high', 'critical'
    alert_message TEXT NOT NULL,
    equipment_id VARCHAR(50),
    model_id VARCHAR(100),
    alert_data JSONB,
    is_resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    resolved_by VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- 8. Триггеры для автоматического обновления временных меток
-- =============================================

-- Функция для обновления updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Применение триггера к основным таблицам
CREATE TRIGGER update_raw_equipment_updated_at 
    BEFORE UPDATE ON raw.equipment_data 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_cleaned_equipment_updated_at 
    BEFORE UPDATE ON cleaned.equipment_cleaned 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_features_updated_at 
    BEFORE UPDATE ON features.equipment_features 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_metadata_updated_at 
    BEFORE UPDATE ON models.model_metadata 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================
-- 9. Представления (Views) для удобства доступа
-- =============================================

-- Представление для быстрого доступа к активным признакам
CREATE VIEW features.active_features AS
SELECT 
    feature_name,
    feature_description,
    feature_type,
    source_table
FROM features.feature_metadata
WHERE is_active = TRUE;

-- Представление для текущих прогнозов
CREATE VIEW models.current_predictions AS
SELECT 
    p.equipment_id,
    p.prediction_timestamp,
    p.failure_probability,
    p.predicted_class,
    m.model_name,
    m.model_version
FROM models.predictions p
JOIN models.model_metadata m ON p.model_id = m.model_id
WHERE m.is_production = TRUE
ORDER BY p.prediction_timestamp DESC;

-- =============================================
-- 10. Права доступа и роли
-- =============================================

-- Создание ролей
CREATE ROLE data_scientist;
CREATE ROLE data_engineer;
CREATE ROLE ml_engineer;
CREATE ROLE analyst;
CREATE ROLE viewer;

-- Назначение прав для data_scientist
GRANT USAGE ON SCHEMA cleaned, features, models TO data_scientist;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA cleaned, features, models TO data_scientist;

-- Назначение прав для data_engineer
GRANT USAGE ON SCHEMA raw, cleaned, features TO data_engineer;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA raw, cleaned, features TO data_engineer;

-- Назначение прав для viewer (только чтение)
GRANT USAGE ON SCHEMA cleaned, features, models, monitoring TO viewer;
GRANT SELECT ON ALL TABLES IN SCHEMA cleaned, features, models, monitoring TO viewer;

-- =============================================
-- 11. Создание пользователей (пример)
-- =============================================
-- Раскомментировать при необходимости
/*
CREATE USER ml_user WITH PASSWORD 'secure_password';
GRANT data_scientist TO ml_user;

CREATE USER etl_user WITH PASSWORD 'secure_password';
GRANT data_engineer TO etl_user;

CREATE USER analyst_user WITH PASSWORD 'secure_password';
GRANT analyst TO analyst_user;
*/