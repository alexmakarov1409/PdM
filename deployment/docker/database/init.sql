-- Инициализация базы данных для Predictive Maintenance

-- Таблица для оборудования
CREATE TABLE IF NOT EXISTS equipment (
    equipment_id VARCHAR(50) PRIMARY KEY,
    equipment_type VARCHAR(50) NOT NULL,
    manufacturer VARCHAR(100),
    model VARCHAR(100),
    serial_number VARCHAR(100),
    installation_date DATE,
    location VARCHAR(200),
    department VARCHAR(100),
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Таблица для данных датчиков
CREATE TABLE IF NOT EXISTS sensor_data (
    id SERIAL PRIMARY KEY,
    equipment_id VARCHAR(50) REFERENCES equipment(equipment_id),
    timestamp TIMESTAMP NOT NULL,
    temperature DECIMAL(5,2),
    vibration DECIMAL(5,2),
    pressure DECIMAL(6,2),
    rotation_speed DECIMAL(8,2),
    power_consumption DECIMAL(8,2),
    efficiency DECIMAL(6,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(equipment_id, timestamp)
);

-- Таблица для предсказаний
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id VARCHAR(50) PRIMARY KEY,
    request_id VARCHAR(50),
    equipment_id VARCHAR(50) REFERENCES equipment(equipment_id),
    timestamp TIMESTAMP NOT NULL,
    failure_probability DECIMAL(5,4),
    prediction BOOLEAN,
    confidence DECIMAL(5,4),
    alert_level VARCHAR(20),
    inference_time_ms DECIMAL(8,2),
    model_version VARCHAR(20),
    features_used JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Таблица для обслуживания
CREATE TABLE IF NOT EXISTS maintenance_logs (
    log_id SERIAL PRIMARY KEY,
    equipment_id VARCHAR(50) REFERENCES equipment(equipment_id),
    maintenance_type VARCHAR(50),
    description TEXT,
    performed_by VARCHAR(100),
    performed_at TIMESTAMP,
    duration_minutes INTEGER,
    cost DECIMAL(10,2),
    parts_used JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Таблица для алертов
CREATE TABLE IF NOT EXISTS alerts (
    alert_id SERIAL PRIMARY KEY,
    equipment_id VARCHAR(50) REFERENCES equipment(equipment_id),
    alert_level VARCHAR(20),
    alert_type VARCHAR(50),
    message TEXT,
    prediction_id VARCHAR(50) REFERENCES predictions(prediction_id),
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMP,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Индексы для производительности
CREATE INDEX idx_sensor_data_equipment_timestamp 
ON sensor_data(equipment_id, timestamp DESC);

CREATE INDEX idx_predictions_equipment_timestamp 
ON predictions(equipment_id, timestamp DESC);

CREATE INDEX idx_predictions_created_at 
ON predictions(created_at DESC);

CREATE INDEX idx_alerts_equipment_status 
ON alerts(equipment_id, resolved, acknowledged);

CREATE INDEX idx_alerts_created_at 
ON alerts(created_at DESC);

-- Представление для аналитики
CREATE OR REPLACE VIEW equipment_health_view AS
SELECT 
    e.equipment_id,
    e.equipment_type,
    e.location,
    e.status,
    sd.timestamp as last_reading,
    sd.temperature,
    sd.vibration,
    sd.pressure,
    p.timestamp as last_prediction,
    p.failure_probability,
    p.alert_level,
    p.prediction,
    a.alert_level as active_alert,
    a.acknowledged,
    a.resolved
FROM equipment e
LEFT JOIN sensor_data sd ON e.equipment_id = sd.equipment_id 
    AND sd.timestamp = (SELECT MAX(timestamp) FROM sensor_data WHERE equipment_id = e.equipment_id)
LEFT JOIN predictions p ON e.equipment_id = p.equipment_id 
    AND p.timestamp = (SELECT MAX(timestamp) FROM predictions WHERE equipment_id = e.equipment_id)
LEFT JOIN alerts a ON e.equipment_id = a.equipment_id 
    AND a.resolved = FALSE;

-- Функция для обновления updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Триггеры для обновления updated_at
CREATE TRIGGER update_equipment_updated_at 
    BEFORE UPDATE ON equipment
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Начальные данные (опционально)
INSERT INTO equipment (equipment_id, equipment_type, manufacturer, model, location) VALUES
('EQ-001', 'pump', 'Grundfos', 'CR 45', 'Factory A - Line 1'),
('EQ-002', 'compressor', 'Atlas Copco', 'GA 75', 'Factory A - Line 2'),
('EQ-003', 'turbine', 'Siemens', 'SGT-400', 'Factory B - Power Plant'),
('EQ-004', 'generator', 'Caterpillar', 'C175', 'Factory B - Backup'),
('EQ-005', 'conveyor', 'Siemens', 'Simodrive', 'Factory A - Assembly');

-- Создание пользователя для приложения
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'app_user') THEN
        CREATE USER app_user WITH PASSWORD 'app_password';
    END IF;
END
$$;

-- Предоставление прав
GRANT CONNECT ON DATABASE predictive_maintenance TO app_user;
GRANT USAGE ON SCHEMA public TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO app_user;