-- queries.sql
-- Полезные SQL-запросы для анализа данных и мониторинга системы

-- =============================================
-- 1. Запросы для мониторинга качества данных
-- =============================================

-- Общая статистика по данным
SELECT 
    'raw' as layer,
    COUNT(*) as total_records,
    COUNT(DISTINCT udi) as unique_records,
    SUM(CASE WHEN udi IS NULL THEN 1 ELSE 0 END) as null_udi,
    MIN(timestamp) as earliest_record,
    MAX(timestamp) as latest_record
FROM raw.equipment_data

UNION ALL

SELECT 
    'cleaned' as layer,
    COUNT(*) as total_records,
    COUNT(DISTINCT equipment_id) as unique_records,
    SUM(CASE WHEN is_valid = FALSE THEN 1 ELSE 0 END) as invalid_records,
    MIN(created_at) as earliest_record,
    MAX(created_at) as latest_record
FROM cleaned.equipment_cleaned

UNION ALL

SELECT 
    'features' as layer,
    COUNT(*) as total_records,
    COUNT(DISTINCT equipment_id) as unique_equipment,
    SUM(CASE WHEN is_active = FALSE THEN 1 ELSE 0 END) as inactive_records,
    MIN(generated_at) as earliest_record,
    MAX(generated_at) as latest_record
FROM features.equipment_features;

-- Распределение отказов по типам оборудования
SELECT 
    equipment_type,
    COUNT(*) as total_records,
    SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) as failure_count,
    ROUND(100.0 * SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as failure_rate_percent,
    AVG(tool_wear_min) as avg_tool_wear,
    AVG(torque_nm) as avg_torque
FROM cleaned.equipment_cleaned
WHERE is_valid = TRUE
GROUP BY equipment_type
ORDER BY failure_rate_percent DESC;

-- Анализ пропущенных значений
SELECT 
    'air_temperature_k' as column_name,
    SUM(CASE WHEN air_temperature_k IS NULL THEN 1 ELSE 0 END) as null_count,
    ROUND(100.0 * SUM(CASE WHEN air_temperature_k IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as null_percent
FROM cleaned.equipment_cleaned

UNION ALL

SELECT 
    'torque_nm',
    SUM(CASE WHEN torque_nm IS NULL THEN 1 ELSE 0 END),
    ROUND(100.0 * SUM(CASE WHEN torque_nm IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2)
FROM cleaned.equipment_cleaned

UNION ALL

SELECT 
    'tool_wear_min',
    SUM(CASE WHEN tool_wear_min IS NULL THEN 1 ELSE 0 END),
    ROUND(100.0 * SUM(CASE WHEN tool_wear_min IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2)
FROM cleaned.equipment_cleaned;

-- =============================================
-- 2. Запросы для анализа работы моделей
-- =============================================

-- Текущая production модель и ее метрики
SELECT 
    model_name,
    model_version,
    model_type,
    training_date,
    performance_metrics->>'recall' as recall,
    performance_metrics->>'fpr' as fpr,
    performance_metrics->>'precision' as precision,
    performance_metrics->>'auc_roc' as auc_roc,
    is_production
FROM models.model_metadata
WHERE is_active = TRUE
ORDER BY training_date DESC
LIMIT 5;

-- Статистика предсказаний за последние 7 дней
SELECT 
    DATE(prediction_timestamp) as prediction_date,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN predicted_class = 1 THEN 1 ELSE 0 END) as positive_predictions,
    SUM(CASE WHEN actual_class = 1 THEN 1 ELSE 0 END) as actual_failures,
    ROUND(AVG(failure_probability)::NUMERIC, 3) as avg_failure_probability,
    ROUND(AVG(inference_time_ms)::NUMERIC, 2) as avg_inference_time_ms
FROM models.predictions
WHERE prediction_timestamp >= CURRENT_DATE - INTERVAL '7 days'
  AND model_id IN (SELECT model_id FROM models.model_metadata WHERE is_production = TRUE)
GROUP BY DATE(prediction_timestamp)
ORDER BY prediction_date DESC;

-- Матрица ошибок для production модели
WITH predictions_actual AS (
    SELECT 
        predicted_class,
        actual_class,
        COUNT(*) as count
    FROM models.predictions p
    JOIN models.model_metadata m ON p.model_id = m.model_id
    WHERE m.is_production = TRUE
      AND actual_class IS NOT NULL
      AND prediction_timestamp >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY predicted_class, actual_class
)
SELECT 
    actual_class,
    SUM(CASE WHEN predicted_class = 0 THEN count ELSE 0 END) as predicted_0,
    SUM(CASE WHEN predicted_class = 1 THEN count ELSE 0 END) as predicted_1,
    SUM(count) as total
FROM predictions_actual
GROUP BY actual_class
ORDER BY actual_class;

-- =============================================
-- 3. Бизнес-метрики и KPI
-- =============================================

-- Расчет экономического эффекта
WITH daily_stats AS (
    SELECT 
        DATE(prediction_timestamp) as date,
        -- Ложные срабатывания
        SUM(CASE WHEN predicted_class = 1 AND actual_class = 0 THEN 1 ELSE 0 END) as false_positives,
        -- Пропущенные отказы
        SUM(CASE WHEN predicted_class = 0 AND actual_class = 1 THEN 1 ELSE 0 END) as false_negatives,
        -- Правильные предсказания отказов
        SUM(CASE WHEN predicted_class = 1 AND actual_class = 1 THEN 1 ELSE 0 END) as true_positives
    FROM models.predictions
    WHERE actual_class IS NOT NULL
      AND prediction_timestamp >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY DATE(prediction_timestamp)
)
SELECT 
    'Last 30 days' as period,
    SUM(false_positives) as total_false_positives,
    SUM(false_negatives) as total_false_negatives,
    SUM(true_positives) as total_true_positives,
    -- Экономический расчет (примерные стоимости)
    SUM(false_positives) * 5000 as fp_cost, -- 5000 руб за ложный вызов
    SUM(false_negatives) * 15000 as fn_cost, -- 15000 руб за пропущенный отказ
    SUM(true_positives) * 10000 as tp_savings -- 10000 руб экономии на каждом предотвращенном отказе
FROM daily_stats;

-- Мониторинг метрик качества модели во времени
SELECT 
    metric_date,
    metric_name,
    metric_value,
    LAG(metric_value) OVER (PARTITION BY metric_name ORDER BY metric_date) as previous_value,
    metric_value - LAG(metric_value) OVER (PARTITION BY metric_name ORDER BY metric_date) as change
FROM monitoring.model_metrics
WHERE model_id IN (SELECT model_id FROM models.model_metadata WHERE is_production = TRUE)
  AND metric_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY metric_date DESC, metric_name;

-- =============================================
-- 4. Анализ оборудования с высоким риском
-- =============================================

-- Топ 10 оборудования с наибольшей вероятностью отказа
SELECT 
    p.equipment_id,
    c.equipment_type,
    c.tool_wear_min,
    c.torque_nm,
    p.failure_probability,
    p.prediction_timestamp,
    RANK() OVER (ORDER BY p.failure_probability DESC) as risk_rank
FROM models.predictions p
JOIN cleaned.equipment_cleaned c ON p.equipment_id = c.equipment_id
WHERE p.prediction_timestamp >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
  AND p.model_id IN (SELECT model_id FROM models.model_metadata WHERE is_production = TRUE)
  AND p.predicted_class = 1
ORDER BY p.failure_probability DESC
LIMIT 10;

-- Анализ параметров оборудования с отказами
SELECT 
    equipment_type,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tool_wear_min) as median_tool_wear,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY torque_nm) as median_torque,
    AVG(process_temperature_k - air_temperature_k) as avg_temp_diff,
    COUNT(*) as failure_count
FROM cleaned.equipment_cleaned
WHERE target = 1
  AND is_valid = TRUE
GROUP BY equipment_type
ORDER BY failure_count DESC;

-- =============================================
-- 5. Запросы для мониторинга системы
-- =============================================

-- Статус последних загрузок данных
SELECT 
    file_name,
    status,
    records_count,
    error_message,
    started_at,
    completed_at,
    AGE(completed_at, started_at) as duration
FROM raw.data_imports
ORDER BY started_at DESC
LIMIT 10;

-- Мониторинг дрейфа данных
SELECT 
    check_date,
    feature_name,
    drift_score,
    drift_threshold,
    is_drifted,
    CASE 
        WHEN drift_score > drift_threshold * 1.5 THEN 'HIGH'
        WHEN drift_score > drift_threshold THEN 'MEDIUM'
        ELSE 'LOW'
    END as drift_severity
FROM monitoring.data_drift
WHERE check_date >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY check_date DESC, drift_score DESC;

-- Необработанные алерты
SELECT 
    alert_id,
    alert_type,
    alert_severity,
    alert_message,
    equipment_id,
    created_at,
    AGE(CURRENT_TIMESTAMP, created_at) as age
FROM monitoring.alerts
WHERE is_resolved = FALSE
ORDER BY 
    CASE alert_severity
        WHEN 'critical' THEN 1
        WHEN 'high' THEN 2
        WHEN 'medium' THEN 3
        WHEN 'low' THEN 4
    END,
    created_at DESC;

-- =============================================
-- 6. Экспортные запросы для отчетов
-- =============================================

-- Ежедневный отчет по качеству данных
SELECT 
    ds.statistic_date,
    ds.total_records,
    ds.valid_records,
    ds.invalid_records,
    ds.failure_count,
    ds.failure_rate_percent,
    COALESCE(mm.recall, 0) as model_recall,
    COALESCE(mm.precision, 0) as model_precision
FROM monitoring.daily_statistics ds
LEFT JOIN (
    SELECT 
        metric_date,
        MAX(CASE WHEN metric_name = 'recall' THEN metric_value END) as recall,
        MAX(CASE WHEN metric_name = 'precision' THEN metric_value END) as precision
    FROM monitoring.model_metrics
    WHERE model_id IN (SELECT model_id FROM models.model_metadata WHERE is_production = TRUE)
    GROUP BY metric_date
) mm ON ds.statistic_date = mm.metric_date
WHERE ds.statistic_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY ds.statistic_date DESC;

-- Отчет по эффективности модели по типам оборудования
SELECT 
    c.equipment_type,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN p.actual_class = 1 THEN 1 ELSE 0 END) as actual_failures,
    SUM(CASE WHEN p.predicted_class = 1 THEN 1 ELSE 0 END) as predicted_failures,
    SUM(CASE WHEN p.predicted_class = 1 AND p.actual_class = 1 THEN 1 ELSE 0 END) as true_positives,
    SUM(CASE WHEN p.predicted_class = 0 AND p.actual_class = 1 THEN 1 ELSE 0 END) as false_negatives,
    SUM(CASE WHEN p.predicted_class = 1 AND p.actual_class = 0 THEN 1 ELSE 0 END) as false_positives,
    ROUND(100.0 * SUM(CASE WHEN p.predicted_class = 1 AND p.actual_class = 1 THEN 1 ELSE 0 END) / 
          NULLIF(SUM(CASE WHEN p.actual_class = 1 THEN 1 ELSE 0 END), 0), 2) as recall_percent,
    ROUND(100.0 * SUM(CASE WHEN p.predicted_class = 1 AND p.actual_class = 0 THEN 1 ELSE 0 END) / 
          NULLIF(SUM(CASE WHEN p.predicted_class = 1 THEN 1 ELSE 0 END), 0), 2) as fpr_percent
FROM models.predictions p
JOIN cleaned.equipment_cleaned c ON p.equipment_id = c.equipment_id
WHERE p.actual_class IS NOT NULL
  AND p.prediction_timestamp >= CURRENT_DATE - INTERVAL '30 days'
  AND p.model_id IN (SELECT model_id FROM models.model_metadata WHERE is_production = TRUE)
GROUP BY c.equipment_type
ORDER BY recall_percent DESC;

-- =============================================
-- 7. Административные запросы
-- =============================================

-- Использование дискового пространства
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) as total_size,
    pg_size_pretty(pg_relation_size(schemaname || '.' || tablename)) as table_size,
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename) - 
                   pg_relation_size(schemaname || '.' || tablename)) as index_size
FROM pg_tables
WHERE schemaname IN ('raw', 'cleaned', 'features', 'models', 'monitoring')
ORDER BY pg_total_relation_size(schemaname || '.' || tablename) DESC;

-- Активные подключения к базе данных
SELECT 
    datname as database,
    usename as username,
    client_addr as client_address,
    application_name,
    state,
    COUNT(*) as connection_count
FROM pg_stat_activity
WHERE datname = 'predictive_maintenance'
GROUP BY datname, usename, client_addr, application_name, state
ORDER BY connection_count DESC;

-- Статистика по индексам
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(schemaname || '.' || indexname)) as index_size,
    idx_scan as index_scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched
FROM pg_stat_user_indexes
WHERE schemaname IN ('raw', 'cleaned', 'features', 'models', 'monitoring')
ORDER BY idx_scan DESC
LIMIT 20;