-- functions.sql
-- Пользовательские функции для работы с данными предиктивного обслуживания

-- =============================================
-- 1. Функции для работы с временными рядами
-- =============================================

-- Функция для вычисления скользящего среднего
CREATE OR REPLACE FUNCTION features.calculate_moving_average(
    p_equipment_id VARCHAR,
    p_feature_name VARCHAR,
    p_window_size INTEGER DEFAULT 10
)
RETURNS TABLE (
    feature_date DATE,
    moving_average NUMERIC
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH ranked_features AS (
        SELECT 
            feature_date,
            CASE p_feature_name
                WHEN 'torque' THEN torque_normalized
                WHEN 'temp_diff' THEN temperature_difference
                WHEN 'tool_wear' THEN tool_wear_normalized
                ELSE 0
            END as feature_value,
            ROW_NUMBER() OVER (ORDER BY feature_date) as rn
        FROM features.equipment_features
        WHERE equipment_id = p_equipment_id
          AND is_active = TRUE
        ORDER BY feature_date
    )
    SELECT 
        rf.feature_date,
        AVG(rf.feature_value) OVER (
            ORDER BY rf.rn 
            ROWS BETWEEN p_window_size - 1 PRECEDING AND CURRENT ROW
        ) as moving_average
    FROM ranked_features rf
    ORDER BY rf.feature_date;
END;
$$;

-- Функция для вычисления тренда (производной)
CREATE OR REPLACE FUNCTION features.calculate_trend(
    p_equipment_id VARCHAR,
    p_feature_name VARCHAR,
    p_lookback_days INTEGER DEFAULT 7
)
RETURNS NUMERIC
LANGUAGE plpgsql
AS $$
DECLARE
    v_slope NUMERIC;
    v_intercept NUMERIC;
BEGIN
    EXECUTE '
        SELECT 
            regr_slope(feature_value, day_number),
            regr_intercept(feature_value, day_number)
        FROM (
            SELECT 
                CASE $2
                    WHEN ''torque'' THEN torque_normalized
                    WHEN ''temp_diff'' THEN temperature_difference
                    WHEN ''tool_wear'' THEN tool_wear_normalized
                    ELSE 0
                END as feature_value,
                ROW_NUMBER() OVER (ORDER BY feature_date) as day_number
            FROM features.equipment_features
            WHERE equipment_id = $1
              AND feature_date >= CURRENT_DATE - $3
              AND is_active = TRUE
            ORDER BY feature_date
        ) t' 
    INTO v_slope, v_intercept
    USING p_equipment_id, p_feature_name, p_lookback_days;
    
    RETURN v_slope;
END;
$$;

-- =============================================
-- 2. Функции для работы с вероятностями отказов
-- =============================================

-- Функция для получения текущего риска оборудования
CREATE OR REPLACE FUNCTION models.get_equipment_risk(
    p_equipment_id VARCHAR
)
RETURNS TABLE (
    risk_category VARCHAR,
    failure_probability NUMERIC,
    last_prediction TIMESTAMP,
    risk_score INTEGER
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH latest_prediction AS (
        SELECT 
            p.failure_probability,
            p.prediction_timestamp,
            p.predicted_class
        FROM models.predictions p
        WHERE p.equipment_id = p_equipment_id
          AND p.model_id IN (SELECT model_id FROM models.model_metadata WHERE is_production = TRUE)
        ORDER BY p.prediction_timestamp DESC
        LIMIT 1
    )
    SELECT 
        CASE 
            WHEN lp.failure_probability > 0.7 THEN 'CRITICAL'
            WHEN lp.failure_probability > 0.5 THEN 'HIGH'
            WHEN lp.failure_probability > 0.3 THEN 'MEDIUM'
            ELSE 'LOW'
        END as risk_category,
        lp.failure_probability,
        lp.prediction_timestamp,
        CASE 
            WHEN lp.predicted_class = 1 THEN 
                CASE 
                    WHEN lp.failure_probability > 0.7 THEN 100
                    WHEN lp.failure_probability > 0.5 THEN 75
                    WHEN lp.failure_probability > 0.3 THEN 50
                    ELSE 25
                END
            ELSE 0
        END as risk_score
    FROM latest_prediction lp;
END;
$$;

-- Функция для вычисления приоритета обслуживания
CREATE OR REPLACE FUNCTION maintenance.calculate_maintenance_priority(
    p_equipment_id VARCHAR
)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_risk_score INTEGER;
    v_tool_wear INTEGER;
    v_days_since_maintenance INTEGER;
    v_priority_score INTEGER;
BEGIN
    -- Получение риска
    SELECT risk_score INTO v_risk_score
    FROM models.get_equipment_risk(p_equipment_id);
    
    -- Получение износа инструмента
    SELECT tool_wear_min INTO v_tool_wear
    FROM cleaned.equipment_cleaned
    WHERE equipment_id = p_equipment_id
    ORDER BY cleaned_at DESC
    LIMIT 1;
    
    -- Расчет дней с последнего обслуживания (упрощенно)
    SELECT COALESCE(EXTRACT(DAY FROM CURRENT_DATE - MIN(feature_date)), 0)
    INTO v_days_since_maintenance
    FROM features.equipment_features
    WHERE equipment_id = p_equipment_id;
    
    -- Расчет приоритета (примерная формула)
    v_priority_score := 
        (v_risk_score * 5) + 
        (LEAST(v_tool_wear, 300) / 3) + 
        (v_days_since_maintenance * 2);
    
    RETURN v_priority_score;
END;
$$;

-- =============================================
-- 3. Функции для анализа дрейфа данных
-- =============================================

-- Функция для вычисления PSI (Population Stability Index)
CREATE OR REPLACE FUNCTION monitoring.calculate_psi(
    p_feature_name VARCHAR,
    p_reference_date DATE DEFAULT CURRENT_DATE - 30,
    p_current_date DATE DEFAULT CURRENT_DATE
)
RETURNS NUMERIC
LANGUAGE plpgsql
AS $$
DECLARE
    v_psi NUMERIC := 0;
    v_bin RECORD;
BEGIN
    FOR v_bin IN 
        SELECT 
            ref_bin,
            ref_percentage,
            curr_percentage
        FROM (
            -- Распределение в референсном периоде
            SELECT 
                width_bucket(
                    CASE p_feature_name
                        WHEN 'air_temp' THEN air_temp_normalized
                        WHEN 'torque' THEN torque_normalized
                        WHEN 'tool_wear' THEN tool_wear_normalized
                        ELSE 0
                    END, 
                    -5, 5, 10  -- Диапазон и количество бинов
                ) as ref_bin,
                COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as ref_percentage
            FROM features.equipment_features
            WHERE feature_date BETWEEN p_reference_date - 30 AND p_reference_date
            GROUP BY 1
        ) ref
        FULL OUTER JOIN (
            -- Распределение в текущем периоде
            SELECT 
                width_bucket(
                    CASE p_feature_name
                        WHEN 'air_temp' THEN air_temp_normalized
                        WHEN 'torque' THEN torque_normalized
                        WHEN 'tool_wear' THEN tool_wear_normalized
                        ELSE 0
                    END, 
                    -5, 5, 10
                ) as curr_bin,
                COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as curr_percentage
            FROM features.equipment_features
            WHERE feature_date BETWEEN p_current_date - 30 AND p_current_date
            GROUP BY 1
        ) curr ON ref.ref_bin = curr.curr_bin
    LOOP
        IF v_bin.ref_percentage > 0 AND v_bin.curr_percentage > 0 THEN
            v_psi := v_psi + 
                (v_bin.curr_percentage - v_bin.ref_percentage) * 
                LN(v_bin.curr_percentage / v_bin.ref_percentage);
        END IF;
    END LOOP;
    
    RETURN v_psi;
END;
$$;

-- =============================================
-- 4. Утилитарные функции
-- =============================================

-- Функция для форматирования вероятности отказа
CREATE OR REPLACE FUNCTION utils.format_probability(
    p_probability NUMERIC
)
RETURNS VARCHAR
LANGUAGE plpgsql
IMMUTABLE
AS $$
BEGIN
    RETURN CASE 
        WHEN p_probability IS NULL THEN 'N/A'
        WHEN p_probability < 0.001 THEN '<0.1%'
        WHEN p_probability > 0.999 THEN '>99.9%'
        ELSE ROUND(p_probability * 100, 1) || '%'
    END;
END;
$$;

-- Функция для преобразования типа оборудования в читаемый формат
CREATE OR REPLACE FUNCTION utils.format_equipment_type(
    p_type_code VARCHAR
)
RETURNS VARCHAR
LANGUAGE plpgsql
IMMUTABLE
AS $$
BEGIN
    RETURN CASE p_type_code
        WHEN 'L' THEN 'Low Quality'
        WHEN 'M' THEN 'Medium Quality'
        WHEN 'H' THEN 'High Quality'
        ELSE 'Unknown'
    END;
END;
$$;

-- Функция для вычисления возраста оборудования (упрощенно)
CREATE OR REPLACE FUNCTION utils.calculate_equipment_age(
    p_equipment_id VARCHAR
)
RETURNS INTERVAL
LANGUAGE plpgsql
AS $$
DECLARE
    v_first_record TIMESTAMP;
BEGIN
    SELECT MIN(timestamp)
    INTO v_first_record
    FROM cleaned.equipment_cleaned
    WHERE equipment_id = p_equipment_id;
    
    RETURN CURRENT_TIMESTAMP - v_first_record;
END;
$$;

-- =============================================
-- 5. Функции для отчетов и дашбордов
-- =============================================

-- Функция для генерации ежедневного отчета
CREATE OR REPLACE FUNCTION reports.generate_daily_report(
    p_report_date DATE DEFAULT CURRENT_DATE
)
RETURNS TABLE (
    metric_name VARCHAR,
    metric_value VARCHAR,
    metric_trend VARCHAR
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH daily_metrics AS (
        -- Количество обработанных записей
        SELECT 
            'Processed Records' as metric_name,
            COUNT(*)::VARCHAR as metric_value,
            CASE 
                WHEN COUNT(*) > LAG(COUNT(*)) OVER (ORDER BY DATE(created_at)) THEN '↑'
                WHEN COUNT(*) < LAG(COUNT(*)) OVER (ORDER BY DATE(created_at)) THEN '↓'
                ELSE '→'
            END as metric_trend
        FROM cleaned.equipment_cleaned
        WHERE DATE(created_at) = p_report_date
        
        UNION ALL
        
        -- Качество данных
        SELECT 
            'Data Quality (%)',
            ROUND(100.0 * SUM(CASE WHEN is_valid THEN 1 ELSE 0 END) / COUNT(*), 1)::VARCHAR,
            '→'
        FROM cleaned.equipment_cleaned
        WHERE DATE(created_at) = p_report_date
        
        UNION ALL
        
        -- Количество отказов
        SELECT 
            'Failure Count',
            SUM(target)::VARCHAR,
            CASE 
                WHEN SUM(target) > LAG(SUM(target)) OVER (ORDER BY DATE(created_at)) THEN '↑'
                WHEN SUM(target) < LAG(SUM(target)) OVER (ORDER BY DATE(created_at)) THEN '↓'
                ELSE '→'
            END
        FROM cleaned.equipment_cleaned
        WHERE DATE(created_at) = p_report_date
        
        UNION ALL
        
        -- Производительность модели
        SELECT 
            'Model Recall (%)',
            ROUND(MAX(metric_value) * 100, 1)::VARCHAR,
            '→'
        FROM monitoring.model_metrics
        WHERE metric_date = p_report_date
          AND metric_name = 'recall'
    )
    SELECT * FROM daily_metrics;
END;
$$;

-- Функция для получения топ-N оборудования по риску
CREATE OR REPLACE FUNCTION reports.get_top_risky_equipment(
    p_limit INTEGER DEFAULT 10,
    p_min_probability NUMERIC DEFAULT 0.3
)
RETURNS TABLE (
    equipment_id VARCHAR,
    equipment_type VARCHAR,
    failure_probability NUMERIC,
    risk_category VARCHAR,
    tool_wear INTEGER,
    last_maintenance_days INTEGER
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH latest_predictions AS (
        SELECT DISTINCT ON (p.equipment_id)
            p.equipment_id,
            p.failure_probability,
            p.prediction_timestamp,
            CASE 
                WHEN p.failure_probability > 0.7 THEN 'CRITICAL'
                WHEN p.failure_probability > 0.5 THEN 'HIGH'
                WHEN p.failure_probability > 0.3 THEN 'MEDIUM'
                ELSE 'LOW'
            END as risk_category
        FROM models.predictions p
        WHERE p.model_id IN (SELECT model_id FROM models.model_metadata WHERE is_production = TRUE)
          AND p.predicted_class = 1
          AND p.failure_probability >= p_min_probability
        ORDER BY p.equipment_id, p.prediction_timestamp DESC
    )
    SELECT 
        lp.equipment_id,
        c.equipment_type,
        lp.failure_probability,
        lp.risk_category,
        c.tool_wear_min as tool_wear,
        EXTRACT(DAY FROM CURRENT_DATE - MIN(f.feature_date))::INTEGER as last_maintenance_days
    FROM latest_predictions lp
    JOIN cleaned.equipment_cleaned c ON lp.equipment_id = c.equipment_id
    LEFT JOIN features.equipment_features f ON lp.equipment_id = f.equipment_id
    WHERE c.is_valid = TRUE
    GROUP BY lp.equipment_id, lp.failure_probability, lp.risk_category, c.equipment_type, c.tool_wear_min
    ORDER BY lp.failure_probability DESC
    LIMIT p_limit;
END;
$$;