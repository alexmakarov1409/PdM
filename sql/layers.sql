-- layers.sql
-- Скрипты для управления логическими слоями данных

-- =============================================
-- 1. Процедуры для работы с Raw Layer
-- =============================================

-- Процедура для загрузки сырых данных
CREATE OR REPLACE PROCEDURE raw.load_raw_data(
    p_file_path VARCHAR,
    p_batch_id UUID DEFAULT gen_random_uuid()
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_records_processed INTEGER;
    v_error_message TEXT;
BEGIN
    -- Начало транзакции
    BEGIN
        -- В реальном проекте здесь была бы команда COPY или импорт из файла
        -- Для примера оставим заглушку
        INSERT INTO raw.data_imports (
            file_name,
            file_size,
            status,
            started_at
        ) VALUES (
            p_file_path,
            0, -- Заглушка для размера файла
            'in_progress',
            CURRENT_TIMESTAMP
        );
        
        -- Здесь будет логика загрузки данных
        -- ...
        
        -- Обновление статуса загрузки
        UPDATE raw.data_imports 
        SET 
            status = 'completed',
            completed_at = CURRENT_TIMESTAMP,
            records_count = v_records_processed
        WHERE file_name = p_file_path;
        
        COMMIT;
        
    EXCEPTION WHEN OTHERS THEN
        v_error_message := SQLERRM;
        
        UPDATE raw.data_imports 
        SET 
            status = 'failed',
            error_message = v_error_message
        WHERE file_name = p_file_path;
        
        RAISE NOTICE 'Ошибка при загрузке данных: %', v_error_message;
        RAISE;
    END;
END;
$$;

-- =============================================
-- 2. Процедуры для работы с Cleaned Layer
-- =============================================

-- Процедура для очистки и валидации данных
CREATE OR REPLACE PROCEDURE cleaned.clean_and_validate_data(
    p_batch_id UUID DEFAULT NULL
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_cleaning_id BIGINT;
    v_start_time TIMESTAMP;
    v_records_processed INTEGER;
    v_records_validated INTEGER;
    v_records_invalidated INTEGER;
BEGIN
    v_start_time := CURRENT_TIMESTAMP;
    
    -- Создание записи в истории очистки
    INSERT INTO cleaned.cleaning_history (
        cleaning_date,
        cleaning_rules
    ) VALUES (
        CURRENT_DATE,
        '{
            "remove_duplicates": true,
            "validate_ranges": true,
            "handle_missing": "impute"
        }'::JSONB
    ) RETURNING id INTO v_cleaning_id;
    
    -- Очистка дубликатов (если batch_id не указан, очищаем все)
    IF p_batch_id IS NULL THEN
        -- Удаление дубликатов по UDI
        WITH duplicates AS (
            SELECT udi,
                   MIN(id) as min_id,
                   COUNT(*) as dup_count
            FROM raw.equipment_data
            GROUP BY udi
            HAVING COUNT(*) > 1
        )
        DELETE FROM raw.equipment_data r
        USING duplicates d
        WHERE r.udi = d.udi 
          AND r.id != d.min_id;
    END IF;
    
    -- Вставка очищенных данных с валидацией
    INSERT INTO cleaned.equipment_cleaned (
        equipment_id,
        udi,
        product_id,
        equipment_type,
        air_temperature_k,
        process_temperature_k,
        rotational_speed_rpm,
        torque_nm,
        tool_wear_min,
        target,
        failure_type,
        timestamp,
        is_valid,
        validation_errors
    )
    SELECT 
        -- Генерация equipment_id
        md5(r.udi || r.product_id)::VARCHAR(50) as equipment_id,
        r.udi,
        r.product_id,
        r.equipment_type,
        
        -- Валидация и преобразование температур
        CASE 
            WHEN r.air_temperature_k BETWEEN 290 AND 310 THEN r.air_temperature_k
            ELSE NULL
        END as air_temperature_k,
        
        CASE 
            WHEN r.process_temperature_k BETWEEN 305 AND 320 THEN r.process_temperature_k
            ELSE NULL
        END as process_temperature_k,
        
        -- Валидация скорости вращения
        CASE 
            WHEN r.rotational_speed_rpm BETWEEN 1000 AND 3000 THEN r.rotational_speed_rpm
            ELSE NULL
        END as rotational_speed_rpm,
        
        -- Валидация крутящего момента
        CASE 
            WHEN r.torque_nm BETWEEN 0 AND 100 THEN r.torque_nm
            ELSE NULL
        END as torque_nm,
        
        -- Валидация износа инструмента
        CASE 
            WHEN r.tool_wear_min BETWEEN 0 AND 300 THEN r.tool_wear_min
            ELSE NULL
        END as tool_wear_min,
        
        r.target,
        r.failure_type,
        r.timestamp,
        
        -- Флаг валидности (все значения должны быть не NULL)
        NOT (
            r.air_temperature_k IS NULL OR
            r.process_temperature_k IS NULL OR
            r.rotational_speed_rpm IS NULL OR
            r.torque_nm IS NULL OR
            r.tool_wear_min IS NULL
        ) as is_valid,
        
        -- Сбор ошибок валидации
        jsonb_build_object(
            'air_temp_invalid', r.air_temperature_k NOT BETWEEN 290 AND 310,
            'process_temp_invalid', r.process_temperature_k NOT BETWEEN 305 AND 320,
            'speed_invalid', r.rotational_speed_rpm NOT BETWEEN 1000 AND 3000,
            'torque_invalid', r.torque_nm NOT BETWEEN 0 AND 100,
            'wear_invalid', r.tool_wear_min NOT BETWEEN 0 AND 300
        ) as validation_errors
        
    FROM raw.equipment_data r
    WHERE (p_batch_id IS NULL OR r.batch_id = p_batch_id);
    
    -- Подсчет статистики
    GET DIAGNOSTICS v_records_processed = ROW_COUNT;
    
    SELECT COUNT(*) INTO v_records_validated
    FROM cleaned.equipment_cleaned
    WHERE is_valid = TRUE;
    
    SELECT COUNT(*) INTO v_records_invalidated
    FROM cleaned.equipment_cleaned
    WHERE is_valid = FALSE;
    
    -- Обновление истории очистки
    UPDATE cleaned.cleaning_history
    SET 
        records_processed = v_records_processed,
        records_validated = v_records_validated,
        records_invalidated = v_records_invalidated,
        cleaning_duration = CURRENT_TIMESTAMP - v_start_time
    WHERE id = v_cleaning_id;
    
    RAISE NOTICE 'Очистка завершена. Обработано: %, Валидных: %, Невалидных: %', 
        v_records_processed, v_records_validated, v_records_invalidated;
END;
$$;

-- =============================================
-- 3. Функции для работы с Features Layer
-- =============================================

-- Функция для генерации признаков
CREATE OR REPLACE PROCEDURE features.generate_features(
    p_date DATE DEFAULT CURRENT_DATE
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_feature_version INTEGER;
BEGIN
    -- Получение текущей версии признаков
    SELECT COALESCE(MAX(feature_version), 0) + 1 
    INTO v_feature_version
    FROM features.equipment_features;
    
    -- Генерация признаков
    INSERT INTO features.equipment_features (
        equipment_id,
        feature_date,
        
        -- Базовые признаки (нормализованные)
        equipment_type_encoded,
        air_temp_normalized,
        process_temp_normalized,
        rotational_speed_normalized,
        torque_normalized,
        tool_wear_normalized,
        
        -- Инженерные признаки
        temperature_difference,
        torque_speed_ratio,
        power_estimate,
        wear_category,
        
        -- Скользящие статистики
        rolling_mean_torque_10,
        rolling_std_torque_10,
        rolling_mean_temp_diff_10,
        rolling_std_temp_diff_10,
        
        -- Целевая переменная
        failure_target,
        failure_probability,
        
        feature_version,
        generated_at
    )
    SELECT 
        c.equipment_id,
        p_date as feature_date,
        
        -- Кодирование типа оборудования
        CASE c.equipment_type
            WHEN 'L' THEN 0
            WHEN 'M' THEN 1
            WHEN 'H' THEN 2
            ELSE 0
        END as equipment_type_encoded,
        
        -- Нормализация числовых признаков (z-score)
        (c.air_temperature_k - avg_temp.avg_air_temp) / std_temp.std_air_temp as air_temp_normalized,
        (c.process_temperature_k - avg_temp.avg_process_temp) / std_temp.std_process_temp as process_temp_normalized,
        (c.rotational_speed_rpm - avg_speed.avg_speed) / std_speed.std_speed as rotational_speed_normalized,
        (c.torque_nm - avg_torque.avg_torque) / std_torque.std_torque as torque_normalized,
        (c.tool_wear_min - avg_wear.avg_wear) / std_wear.std_wear as tool_wear_normalized,
        
        -- Инженерные признаки
        c.process_temperature_k - c.air_temperature_k as temperature_difference,
        CASE 
            WHEN c.rotational_speed_rpm > 0 THEN c.torque_nm / c.rotational_speed_rpm
            ELSE 0
        END as torque_speed_ratio,
        c.torque_nm * c.rotational_speed_rpm / 9549.3 as power_estimate, -- Примерная мощность в кВт
        
        -- Категоризация износа
        CASE 
            WHEN c.tool_wear_min < 50 THEN 'low'
            WHEN c.tool_wear_min BETWEEN 50 AND 150 THEN 'medium'
            WHEN c.tool_wear_min > 150 THEN 'high'
            ELSE 'unknown'
        END as wear_category,
        
        -- Скользящие средние (упрощенный пример)
        0.0 as rolling_mean_torque_10, -- В реальности вычисляется по истории
        0.0 as rolling_std_torque_10,
        0.0 as rolling_mean_temp_diff_10,
        0.0 as rolling_std_temp_diff_10,
        
        -- Целевая переменная
        c.target as failure_target,
        CASE 
            WHEN c.target = 1 THEN 0.85 -- Пример начальной вероятности
            ELSE 0.15
        END as failure_probability,
        
        v_feature_version,
        CURRENT_TIMESTAMP
        
    FROM cleaned.equipment_cleaned c
    CROSS JOIN (
        SELECT 
            AVG(air_temperature_k) as avg_air_temp,
            STDDEV(air_temperature_k) as std_air_temp,
            AVG(process_temperature_k) as avg_process_temp,
            STDDEV(process_temperature_k) as std_process_temp
        FROM cleaned.equipment_cleaned
    ) avg_temp
    CROSS JOIN (
        SELECT 
            AVG(rotational_speed_rpm) as avg_speed,
            STDDEV(rotational_speed_rpm) as std_speed
        FROM cleaned.equipment_cleaned
    ) avg_speed
    CROSS JOIN (
        SELECT 
            AVG(torque_nm) as avg_torque,
            STDDEV(torque_nm) as std_torque
        FROM cleaned.equipment_cleaned
    ) avg_torque
    CROSS JOIN (
        SELECT 
            AVG(tool_wear_min) as avg_wear,
            STDDEV(tool_wear_min) as std_wear
        FROM cleaned.equipment_cleaned
    ) avg_wear
    WHERE c.is_valid = TRUE;
    
    -- Обновление метаданных признаков
    INSERT INTO features.feature_metadata (
        feature_name,
        feature_description,
        feature_type,
        source_table,
        calculation_logic
    ) VALUES 
    ('temperature_difference', 'Разность технологической и воздушной температуры', 'engineered', 'cleaned.equipment_cleaned', 'process_temp - air_temp'),
    ('torque_speed_ratio', 'Отношение крутящего момента к скорости вращения', 'engineered', 'cleaned.equipment_cleaned', 'torque / speed'),
    ('power_estimate', 'Оценочная мощность оборудования', 'engineered', 'cleaned.equipment_cleaned', 'torque * speed / 9549.3')
    ON CONFLICT (feature_name) DO UPDATE 
    SET updated_at = CURRENT_TIMESTAMP;
    
    RAISE NOTICE 'Сгенерированы признаки версии % для даты %', v_feature_version, p_date;
END;
$$;

-- =============================================
-- 4. Процедуры для работы с Models Layer
-- =============================================

-- Процедура для сохранения метаданных модели
CREATE OR REPLACE PROCEDURE models.save_model_metadata(
    p_model_id VARCHAR,
    p_model_name VARCHAR,
    p_model_version VARCHAR,
    p_model_type VARCHAR,
    p_model_path VARCHAR,
    p_features_used JSONB,
    p_hyperparameters JSONB,
    p_performance_metrics JSONB,
    p_is_production BOOLEAN DEFAULT FALSE
)
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO models.model_metadata (
        model_id,
        model_name,
        model_version,
        model_type,
        model_path,
        training_date,
        features_used,
        hyperparameters,
        performance_metrics,
        is_production,
        training_data_range
    ) VALUES (
        p_model_id,
        p_model_name,
        p_model_version,
        p_model_type,
        p_model_path,
        CURRENT_DATE,
        p_features_used,
        p_hyperparameters,
        p_performance_metrics,
        p_is_production,
        ARRAY[CURRENT_DATE - INTERVAL '30 days', CURRENT_DATE]::DATE[]
    )
    ON CONFLICT (model_id) DO UPDATE 
    SET 
        model_name = EXCLUDED.model_name,
        model_version = EXCLUDED.model_version,
        model_path = EXCLUDED.model_path,
        features_used = EXCLUDED.features_used,
        hyperparameters = EXCLUDED.hyperparameters,
        performance_metrics = EXCLUDED.performance_metrics,
        is_production = EXCLUDED.is_production,
        updated_at = CURRENT_TIMESTAMP;
    
    -- Если новая модель становится production, деактивируем старые
    IF p_is_production THEN
        UPDATE models.model_metadata
        SET is_production = FALSE
        WHERE model_id != p_model_id
          AND model_type = p_model_type;
    END IF;
    
    RAISE NOTICE 'Метаданные модели % сохранены', p_model_id;
END;
$$;

-- =============================================
-- 5. Материализованные представления для отчетов
-- =============================================

-- Материализованное представление для ежедневной статистики
CREATE MATERIALIZED VIEW monitoring.daily_statistics AS
SELECT 
    DATE(cleaned_at) as statistic_date,
    COUNT(*) as total_records,
    SUM(CASE WHEN is_valid THEN 1 ELSE 0 END) as valid_records,
    SUM(CASE WHEN is_valid = FALSE THEN 1 ELSE 0 END) as invalid_records,
    AVG(air_temperature_k) as avg_air_temp,
    AVG(process_temperature_k) as avg_process_temp,
    SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) as failure_count,
    ROUND(100.0 * SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as failure_rate_percent
FROM cleaned.equipment_cleaned
GROUP BY DATE(cleaned_at)
ORDER BY statistic_date DESC;

-- Индекс для ускорения обновления
CREATE UNIQUE INDEX idx_daily_statistics_date 
ON monitoring.daily_statistics(statistic_date);

-- Процедура для обновления материализованных представлений
CREATE OR REPLACE PROCEDURE monitoring.refresh_materialized_views()
LANGUAGE plpgsql
AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY monitoring.daily_statistics;
    RAISE NOTICE 'Материализованные представления обновлены';
END;
$$;

-- =============================================
-- 6. Утилиты для обслуживания БД
-- =============================================

-- Процедура для очистки старых данных
CREATE OR REPLACE PROCEDURE maintenance.cleanup_old_data(
    p_retention_days INTEGER DEFAULT 365
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_deleted_raw INTEGER;
    v_deleted_cleaned INTEGER;
BEGIN
    -- Удаление старых сырых данных
    DELETE FROM raw.equipment_data
    WHERE timestamp < CURRENT_DATE - p_retention_days
    RETURNING COUNT(*) INTO v_deleted_raw;
    
    -- Удаление старых очищенных данных (сохраняем дольше)
    DELETE FROM cleaned.equipment_cleaned
    WHERE cleaned_at < CURRENT_DATE - (p_retention_days * 2)
    RETURNING COUNT(*) INTO v_deleted_cleaned;
    
    RAISE NOTICE 'Удалено старых данных: raw=%, cleaned=%', v_deleted_raw, v_deleted_cleaned;
    
    -- Vacuum для освобождения пространства
    VACUUM ANALYZE raw.equipment_data;
    VACUUM ANALYZE cleaned.equipment_cleaned;
END;
$$;