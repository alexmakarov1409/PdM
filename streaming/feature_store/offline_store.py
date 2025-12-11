import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OfflineFeatureStore:
    """Offline Feature Store на основе PostgreSQL"""
    
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)
        self.create_schema()
    
    def create_schema(self):
        """Создание схемы для Feature Store"""
        create_tables_sql = """
        CREATE TABLE IF NOT EXISTS feature_registry (
            feature_id SERIAL PRIMARY KEY,
            feature_name VARCHAR(100) UNIQUE NOT NULL,
            feature_type VARCHAR(50) NOT NULL,
            description TEXT,
            source_table VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS feature_values_offline (
            id SERIAL PRIMARY KEY,
            equipment_id VARCHAR(50) NOT NULL,
            feature_name VARCHAR(100) NOT NULL,
            feature_value JSONB NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(equipment_id, feature_name, timestamp)
        );
        
        CREATE TABLE IF NOT EXISTS training_datasets (
            dataset_id SERIAL PRIMARY KEY,
            dataset_name VARCHAR(100) UNIQUE NOT NULL,
            feature_list JSONB NOT NULL,
            start_date TIMESTAMP,
            end_date TIMESTAMP,
            sample_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_feature_values_offline 
        ON feature_values_offline(equipment_id, timestamp);
        
        CREATE INDEX IF NOT EXISTS idx_feature_values_name 
        ON feature_values_offline(feature_name);
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_tables_sql))
                conn.commit()
            logger.info("Feature Store schema created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to create schema: {str(e)}")
    
    def register_feature(self, feature_name: str, feature_type: str, 
                        description: str = "", source_table: str = None):
        """Регистрация нового признака в реестре"""
        insert_sql = """
        INSERT INTO feature_registry (feature_name, feature_type, description, source_table)
        VALUES (:feature_name, :feature_type, :description, :source_table)
        ON CONFLICT (feature_name) 
        DO UPDATE SET 
            feature_type = EXCLUDED.feature_type,
            description = EXCLUDED.description,
            source_table = EXCLUDED.source_table,
            updated_at = CURRENT_TIMESTAMP
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), {
                    'feature_name': feature_name,
                    'feature_type': feature_type,
                    'description': description,
                    'source_table': source_table
                })
                conn.commit()
            logger.info(f"Feature '{feature_name}' registered successfully")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to register feature: {str(e)}")
            return False
    
    def ingest_features(self, equipment_id: str, features: Dict[str, Any], 
                       timestamp: datetime):
        """Ингестирование признаков в offline store"""
        insert_sql = """
        INSERT INTO feature_values_offline (equipment_id, feature_name, feature_value, timestamp)
        VALUES (:equipment_id, :feature_name, :feature_value, :timestamp)
        ON CONFLICT (equipment_id, feature_name, timestamp) 
        DO UPDATE SET 
            feature_value = EXCLUDED.feature_value,
            created_at = CURRENT_TIMESTAMP
        """
        
        try:
            with self.engine.connect() as conn:
                for feature_name, feature_value in features.items():
                    conn.execute(text(insert_sql), {
                        'equipment_id': equipment_id,
                        'feature_name': feature_name,
                        'feature_value': json.dumps(feature_value) if not isinstance(feature_value, (int, float, bool)) else feature_value,
                        'timestamp': timestamp
                    })
                conn.commit()
            logger.info(f"Features ingested for {equipment_id} at {timestamp}")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to ingest features: {str(e)}")
            return False
    
    def get_features_for_training(self, equipment_id: str, 
                                 start_date: datetime, 
                                 end_date: datetime,
                                 feature_list: List[str] = None) -> pd.DataFrame:
        """Получение признаков для обучения модели"""
        if feature_list:
            features_str = ", ".join([f"'{f}'" for f in feature_list])
            where_clause = f"AND feature_name IN ({features_str})"
        else:
            where_clause = ""
        
        query = f"""
        SELECT 
            equipment_id,
            timestamp,
            feature_name,
            feature_value
        FROM feature_values_offline
        WHERE equipment_id = :equipment_id
            AND timestamp BETWEEN :start_date AND :end_date
            {where_clause}
        ORDER BY timestamp
        """
        
        try:
            df = pd.read_sql_query(
                text(query),
                self.engine,
                params={
                    'equipment_id': equipment_id,
                    'start_date': start_date,
                    'end_date': end_date
                }
            )
            
            # Преобразование в wide format
            if not df.empty:
                df['feature_value'] = df['feature_value'].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )
                df_wide = df.pivot_table(
                    index=['equipment_id', 'timestamp'],
                    columns='feature_name',
                    values='feature_value'
                ).reset_index()
                return df_wide
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to get features for training: {str(e)}")
            return pd.DataFrame()
    
    def create_training_dataset(self, dataset_name: str, feature_list: List[str],
                               start_date: datetime, end_date: datetime):
        """Создание версионированного датасета для обучения"""
        # Получение всех equipment_id
        equipment_query = """
        SELECT DISTINCT equipment_id 
        FROM feature_values_offline 
        WHERE timestamp BETWEEN :start_date AND :end_date
        """
        
        try:
            with self.engine.connect() as conn:
                # Получаем все equipment_id
                equipment_ids = pd.read_sql_query(
                    text(equipment_query),
                    conn,
                    params={'start_date': start_date, 'end_date': end_date}
                )['equipment_id'].tolist()
                
                all_data = []
                for eq_id in equipment_ids:
                    features_df = self.get_features_for_training(
                        eq_id, start_date, end_date, feature_list
                    )
                    if not features_df.empty:
                        all_data.append(features_df)
                
                if all_data:
                    final_df = pd.concat(all_data, ignore_index=True)
                    
                    # Сохраняем метаданные датасета
                    insert_sql = """
                    INSERT INTO training_datasets 
                    (dataset_name, feature_list, start_date, end_date, sample_count)
                    VALUES (:dataset_name, :feature_list, :start_date, :end_date, :sample_count)
                    """
                    
                    conn.execute(text(insert_sql), {
                        'dataset_name': dataset_name,
                        'feature_list': json.dumps(feature_list),
                        'start_date': start_date,
                        'end_date': end_date,
                        'sample_count': len(final_df)
                    })
                    conn.commit()
                    
                    # Сохраняем сам датасет в отдельную таблицу
                    table_name = f"dataset_{dataset_name}"
                    final_df.to_sql(table_name, conn, if_exists='replace', index=False)
                    
                    logger.info(f"Training dataset '{dataset_name}' created with {len(final_df)} samples")
                    return final_df
                
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to create training dataset: {str(e)}")
            return pd.DataFrame()
    
    def get_feature_statistics(self, feature_name: str) -> Dict[str, Any]:
        """Получение статистики по признаку"""
        stats_query = """
        SELECT 
            COUNT(*) as count,
            MIN(timestamp) as min_timestamp,
            MAX(timestamp) as max_timestamp,
            COUNT(DISTINCT equipment_id) as unique_equipment
        FROM feature_values_offline
        WHERE feature_name = :feature_name
        """
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(stats_query),
                    {'feature_name': feature_name}
                ).fetchone()
                
                if result:
                    return {
                        'count': result[0],
                        'min_timestamp': result[1],
                        'max_timestamp': result[2],
                        'unique_equipment': result[3]
                    }
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get feature statistics: {str(e)}")
            return {}