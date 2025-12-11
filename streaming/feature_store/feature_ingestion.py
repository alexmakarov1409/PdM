import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

from offline_store import OfflineFeatureStore
from online_store import OnlineFeatureStore
from feature_registry import FeatureRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureIngestionPipeline:
    """Пайплайн для инжекции признаков в Feature Store"""
    
    def __init__(self, offline_store: OfflineFeatureStore, 
                 online_store: OnlineFeatureStore,
                 feature_registry: FeatureRegistry):
        self.offline_store = offline_store
        self.online_store = online_store
        self.registry = feature_registry
    
    def ingest_from_dataframe(self, df: pd.DataFrame, 
                            equipment_id_col: str = 'equipment_id',
                            timestamp_col: str = 'timestamp'):
        """Ингестирование признаков из DataFrame"""
        try:
            if equipment_id_col not in df.columns:
                raise ValueError(f"Column {equipment_id_col} not found in DataFrame")
            if timestamp_col not in df.columns:
                raise ValueError(f"Column {timestamp_col} not found in DataFrame")
            
            # Группировка по equipment_id
            grouped = df.groupby(equipment_id_col)
            
            for equipment_id, group in grouped:
                # Сортировка по времени
                group = group.sort_values(timestamp_col)
                
                for _, row in group.iterrows():
                    timestamp = row[timestamp_col]
                    
                    # Извлекаем признаки (все кроме id и timestamp)
                    feature_columns = [
                        col for col in df.columns 
                        if col not in [equipment_id_col, timestamp_col]
                    ]
                    
                    features = {
                        col: row[col] for col in feature_columns
                    }
                    
                    # Валидация признаков
                    valid_features = {}
                    for feature_name, value in features.items():
                        validation = self.registry.validate_feature_value(feature_name, value)
                        if validation['valid']:
                            valid_features[feature_name] = value
                        else:
                            logger.warning(
                                f"Validation failed for {feature_name}: {validation['errors']}"
                            )
                    
                    # Ингестирование в offline store
                    if valid_features:
                        self.offline_store.ingest_features(
                            equipment_id, valid_features, timestamp
                        )
                        
                        # Ингестирование в online store (только последние данные)
                        if group.index[-1] == row.name:  # Последняя запись
                            self.online_store.store_features(
                                equipment_id, valid_features, timestamp
                            )
            
            logger.info(f"Ingested {len(df)} rows into Feature Store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest from DataFrame: {str(e)}")
            return False
    
    def ingest_from_database(self, query: str, connection_string: str,
                           equipment_id_col: str = 'equipment_id',
                           timestamp_col: str = 'timestamp'):
        """Ингестирование признаков из базы данных"""
        try:
            from sqlalchemy import create_engine
            
            engine = create_engine(connection_string)
            df = pd.read_sql_query(query, engine)
            
            return self.ingest_from_dataframe(df, equipment_id_col, timestamp_col)
            
        except Exception as e:
            logger.error(f"Failed to ingest from database: {str(e)}")
            return False
    
    def batch_ingestion(self, data_list: List[Dict[str, Any]]):
        """Пакетная инжекция данных"""
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for data in data_list:
                    future = executor.submit(
                        self._ingest_single_record, data
                    )
                    futures.append(future)
                
                # Ожидание завершения
                results = [f.result() for f in futures]
                
            success_count = sum(results)
            logger.info(f"Batch ingestion completed: {success_count}/{len(data_list)} successful")
            return success_count
            
        except Exception as e:
            logger.error(f"Failed batch ingestion: {str(e)}")
            return 0
    
    def _ingest_single_record(self, data: Dict[str, Any]) -> bool:
        """Ингестирование одной записи"""
        try:
            equipment_id = data.get('equipment_id')
            timestamp = data.get('timestamp', datetime.utcnow())
            
            if not equipment_id:
                logger.warning("Missing equipment_id in record")
                return False
            
            # Извлекаем признаки
            features = {k: v for k, v in data.items() 
                       if k not in ['equipment_id', 'timestamp']}
            
            # Ингестирование
            offline_success = self.offline_store.ingest_features(
                equipment_id, features, timestamp
            )
            
            online_success = self.online_store.store_features(
                equipment_id, features, timestamp
            )
            
            return offline_success and online_success
            
        except Exception as e:
            logger.error(f"Failed to ingest single record: {str(e)}")
            return False
    
    def sync_offline_to_online(self, equipment_ids: List[str] = None):
        """Синхронизация данных из offline в online store"""
        try:
            # Если equipment_ids не указаны, берем все
            if not equipment_ids:
                # Получаем все equipment_id из offline store
                query = "SELECT DISTINCT equipment_id FROM feature_values_offline"
                from sqlalchemy import create_engine
                engine = create_engine(self.offline_store.engine)
                equipment_ids_df = pd.read_sql_query(query, engine)
                equipment_ids = equipment_ids_df['equipment_id'].tolist()
            
            synced_count = 0
            
            for eq_id in equipment_ids:
                # Получаем последние данные из offline store
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=1)
                
                features_df = self.offline_store.get_features_for_training(
                    eq_id, start_date, end_date
                )
                
                if not features_df.empty:
                    # Берем последнюю запись
                    latest = features_df.iloc[-1]
                    timestamp = latest['timestamp']
                    
                    # Преобразуем в словарь признаков
                    features = latest.to_dict()
                    features.pop('equipment_id', None)
                    features.pop('timestamp', None)
                    
                    # Сохраняем в online store
                    self.online_store.store_features(
                        eq_id, features, timestamp
                    )
                    synced_count += 1
            
            logger.info(f"Synced {synced_count} equipment records to online store")
            return synced_count
            
        except Exception as e:
            logger.error(f"Failed to sync offline to online: {str(e)}")
            return 0
    
    def monitor_ingestion_pipeline(self) -> Dict[str, Any]:
        """Мониторинг пайплайна инжекции"""
        try:
            # Статистика offline store
            offline_stats_query = """
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT equipment_id) as unique_equipment,
                MIN(timestamp) as earliest_timestamp,
                MAX(timestamp) as latest_timestamp
            FROM feature_values_offline
            """
            
            from sqlalchemy import create_engine
            engine = create_engine(self.offline_store.engine)
            offline_stats = pd.read_sql_query(offline_stats_query, engine).iloc[0]
            
            # Статистика online store
            online_stats = self.online_store.get_stats()
            
            return {
                'offline_store': {
                    'total_records': int(offline_stats['total_records']),
                    'unique_equipment': int(offline_stats['unique_equipment']),
                    'earliest_timestamp': offline_stats['earliest_timestamp'],
                    'latest_timestamp': offline_stats['latest_timestamp']
                },
                'online_store': online_stats,
                'monitoring_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to monitor pipeline: {str(e)}")
            return {}