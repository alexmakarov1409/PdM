from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from offline_store import OfflineFeatureStore
from online_store import OnlineFeatureStore
from feature_registry import FeatureRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureServingService:
    """Сервис для сервинга признаков"""
    
    def __init__(self, offline_store: OfflineFeatureStore,
                 online_store: OnlineFeatureStore,
                 feature_registry: FeatureRegistry):
        self.offline_store = offline_store
        self.online_store = online_store
        self.registry = feature_registry
    
    def get_features_for_inference(self, equipment_id: str, 
                                 feature_names: List[str],
                                 use_cache: bool = True) -> Dict[str, Any]:
        """Получение признаков для инференса"""
        try:
            features = {}
            
            if use_cache:
                # Пытаемся получить из online store
                online_data = self.online_store.get_features(
                    equipment_id, feature_names
                )
                
                if online_data and 'features' in online_data:
                    features = online_data['features']
                    
                    # Проверяем свежесть данных
                    is_fresh = self.online_store.check_freshness(
                        equipment_id, max_age_minutes=5
                    )
                    
                    if is_fresh:
                        logger.info(f"Using fresh features from online store for {equipment_id}")
                        return {
                            'features': features,
                            'source': 'online_store',
                            'timestamp': online_data.get('timestamp'),
                            'fresh': True
                        }
            
            # Если нет в online store или данные устарели, берем из offline store
            logger.info(f"Fetching features from offline store for {equipment_id}")
            
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(hours=24)  # Последние 24 часа
            
            features_df = self.offline_store.get_features_for_training(
                equipment_id, start_date, end_date, feature_names
            )
            
            if not features_df.empty:
                # Берем последнюю запись
                latest = features_df.iloc[-1]
                timestamp = latest['timestamp']
                
                # Преобразуем в словарь
                features = {
                    col: latest[col] for col in feature_names 
                    if col in latest and pd.notna(latest[col])
                }
                
                # Обновляем online store для будущих запросов
                self.online_store.store_features(
                    equipment_id, features, timestamp
                )
                
                return {
                    'features': features,
                    'source': 'offline_store',
                    'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                    'fresh': False
                }
            
            logger.warning(f"No features found for {equipment_id}")
            return {
                'features': {},
                'source': 'none',
                'timestamp': None,
                'fresh': False
            }
            
        except Exception as e:
            logger.error(f"Failed to get features for inference: {str(e)}")
            return {
                'features': {},
                'source': 'error',
                'error': str(e)
            }
    
    def get_feature_vector_for_inference(self, equipment_id: str,
                                       feature_names: List[str]) -> List[float]:
        """Получение вектора признаков для инференса"""
        try:
            features_data = self.get_features_for_inference(
                equipment_id, feature_names
            )
            
            features = features_data.get('features', {})
            
            # Преобразуем в вектор в правильном порядке
            feature_vector = []
            for name in feature_names:
                value = features.get(name, 0.0)
                
                # Проверяем, что значение числовое
                if isinstance(value, (int, float)):
                    feature_vector.append(float(value))
                else:
                    # Пытаемся преобразовать
                    try:
                        feature_vector.append(float(value))
                    except:
                        feature_vector.append(0.0)
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Failed to get feature vector: {str(e)}")
            return []
    
    def batch_get_features(self, equipment_ids: List[str],
                          feature_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Пакетное получение признаков"""
        try:
            results = {}
            
            # Пытаемся получить все из online store
            batch_features = self.online_store.get_multiple_features(
                equipment_ids, feature_names
            )
            
            for eq_id in equipment_ids:
                if eq_id in batch_features:
                    results[eq_id] = {
                        'features': batch_features[eq_id]['features'],
                        'source': 'online_store',
                        'timestamp': batch_features[eq_id].get('timestamp')
                    }
                else:
                    # Берем из offline store
                    features_data = self.get_features_for_inference(
                        eq_id, feature_names
                    )
                    results[eq_id] = features_data
            
            return results
            
        except Exception as e:
            logger.error(f"Failed batch get features: {str(e)}")
            return {}
    
    def get_feature_statistics(self, equipment_id: str, 
                              feature_names: List[str],
                              lookback_days: int = 30) -> Dict[str, Any]:
        """Получение статистики по признакам за период"""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=lookback_days)
            
            features_df = self.offline_store.get_features_for_training(
                equipment_id, start_date, end_date, feature_names
            )
            
            if features_df.empty:
                return {}
            
            stats = {}
            for feature in feature_names:
                if feature in features_df.columns:
                    values = features_df[feature].dropna()
                    if len(values) > 0:
                        stats[feature] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'count': int(len(values))
                        }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get feature statistics: {str(e)}")
            return {}
    
    def validate_features(self, equipment_id: str, 
                         features: Dict[str, Any]) -> Dict[str, Any]:
        """Валидация признаков перед инференсом"""
        try:
            validation_results = {}
            all_valid = True
            
            for feature_name, value in features.items():
                validation = self.registry.validate_feature_value(feature_name, value)
                validation_results[feature_name] = validation
                
                if not validation['valid']:
                    all_valid = False
            
            return {
                'all_valid': all_valid,
                'validation_results': validation_results,
                'valid_features': all_valid
            }
            
        except Exception as e:
            logger.error(f"Failed to validate features: {str(e)}")
            return {'all_valid': False, 'error': str(e)}
    
    def get_feature_drift(self, equipment_id: str, 
                         feature_names: List[str],
                         reference_start: datetime,
                         reference_end: datetime,
                         current_start: datetime,
                         current_end: datetime) -> Dict[str, Any]:
        """Вычисление дрифта признаков"""
        try:
            # Получаем данные за референсный период
            reference_df = self.offline_store.get_features_for_training(
                equipment_id, reference_start, reference_end, feature_names
            )
            
            # Получаем данные за текущий период
            current_df = self.offline_store.get_features_for_training(
                equipment_id, current_start, current_end, feature_names
            )
            
            if reference_df.empty or current_df.empty:
                return {}
            
            drift_results = {}
            
            for feature in feature_names:
                if feature in reference_df.columns and feature in current_df.columns:
                    ref_values = reference_df[feature].dropna()
                    curr_values = current_df[feature].dropna()
                    
                    if len(ref_values) > 0 and len(curr_values) > 0:
                        # Вычисляем статистики
                        ref_mean = ref_values.mean()
                        curr_mean = curr_values.mean()
                        
                        # Вычисляем дрифт (разница в средних)
                        mean_drift = abs(curr_mean - ref_mean) / (ref_mean + 1e-10)
                        
                        # KS-тест для распределений
                        from scipy import stats
                        if len(ref_values) > 30 and len(curr_values) > 30:
                            ks_stat, ks_pvalue = stats.ks_2samp(ref_values, curr_values)
                        else:
                            ks_stat, ks_pvalue = 0, 1
                        
                        drift_results[feature] = {
                            'reference_mean': float(ref_mean),
                            'current_mean': float(curr_mean),
                            'mean_drift': float(mean_drift),
                            'ks_statistic': float(ks_stat),
                            'ks_pvalue': float(ks_pvalue),
                            'drift_detected': ks_pvalue < 0.05  # 95% confidence
                        }
            
            return drift_results
            
        except Exception as e:
            logger.error(f"Failed to compute feature drift: {str(e)}")
            return {}