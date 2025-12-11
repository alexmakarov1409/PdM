import redis
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OnlineFeatureStore:
    """Online Feature Store на основе Redis"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, 
                 db: int = 0, password: str = None):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False  # Для работы с binary данными
        )
        self.ttl = 3600 * 24 * 7  # TTL 7 дней
        logger.info(f"Connected to Redis at {host}:{port}")
    
    def _get_key(self, equipment_id: str, feature_set: str = "latest") -> str:
        """Генерация ключа Redis"""
        return f"features:{equipment_id}:{feature_set}"
    
    def _get_metadata_key(self, equipment_id: str) -> str:
        """Ключ для метаданных"""
        return f"metadata:{equipment_id}"
    
    def store_features(self, equipment_id: str, features: Dict[str, Any], 
                      timestamp: datetime = None):
        """Сохранение признаков в online store"""
        try:
            key = self._get_key(equipment_id)
            
            # Добавляем timestamp к признакам
            feature_data = {
                'features': features,
                'timestamp': timestamp.isoformat() if timestamp else datetime.utcnow().isoformat(),
                'ingestion_time': datetime.utcnow().isoformat()
            }
            
            # Сериализация данных
            serialized_data = pickle.dumps(feature_data)
            
            # Сохранение в Redis
            self.redis_client.setex(key, self.ttl, serialized_data)
            
            # Сохранение метаданных
            metadata_key = self._get_metadata_key(equipment_id)
            metadata = {
                'last_update': datetime.utcnow().isoformat(),
                'feature_count': len(features),
                'feature_names': list(features.keys())
            }
            self.redis_client.setex(
                metadata_key, 
                self.ttl, 
                json.dumps(metadata)
            )
            
            logger.info(f"Features stored for {equipment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store features: {str(e)}")
            return False
    
    def get_features(self, equipment_id: str, 
                    feature_names: List[str] = None) -> Dict[str, Any]:
        """Получение признаков из online store"""
        try:
            key = self._get_key(equipment_id)
            data = self.redis_client.get(key)
            
            if data:
                # Десериализация данных
                feature_data = pickle.loads(data)
                features = feature_data['features']
                
                # Фильтрация по нужным признакам
                if feature_names:
                    filtered_features = {
                        k: v for k, v in features.items() 
                        if k in feature_names
                    }
                    return {
                        'features': filtered_features,
                        'timestamp': feature_data['timestamp'],
                        'ingestion_time': feature_data['ingestion_time']
                    }
                
                return feature_data
            
            logger.warning(f"No features found for {equipment_id}")
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get features: {str(e)}")
            return {}
    
    def get_multiple_features(self, equipment_ids: List[str], 
                            feature_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Пакетное получение признаков для нескольких equipment_id"""
        try:
            # Используем pipeline для оптимизации
            pipeline = self.redis_client.pipeline()
            
            for eq_id in equipment_ids:
                key = self._get_key(eq_id)
                pipeline.get(key)
            
            results = pipeline.execute()
            batch_features = {}
            
            for eq_id, data in zip(equipment_ids, results):
                if data:
                    feature_data = pickle.loads(data)
                    features = feature_data['features']
                    
                    if feature_names:
                        filtered_features = {
                            k: v for k, v in features.items() 
                            if k in feature_names
                        }
                        batch_features[eq_id] = {
                            'features': filtered_features,
                            'timestamp': feature_data['timestamp']
                        }
                    else:
                        batch_features[eq_id] = feature_data
            
            return batch_features
            
        except Exception as e:
            logger.error(f"Failed to get batch features: {str(e)}")
            return {}
    
    def store_feature_vector(self, equipment_id: str, 
                           feature_vector: List[float],
                           feature_names: List[str]):
        """Сохранение вектора признаков"""
        try:
            # Преобразуем вектор в словарь
            features = dict(zip(feature_names, feature_vector))
            return self.store_features(equipment_id, features)
            
        except Exception as e:
            logger.error(f"Failed to store feature vector: {str(e)}")
            return False
    
    def get_feature_vector(self, equipment_id: str, 
                          feature_names: List[str]) -> List[float]:
        """Получение вектора признаков"""
        try:
            features_data = self.get_features(equipment_id, feature_names)
            
            if features_data and 'features' in features_data:
                features = features_data['features']
                # Возвращаем признаки в том же порядке
                return [features.get(name, 0.0) for name in feature_names]
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get feature vector: {str(e)}")
            return []
    
    def check_freshness(self, equipment_id: str, 
                       max_age_minutes: int = 5) -> bool:
        """Проверка свежести признаков"""
        try:
            metadata_key = self._get_metadata_key(equipment_id)
            metadata_json = self.redis_client.get(metadata_key)
            
            if metadata_json:
                metadata = json.loads(metadata_json)
                last_update = datetime.fromisoformat(metadata['last_update'])
                age = datetime.utcnow() - last_update
                
                return age.total_seconds() < (max_age_minutes * 60)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check freshness: {str(e)}")
            return False
    
    def delete_features(self, equipment_id: str):
        """Удаление признаков для equipment_id"""
        try:
            key = self._get_key(equipment_id)
            metadata_key = self._get_metadata_key(equipment_id)
            
            self.redis_client.delete(key)
            self.redis_client.delete(metadata_key)
            
            logger.info(f"Features deleted for {equipment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete features: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики по online store"""
        try:
            # Получаем все ключи с префиксом features:
            keys = self.redis_client.keys("features:*")
            
            total_keys = len(keys)
            total_size = 0
            
            for key in keys:
                size = self.redis_client.memory_usage(key)
                if size:
                    total_size += size
            
            return {
                'total_keys': total_keys,
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'avg_ttl': self.ttl
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {}