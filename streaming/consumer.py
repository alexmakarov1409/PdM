from kafka import KafkaConsumer
from kafka.errors import KafkaError
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import signal
import sys

from kafka_config import KafkaConfig
from feature_store.online_store import OnlineFeatureStore
from feature_store.offline_store import OfflineFeatureStore
from feature_store.feature_registry import FeatureRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorDataConsumer:
    """Kafka Consumer для обработки данных с датчиков"""
    
    def __init__(self, config: KafkaConfig):
        self.config = config
        self.consumer = None
        self.running = False
        
        # Инициализация Feature Store
        offline_store = OfflineFeatureStore(
            connection_string="postgresql://postgres:password@localhost:5432/predictive_maintenance"
        )
        online_store = OnlineFeatureStore()
        feature_registry = FeatureRegistry()
        
        self.online_store = online_store
        self.offline_store = offline_store
        self.feature_registry = feature_registry
        
        # Статистика
        self.stats = {
            'messages_processed': 0,
            'messages_failed': 0,
            'last_message_time': None,
            'processing_times': [],
            'start_time': None
        }
    
    def connect(self):
        """Подключение к Kafka"""
        try:
            consumer_config = self.config.get_consumer_config()
            self.consumer = KafkaConsumer(**consumer_config)
            
            # Подписываемся на топик
            self.consumer.subscribe([self.config.topic_name])
            
            logger.info(f"Connected to Kafka and subscribed to '{self.config.topic_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {str(e)}")
            return False
    
    def process_message(self, message: Dict[str, Any]) -> bool:
        """Обработка одного сообщения"""
        try:
            start_time = time.time()
            
            # Извлекаем данные
            equipment_id = message.get('equipment_id')
            timestamp = message.get('timestamp')
            sensor_data = message.get('data', {})
            
            if not equipment_id or not sensor_data:
                logger.warning(f"Invalid message format: {message}")
                return False
            
            # Валидация данных
            if not self._validate_sensor_data(sensor_data):
                logger.warning(f"Invalid sensor data for {equipment_id}")
                return False
            
            # Обогащение признаками
            enriched_data = self._enrich_with_features(equipment_id, sensor_data, timestamp)
            
            # Сохранение в Feature Store
            self._store_in_feature_store(equipment_id, enriched_data, timestamp)
            
            # Сохранение в сыром виде (для истории)
            self._store_raw_data(equipment_id, message)
            
            # Обновление статистики
            processing_time = time.time() - start_time
            self.stats['processing_times'].append(processing_time)
            self.stats['messages_processed'] += 1
            self.stats['last_message_time'] = datetime.utcnow().isoformat()
            
            # Логирование
            if self.stats['messages_processed'] % 100 == 0:
                avg_time = sum(self.stats['processing_times'][-100:]) / 100
                logger.info(
                    f"Processed {self.stats['messages_processed']} messages. "
                    f"Avg processing time: {avg_time*1000:.2f}ms"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process message: {str(e)}")
            self.stats['messages_failed'] += 1
            return False
    
    def _validate_sensor_data(self, sensor_data: Dict[str, Any]) -> bool:
        """Валидация данных датчиков"""
        required_fields = ['temperature', 'vibration', 'pressure']
        
        for field in required_fields:
            if field not in sensor_data:
                logger.warning(f"Missing required field: {field}")
                return False
            
            value = sensor_data[field]
            if not isinstance(value, (int, float)):
                logger.warning(f"Invalid type for {field}: {type(value)}")
                return False
            
            # Проверка диапазонов
            if field == 'temperature' and not (0 <= value <= 150):
                logger.warning(f"Temperature out of range: {value}")
                return False
            elif field == 'vibration' and not (0 <= value <= 100):
                logger.warning(f"Vibration out of range: {value}")
                return False
            elif field == 'pressure' and not (0 <= value <= 200):
                logger.warning(f"Pressure out of range: {value}")
                return False
        
        return True
    
    def _enrich_with_features(self, equipment_id: str, 
                             sensor_data: Dict[str, Any],
                             timestamp: str) -> Dict[str, Any]:
        """Обогащение данных дополнительными признаками"""
        try:
            # Базовые признаки из датчиков
            enriched = sensor_data.copy()
            
            # Добавляем timestamp признаки
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            enriched['hour'] = dt.hour
            enriched['day_of_week'] = dt.weekday()
            enriched['month'] = dt.month
            enriched['is_weekend'] = 1 if dt.weekday() >= 5 else 0
            
            # Вычисляем производные признаки
            enriched['temperature_pressure_ratio'] = (
                sensor_data['temperature'] / (sensor_data['pressure'] + 1e-10)
            )
            enriched['vibration_energy'] = sensor_data['vibration'] ** 2
            
            # Получаем исторические данные для вычисления трендов
            historical_features = self._get_historical_features(equipment_id)
            if historical_features:
                # Вычисляем скользящие средние
                for feature in ['temperature', 'vibration', 'pressure']:
                    if feature in historical_features:
                        values = historical_features[feature]
                        if len(values) >= 3:
                            enriched[f'{feature}_rolling_mean_3'] = sum(values[-3:]) / 3
                            enriched[f'{feature}_trend'] = values[-1] - values[0]
            
            return enriched
            
        except Exception as e:
            logger.error(f"Failed to enrich features: {str(e)}")
            return sensor_data
    
    def _get_historical_features(self, equipment_id: str) -> Dict[str, List[float]]:
        """Получение исторических данных для equipment"""
        try:
            # Получаем последние 10 записей из online store
            features_data = self.online_store.get_features(equipment_id)
            
            if features_data and 'features' in features_data:
                historical = {}
                for feature in ['temperature', 'vibration', 'pressure']:
                    if feature in features_data['features']:
                        # В реальной реализации здесь был бы запрос к offline store
                        historical[feature] = [features_data['features'][feature]]
                
                return historical
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get historical features: {str(e)}")
            return {}
    
    def _store_in_feature_store(self, equipment_id: str, 
                               features: Dict[str, Any],
                               timestamp: str):
        """Сохранение признаков в Feature Store"""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            # Сохранение в offline store
            self.offline_store.ingest_features(equipment_id, features, dt)
            
            # Сохранение в online store
            self.online_store.store_features(equipment_id, features, dt)
            
            logger.debug(f"Features stored for {equipment_id}")
            
        except Exception as e:
            logger.error(f"Failed to store in feature store: {str(e)}")
    
    def _store_raw_data(self, equipment_id: str, message: Dict[str, Any]):
        """Сохранение сырых данных (для отладки и аудита)"""
        try:
            # В реальной реализации здесь было бы сохранение в базу данных
            # Для примера просто логируем
            logger.debug(f"Raw data for {equipment_id}: {message.get('data', {})}")
            
        except Exception as e:
            logger.error(f"Failed to store raw data: {str(e)}")
    
    def start_consuming(self, batch_size: int = 100, timeout_ms: int = 1000):
        """Запуск потребления сообщений"""
        if not self.connect():
            logger.error("Failed to connect to Kafka")
            return
        
        self.running = True
        self.stats['start_time'] = datetime.utcnow().isoformat()
        
        logger.info("Starting to consume messages...")
        
        # Обработка сигналов для graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            while self.running:
                # Чтение пакета сообщений
                batch = self.consumer.poll(timeout_ms=timeout_ms)
                
                if not batch:
                    continue
                
                for topic_partition, messages in batch.items():
                    for message in messages:
                        try:
                            # Обработка сообщения
                            success = self.process_message(message.value)
                            
                            if success and not self.config.enable_auto_commit:
                                # Ручное подтверждение обработки
                                self.consumer.commit()
                                
                        except Exception as e:
                            logger.error(f"Error processing message: {str(e)}")
                            self.stats['messages_failed'] += 1
                
                # Периодический отчет о статистике
                if self.stats['messages_processed'] % 1000 == 0:
                    self._print_stats()
        
        except KeyboardInterrupt:
            logger.info("Consumption stopped by user")
        except Exception as e:
            logger.error(f"Error in consumption loop: {str(e)}")
        finally:
            self.stop()
    
    def _signal_handler(self, signum, frame):
        """Обработчик сигналов для graceful shutdown"""
        logger.info(f"Received signal {signum}, stopping consumer...")
        self.running = False
    
    def stop(self):
        """Остановка consumer"""
        self.running = False
        
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")
        
        self._print_final_stats()
    
    def _print_stats(self):
        """Вывод статистики"""
        if self.stats['processing_times']:
            avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
        else:
            avg_time = 0
        
        logger.info(
            f"Stats - Processed: {self.stats['messages_processed']}, "
            f"Failed: {self.stats['messages_failed']}, "
            f"Avg time: {avg_time*1000:.2f}ms"
        )
    
    def _print_final_stats(self):
        """Вывод финальной статистики"""
        total_time = None
        if self.stats['start_time']:
            start_dt = datetime.fromisoformat(self.stats['start_time'])
            total_time = (datetime.utcnow() - start_dt).total_seconds()
        
        logger.info("=" * 50)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total messages processed: {self.stats['messages_processed']}")
        logger.info(f"Total messages failed: {self.stats['messages_failed']}")
        
        if total_time and self.stats['messages_processed'] > 0:
            msg_per_sec = self.stats['messages_processed'] / total_time
            logger.info(f"Processing rate: {msg_per_sec:.2f} messages/sec")
        
        if self.stats['processing_times']:
            avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            max_time = max(self.stats['processing_times'])
            min_time = min(self.stats['processing_times'])
            
            logger.info(f"Processing time - Avg: {avg_time*1000:.2f}ms, "
                       f"Min: {min_time*1000:.2f}ms, "
                       f"Max: {max_time*1000:.2f}ms")
        
        logger.info("=" * 50)
    
    def get_topic_offsets(self):
        """Получение информации о смещениях в топике"""
        try:
            if not self.consumer:
                self.connect()
            
            # Получаем партиции топика
            partitions = self.consumer.partitions_for_topic(self.config.topic_name)
            
            offsets = {}
            for partition in partitions:
                # Текущее смещение consumer
                tp = TopicPartition(self.config.topic_name, partition)
                committed = self.consumer.committed(tp)
                
                # Последнее смещение в топике
                end_offsets = self.consumer.end_offsets([tp])
                end_offset = end_offsets[tp]
                
                offsets[partition] = {
                    'committed_offset': committed,
                    'end_offset': end_offset,
                    'lag': end_offset - (committed or 0)
                }
            
            return offsets
            
        except Exception as e:
            logger.error(f"Failed to get offsets: {str(e)}")
            return {}

# Пример использования
if __name__ == "__main__":
    config = KafkaConfig(
        bootstrap_servers=['localhost:9092'],
        topic_name='sensor-data',
        consumer_group_id='predictive-maintenance-consumer-1'
    )
    
    consumer = SensorDataConsumer(config)
    
    # Запуск consumer
    try:
        consumer.start_consuming(batch_size=100, timeout_ms=1000)
    except KeyboardInterrupt:
        logger.info("Consumer stopped")
    except Exception as e:
        logger.error(f"Consumer error: {str(e)}")
    finally:
        consumer.stop()