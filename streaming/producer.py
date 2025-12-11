from kafka import KafkaProducer
from kafka.errors import KafkaError
import json
import time
import random
from datetime import datetime
from typing import Dict, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
import uuid

from kafka_config import KafkaConfig
from sensor_simulator import SensorDataSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorDataProducer:
    """Kafka Producer для отправки данных с датчиков"""
    
    def __init__(self, config: KafkaConfig):
        self.config = config
        self.producer = None
        self.sensor_simulator = SensorDataSimulator()
        self.connect()
    
    def connect(self):
        """Подключение к Kafka"""
        try:
            producer_config = self.config.get_producer_config()
            self.producer = KafkaProducer(**producer_config)
            logger.info(f"Connected to Kafka at {self.config.bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {str(e)}")
            raise
    
    def send_message(self, message: Dict[str, Any], key: Optional[str] = None):
        """Отправка сообщения в Kafka"""
        try:
            if not self.producer:
                self.connect()
            
            # Добавляем метаданные
            message_with_metadata = {
                **message,
                'producer_timestamp': datetime.utcnow().isoformat(),
                'message_id': str(uuid.uuid4())
            }
            
            future = self.producer.send(
                topic=self.config.topic_name,
                value=message_with_metadata,
                key=key
            )
            
            # Асинхронное ожидание подтверждения
            future.add_callback(self._on_send_success)
            future.add_errback(self._on_send_error)
            
            return future
            
        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")
            return None
    
    def _on_send_success(self, record_metadata):
        """Коллбек при успешной отправке"""
        logger.debug(
            f"Message sent to topic={record_metadata.topic}, "
            f"partition={record_metadata.partition}, "
            f"offset={record_metadata.offset}"
        )
    
    def _on_send_error(self, exc):
        """Коллбек при ошибке отправки"""
        logger.error(f"Failed to send message: {exc}")
    
    def send_sensor_data(self, equipment_id: str, sensor_data: Dict[str, Any]):
        """Отправка данных с датчиков"""
        message = {
            'equipment_id': equipment_id,
            'timestamp': datetime.utcnow().isoformat(),
            'sensor_type': 'predictive_maintenance',
            'data': sensor_data,
            'version': '1.0'
        }
        
        # Используем equipment_id как ключ для партиционирования
        return self.send_message(message, key=equipment_id)
    
    def simulate_and_send(self, equipment_ids: List[str], 
                         interval_seconds: float = 1.0,
                         duration_minutes: Optional[float] = None):
        """Симуляция и отправка данных"""
        try:
            start_time = time.time()
            message_count = 0
            
            logger.info(f"Starting simulation for {len(equipment_ids)} equipment")
            logger.info(f"Interval: {interval_seconds}s, Duration: {duration_minutes}min")
            
            while True:
                # Проверка длительности
                if duration_minutes:
                    elapsed_minutes = (time.time() - start_time) / 60
                    if elapsed_minutes >= duration_minutes:
                        logger.info(f"Simulation completed after {elapsed_minutes:.1f} minutes")
                        break
                
                for eq_id in equipment_ids:
                    # Генерация данных датчиков
                    sensor_data = self.sensor_simulator.generate_sensor_data(eq_id)
                    
                    # Отправка в Kafka
                    future = self.send_sensor_data(eq_id, sensor_data)
                    message_count += 1
                    
                    # Логирование прогресса
                    if message_count % 100 == 0:
                        logger.info(f"Sent {message_count} messages")
                
                # Ожидание перед следующей отправкой
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Simulation stopped by user")
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
        finally:
            self.close()
            logger.info(f"Total messages sent: {message_count}")
    
    def send_batch(self, messages: List[Dict[str, Any]]):
        """Пакетная отправка сообщений"""
        try:
            futures = []
            for message in messages:
                eq_id = message.get('equipment_id')
                future = self.send_sensor_data(eq_id, message.get('data', {}))
                if future:
                    futures.append(future)
            
            # Ожидание подтверждения для всех сообщений
            for future in futures:
                future.get(timeout=10)
            
            logger.info(f"Successfully sent batch of {len(messages)} messages")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send batch: {str(e)}")
            return False
    
    def create_topic(self, topic_name: str, num_partitions: int = 3, 
                    replication_factor: int = 1):
        """Создание топика (требуются права администратора)"""
        try:
            from kafka.admin import KafkaAdminClient, NewTopic
            
            admin_client = KafkaAdminClient(
                bootstrap_servers=','.join(self.config.bootstrap_servers)
            )
            
            topic_list = [NewTopic(
                name=topic_name,
                num_partitions=num_partitions,
                replication_factor=replication_factor
            )]
            
            admin_client.create_topics(new_topics=topic_list, validate_only=False)
            logger.info(f"Topic '{topic_name}' created successfully")
            admin_client.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to create topic: {str(e)}")
            return False
    
    def get_topic_metadata(self, topic_name: str = None):
        """Получение метаданных топика"""
        try:
            if not topic_name:
                topic_name = self.config.topic_name
            
            # Используем producer для получения метаданных
            partitions = self.producer.partitions_for(topic_name)
            logger.info(f"Topic '{topic_name}' has partitions: {partitions}")
            return partitions
            
        except Exception as e:
            logger.error(f"Failed to get topic metadata: {str(e)}")
            return None
    
    def close(self):
        """Закрытие соединения с Kafka"""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Kafka producer closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Пример использования
if __name__ == "__main__":
    config = KafkaConfig(
        bootstrap_servers=['localhost:9092'],
        topic_name='sensor-data'
    )
    
    producer = SensorDataProducer(config)
    
    # Тестовая отправка
    test_data = {
        'temperature': 75.5,
        'vibration': 42.3,
        'pressure': 115.8,
        'rotation_speed': 2450.0,
        'power_consumption': 1250.5
    }
    
    # Отправка одного сообщения
    producer.send_sensor_data('EQ-TEST-001', test_data)
    
    # Симуляция потока данных
    equipment_ids = [f'EQ-{i:03d}' for i in range(1, 11)]
    producer.simulate_and_send(
        equipment_ids=equipment_ids,
        interval_seconds=0.5,
        duration_minutes=1
    )