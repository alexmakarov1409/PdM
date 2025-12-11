from kafka import KafkaConsumer, KafkaProducer
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional
import logging
import threading
import statistics

from kafka_config import KafkaConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamProcessor:
    """Обработчик потоковых данных с оконными агрегациями"""
    
    def __init__(self, input_topic: str, output_topic: str, config: KafkaConfig):
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.config = config
        
        # Оконные буферы
        self.window_size = 60  # секунд
        self.windows = defaultdict(lambda: defaultdict(deque))
        
        # Статистика
        self.stats = {
            'windows_processed': 0,
            'alerts_generated': 0,
            'anomalies_detected': 0
        }
        
        # Инициализация Kafka клиентов
        self.consumer = None
        self.producer = None
        
        # Пороги для алертов
        self.alert_thresholds = {
            'temperature': {'warning': 80, 'critical': 90},
            'vibration': {'warning': 60, 'critical': 75},
            'pressure': {'warning': 140, 'critical': 160}
        }
    
    def connect(self):
        """Подключение к Kafka"""
        try:
            # Consumer для входного топика
            consumer_config = self.config.get_consumer_config()
            consumer_config['group_id'] = f'{consumer_config["group_id"]}-processor'
            self.consumer = KafkaConsumer(**consumer_config)
            self.consumer.subscribe([self.input_topic])
            
            # Producer для выходного топика
            producer_config = self.config.get_producer_config()
            self.producer = KafkaProducer(**producer_config)
            
            logger.info(f"Stream processor connected to Kafka")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {str(e)}")
            return False
    
    def process_window(self, equipment_id: str, sensor_type: str, 
                      values: List[float]) -> Dict[str, Any]:
        """Обработка окна значений"""
        try:
            if not values:
                return {}
            
            # Вычисление статистик
            stats = {
                'count': len(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'range': max(values) - min(values)
            }
            
            # Обнаружение аномалий (правило 3 сигм)
            if stats['std'] > 0:
                upper_bound = stats['mean'] + 3 * stats['std']
                lower_bound = stats['mean'] - 3 * stats['std']
                
                anomalies = [v for v in values if v > upper_bound or v < lower_bound]
                if anomalies:
                    stats['anomalies'] = anomalies
                    stats['anomaly_count'] = len(anomalies)
                    self.stats['anomalies_detected'] += 1
            
            # Проверка порогов для алертов
            alert_level = 'normal'
            if sensor_type in self.alert_thresholds:
                thresholds = self.alert_thresholds[sensor_type]
                max_value = max(values)
                
                if max_value >= thresholds['critical']:
                    alert_level = 'critical'
                    self._generate_alert(equipment_id, sensor_type, max_value, 'critical')
                elif max_value >= thresholds['warning']:
                    alert_level = 'warning'
                    self._generate_alert(equipment_id, sensor_type, max_value, 'warning')
            
            stats['alert_level'] = alert_level
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to process window: {str(e)}")
            return {}
    
    def _generate_alert(self, equipment_id: str, sensor_type: str, 
                       value: float, level: str):
        """Генерация алерта"""
        alert = {
            'alert_id': f"alert_{int(time.time())}_{equipment_id}",
            'equipment_id': equipment_id,
            'sensor_type': sensor_type,
            'value': value,
            'level': level,
            'timestamp': datetime.utcnow().isoformat(),
            'message': f"{sensor_type} {level} alert: {value:.2f}"
        }
        
        # Отправка алерта в отдельный топик
        self.producer.send(
            topic='alerts',
            value=alert,
            key=equipment_id
        )
        
        self.stats['alerts_generated'] += 1
        logger.warning(f"Alert generated: {alert['message']}")
    
    def add_to_window(self, equipment_id: str, sensor_data: Dict[str, Any], 
                     timestamp: datetime):
        """Добавление данных в окно"""
        try:
            current_time = timestamp.timestamp()
            
            for sensor_type, value in sensor_data.items():
                if isinstance(value, (int, float)):
                    # Добавляем значение в окно
                    window = self.windows[equipment_id][sensor_type]
                    window.append((current_time, value))
                    
                    # Удаляем старые значения (старше window_size)
                    while window and window[0][0] < current_time - self.window_size:
                        window.popleft()
            
        except Exception as e:
            logger.error(f"Failed to add to window: {str(e)}")
    
    def process_windows(self):
        """Обработка всех окон"""
        try:
            results = {}
            
            for equipment_id, sensors in self.windows.items():
                equipment_results = {}
                
                for sensor_type, window in sensors.items():
                    if window:
                        # Извлекаем только значения из окна
                        values = [v for _, v in window]
                        
                        # Обработка окна
                        window_stats = self.process_window(equipment_id, sensor_type, values)
                        if window_stats:
                            equipment_results[sensor_type] = window_stats
                
                if equipment_results:
                    results[equipment_id] = equipment_results
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to process windows: {str(e)}")
            return {}
    
    def send_aggregated_results(self, results: Dict[str, Any]):
        """Отправка агрегированных результатов в выходной топик"""
        try:
            for equipment_id, equipment_results in results.items():
                message = {
                    'equipment_id': equipment_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'aggregation_period': self.window_size,
                    'results': equipment_results,
                    'metadata': {
                        'processing_time': time.time(),
                        'version': '1.0'
                    }
                }
                
                self.producer.send(
                    topic=self.output_topic,
                    value=message,
                    key=equipment_id
                )
                
                self.stats['windows_processed'] += 1
            
            logger.debug(f"Sent {len(results)} aggregated results")
            
        except Exception as e:
            logger.error(f"Failed to send aggregated results: {str(e)}")
    
    def start_processing(self, window_interval: int = 10):
        """Запуск потоковой обработки"""
        if not self.connect():
            logger.error("Failed to connect to Kafka")
            return
        
        logger.info(f"Starting stream processing with {window_interval}s intervals")
        
        # Таймер для обработки окон
        def window_timer():
            while True:
                time.sleep(window_interval)
                
                # Обработка окон
                results = self.process_windows()
                
                if results:
                    # Отправка результатов
                    self.send_aggregated_results(results)
                    
                    # Периодический отчет
                    if self.stats['windows_processed'] % 10 == 0:
                        logger.info(
                            f"Stats - Windows: {self.stats['windows_processed']}, "
                            f"Alerts: {self.stats['alerts_generated']}, "
                            f"Anomalies: {self.stats['anomalies_detected']}"
                        )
        
        # Запуск таймера в отдельном потоке
        timer_thread = threading.Thread(target=window_timer, daemon=True)
        timer_thread.start()
        
        # Основной цикл обработки сообщений
        try:
            while True:
                # Чтение сообщений
                batch = self.consumer.poll(timeout_ms=1000)
                
                if not batch:
                    continue
                
                for topic_partition, messages in batch.items():
                    for message in messages:
                        try:
                            data = message.value
                            equipment_id = data.get('equipment_id')
                            sensor_data = data.get('data', {})
                            timestamp_str = data.get('timestamp')
                            
                            if equipment_id and sensor_data and timestamp_str:
                                timestamp = datetime.fromisoformat(
                                    timestamp_str.replace('Z', '+00:00')
                                )
                                
                                # Добавление в окно
                                self.add_to_window(equipment_id, sensor_data, timestamp)
                                
                        except Exception as e:
                            logger.error(f"Error processing message: {str(e)}")
        
        except KeyboardInterrupt:
            logger.info("Stream processing stopped by user")
        except Exception as e:
            logger.error(f"Error in processing loop: {str(e)}")
        finally:
            self.stop()
    
    def stop(self):
        """Остановка обработчика"""
        if self.consumer:
            self.consumer.close()
        
        if self.producer:
            self.producer.flush()
            self.producer.close()
        
        logger.info("Stream processor stopped")
        
        # Вывод финальной статистики
        logger.info("=" * 50)
        logger.info("STREAM PROCESSING FINAL STATS")
        logger.info("=" * 50)
        logger.info(f"Windows processed: {self.stats['windows_processed']}")
        logger.info(f"Alerts generated: {self.stats['alerts_generated']}")
        logger.info(f"Anomalies detected: {self.stats['anomalies_detected']}")
        logger.info("=" * 50)

# Пример использования
if __name__ == "__main__":
    config = KafkaConfig(
        bootstrap_servers=['localhost:9092'],
        topic_name='sensor-data'
    )
    
    processor = StreamProcessor(
        input_topic='sensor-data',
        output_topic='aggregated-sensor-data',
        config=config
    )
    
    try:
        processor.start_processing(window_interval=10)
    except KeyboardInterrupt:
        logger.info("Processor stopped")
    except Exception as e:
        logger.error(f"Processor error: {str(e)}")
    finally:
        processor.stop()