from dataclasses import dataclass
from typing import List
import json

@dataclass
class KafkaConfig:
    """Конфигурация Kafka"""
    bootstrap_servers: List[str] = None
    topic_name: str = None
    consumer_group_id: str = None
    auto_offset_reset: str = None
    enable_auto_commit: bool = None
    security_protocol: str = None
    sasl_mechanism: str = None
    sasl_username: str = None
    sasl_password: str = None
    
    def __post_init__(self):
        if self.bootstrap_servers is None:
            self.bootstrap_servers = ['localhost:9092']
        if self.topic_name is None:
            self.topic_name = 'sensor-data'
        if self.consumer_group_id is None:
            self.consumer_group_id = 'predictive-maintenance-group'
        if self.auto_offset_reset is None:
            self.auto_offset_reset = 'earliest'
        if self.enable_auto_commit is None:
            self.enable_auto_commit = False
    
    @classmethod
    def from_json(cls, json_path: str):
        """Загрузка конфигурации из JSON файла"""
        try:
            with open(json_path, 'r') as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        except FileNotFoundError:
            print(f"Config file {json_path} not found, using defaults")
            return cls()
    
    def to_dict(self) -> dict:
        """Преобразование в словарь"""
        return {
            'bootstrap_servers': self.bootstrap_servers,
            'topic_name': self.topic_name,
            'consumer_group_id': self.consumer_group_id,
            'auto_offset_reset': self.auto_offset_reset,
            'enable_auto_commit': self.enable_auto_commit,
            'security_protocol': self.security_protocol,
            'sasl_mechanism': self.sasl_mechanism,
            'sasl_username': self.sasl_username,
            'sasl_password': self.sasl_password
        }
    
    def get_producer_config(self) -> dict:
        """Конфигурация для Producer"""
        config = {
            'bootstrap_servers': ','.join(self.bootstrap_servers),
            'value_serializer': lambda v: json.dumps(v).encode('utf-8'),
            'key_serializer': lambda k: json.dumps(k).encode('utf-8') if k else None,
        }
        
        # Добавляем security конфигурацию если есть
        if self.security_protocol:
            config.update({
                'security_protocol': self.security_protocol,
                'sasl_mechanism': self.sasl_mechanism,
                'sasl_plain_username': self.sasl_username,
                'sasl_plain_password': self.sasl_password
            })
        
        return config
    
    def get_consumer_config(self) -> dict:
        """Конфигурация для Consumer"""
        config = {
            'bootstrap_servers': ','.join(self.bootstrap_servers),
            'group_id': self.consumer_group_id,
            'auto_offset_reset': self.auto_offset_reset,
            'enable_auto_commit': self.enable_auto_commit,
            'value_deserializer': lambda v: json.loads(v.decode('utf-8')),
            'key_deserializer': lambda k: json.loads(k.decode('utf-8')) if k else None,
        }
        
        # Добавляем security конфигурацию если есть
        if self.security_protocol:
            config.update({
                'security_protocol': self.security_protocol,
                'sasl_mechanism': self.sasl_mechanism,
                'sasl_plain_username': self.sasl_username,
                'sasl_plain_password': self.sasl_password
            })
        
        return config

# Пример конфигурационного файла kafka_config.json
JSON_CONFIG = """
{
  "bootstrap_servers": ["localhost:9092"],
  "topic_name": "sensor-data",
  "consumer_group_id": "predictive-maintenance-group",
  "auto_offset_reset": "earliest",
  "enable_auto_commit": false,
  "security_protocol": null,
  "sasl_mechanism": null,
  "sasl_username": null,
  "sasl_password": null
}
"""