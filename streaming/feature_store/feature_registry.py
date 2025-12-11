import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml
from enum import Enum

class FeatureType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    TEXT = "text"
    EMBEDDING = "embedding"

class FeatureTransformation(Enum):
    STANDARD_SCALER = "standard_scaler"
    MIN_MAX_SCALER = "min_max_scaler"
    ONE_HOT_ENCODING = "one_hot_encoding"
    LOG_TRANSFORM = "log_transform"
    BINNING = "binning"

class FeatureRegistry:
    """Центральный реестр признаков"""
    
    def __init__(self, config_path: str = "feature_registry.yaml"):
        self.config_path = config_path
        self.features = {}
        self.load_registry()
    
    def load_registry(self):
        """Загрузка реестра из YAML файла"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.features = config.get('features', {})
            print(f"Loaded {len(self.features)} features from registry")
        except FileNotFoundError:
            print("Registry file not found, starting with empty registry")
            self.features = {}
    
    def save_registry(self):
        """Сохранение реестра в YAML файл"""
        config = {
            'metadata': {
                'version': '1.0',
                'last_updated': datetime.utcnow().isoformat(),
                'feature_count': len(self.features)
            },
            'features': self.features
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Saved {len(self.features)} features to registry")
    
    def register_feature(self, feature_name: str, feature_type: FeatureType,
                        description: str = "", 
                        source: str = "",
                        transformations: List[FeatureTransformation] = None,
                        validation_rules: Dict[str, Any] = None,
                        owners: List[str] = None):
        """Регистрация нового признака"""
        
        feature_definition = {
            'name': feature_name,
            'type': feature_type.value,
            'description': description,
            'source': source,
            'created_at': datetime.utcnow().isoformat(),
            'transformations': [t.value for t in transformations] if transformations else [],
            'validation_rules': validation_rules or {},
            'owners': owners or [],
            'version': 1
        }
        
        self.features[feature_name] = feature_definition
        self.save_registry()
        
        print(f"Registered feature: {feature_name}")
        return feature_definition
    
    def get_feature(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Получение определения признака"""
        return self.features.get(feature_name)
    
    def get_features_by_type(self, feature_type: FeatureType) -> List[Dict[str, Any]]:
        """Получение признаков по типу"""
        return [
            feature for feature in self.features.values()
            if feature['type'] == feature_type.value
        ]
    
    def update_feature(self, feature_name: str, updates: Dict[str, Any]):
        """Обновление признака"""
        if feature_name in self.features:
            current = self.features[feature_name]
            current.update(updates)
            current['updated_at'] = datetime.utcnow().isoformat()
            current['version'] += 1
            self.features[feature_name] = current
            self.save_registry()
            print(f"Updated feature: {feature_name}")
            return True
        return False
    
    def delete_feature(self, feature_name: str):
        """Удаление признака"""
        if feature_name in self.features:
            del self.features[feature_name]
            self.save_registry()
            print(f"Deleted feature: {feature_name}")
            return True
        return False
    
    def validate_feature_value(self, feature_name: str, value: Any) -> Dict[str, Any]:
        """Валидация значения признака"""
        feature = self.get_feature(feature_name)
        if not feature:
            return {'valid': False, 'error': 'Feature not found'}
        
        validation_rules = feature.get('validation_rules', {})
        errors = []
        
        # Проверка типа
        if feature['type'] == FeatureType.NUMERIC.value:
            if not isinstance(value, (int, float)):
                errors.append(f"Expected numeric, got {type(value)}")
            
            # Проверка диапазона
            min_val = validation_rules.get('min')
            max_val = validation_rules.get('max')
            
            if min_val is not None and value < min_val:
                errors.append(f"Value {value} below minimum {min_val}")
            if max_val is not None and value > max_val:
                errors.append(f"Value {value} above maximum {max_val}")
        
        elif feature['type'] == FeatureType.CATEGORICAL.value:
            allowed_values = validation_rules.get('allowed_values', [])
            if allowed_values and value not in allowed_values:
                errors.append(f"Value {value} not in allowed values: {allowed_values}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'feature_name': feature_name
        }
    
    def export_to_sql(self, connection_string: str):
        """Экспорт реестра в базу данных"""
        from sqlalchemy import create_engine, text
        
        engine = create_engine(connection_string)
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS feature_registry_export (
            feature_name VARCHAR(100) PRIMARY KEY,
            feature_type VARCHAR(50),
            description TEXT,
            source VARCHAR(200),
            transformations JSONB,
            validation_rules JSONB,
            owners JSONB,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            version INTEGER
        )
        """
        
        try:
            with engine.connect() as conn:
                conn.execute(text(create_table_sql))
                
                for feature_name, feature_def in self.features.items():
                    insert_sql = """
                    INSERT INTO feature_registry_export 
                    (feature_name, feature_type, description, source, transformations, 
                     validation_rules, owners, created_at, updated_at, version)
                    VALUES (:feature_name, :feature_type, :description, :source, 
                            :transformations, :validation_rules, :owners, 
                            :created_at, :updated_at, :version)
                    ON CONFLICT (feature_name) DO UPDATE SET
                        feature_type = EXCLUDED.feature_type,
                        description = EXCLUDED.description,
                        source = EXCLUDED.source,
                        transformations = EXCLUDED.transformations,
                        validation_rules = EXCLUDED.validation_rules,
                        owners = EXCLUDED.owners,
                        updated_at = EXCLUDED.updated_at,
                        version = EXCLUDED.version
                    """
                    
                    conn.execute(text(insert_sql), {
                        'feature_name': feature_name,
                        'feature_type': feature_def['type'],
                        'description': feature_def.get('description', ''),
                        'source': feature_def.get('source', ''),
                        'transformations': json.dumps(feature_def.get('transformations', [])),
                        'validation_rules': json.dumps(feature_def.get('validation_rules', {})),
                        'owners': json.dumps(feature_def.get('owners', [])),
                        'created_at': feature_def.get('created_at'),
                        'updated_at': feature_def.get('updated_at', datetime.utcnow().isoformat()),
                        'version': feature_def.get('version', 1)
                    })
                
                conn.commit()
                print(f"Exported {len(self.features)} features to database")
                
        except Exception as e:
            print(f"Failed to export to database: {str(e)}")

# Пример конфигурационного файла feature_registry.yaml
YAML_CONFIG = """
metadata:
  version: "1.0"
  last_updated: "2024-01-15T10:30:00"
  feature_count: 10

features:
  temperature:
    name: "temperature"
    type: "numeric"
    description: "Температура оборудования"
    source: "sensor_data.temperature"
    created_at: "2024-01-15T10:30:00"
    transformations: ["standard_scaler"]
    validation_rules:
      min: 0
      max: 150
      unit: "Celsius"
    owners: ["data_engineer@company.com"]
    version: 1
  
  vibration:
    name: "vibration"
    type: "numeric"
    description: "Уровень вибрации"
    source: "sensor_data.vibration"
    created_at: "2024-01-15T10:30:00"
    transformations: ["log_transform"]
    validation_rules:
      min: 0
      max: 100
      unit: "mm/s"
    owners: ["data_engineer@company.com"]
    version: 1
  
  equipment_type:
    name: "equipment_type"
    type: "categorical"
    description: "Тип оборудования"
    source: "equipment_metadata.type"
    created_at: "2024-01-15T10:30:00"
    transformations: ["one_hot_encoding"]
    validation_rules:
      allowed_values: ["pump", "compressor", "turbine", "generator"]
    owners: ["data_engineer@company.com"]
    version: 1
"""