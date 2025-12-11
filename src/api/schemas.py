from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime

class SensorData(BaseModel):
    """Данные с датчиков"""
    sensor_id: str
    value: float
    unit: str
    reading_time: str

class MaintenanceHistory(BaseModel):
    """История обслуживания"""
    maintenance_id: str
    equipment_id: str
    maintenance_type: str
    date: str
    description: Optional[str]
    cost: Optional[float]

class EquipmentInfo(BaseModel):
    """Информация об оборудовании"""
    equipment_id: str
    manufacturer: str
    model: str
    installation_date: str
    last_maintenance_date: Optional[str]
    operational_hours: float
    location: str