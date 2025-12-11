import random
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
from scipy import signal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorDataSimulator:
    """Симулятор данных с датчиков оборудования"""
    
    def __init__(self):
        # Базовые параметры для разных типов оборудования
        self.equipment_profiles = {
            'pump': {
                'temperature_range': (60, 85),
                'vibration_range': (20, 45),
                'pressure_range': (100, 130),
                'rotation_speed_range': (2000, 3000),
                'failure_probability': 0.01
            },
            'compressor': {
                'temperature_range': (70, 95),
                'vibration_range': (30, 60),
                'pressure_range': (120, 160),
                'rotation_speed_range': (2500, 3500),
                'failure_probability': 0.02
            },
            'turbine': {
                'temperature_range': (80, 110),
                'vibration_range': (40, 70),
                'pressure_range': (90, 120),
                'rotation_speed_range': (3000, 4000),
                'failure_probability': 0.015
            }
        }
        
        # Состояния оборудования (для симуляции износа)
        self.equipment_states = {}
        
        # Шум и тренды
        self.noise_level = 0.05  # 5% шума
        self.trend_duration = 3600  # 1 час в секундах
        
        # Синусоидальные компоненты для реалистичности
        self.sine_frequencies = {
            'temperature': 0.001,  # низкая частота
            'vibration': 0.01,     # средняя частота
            'pressure': 0.005      # низкая-средняя частота
        }
    
    def generate_sensor_data(self, equipment_id: str, 
                           equipment_type: str = None) -> Dict[str, Any]:
        """Генерация данных с датчиков для оборудования"""
        try:
            # Определяем тип оборудования
            if not equipment_type:
                # Берем из ID или случайно
                if equipment_id.startswith('PUMP'):
                    equipment_type = 'pump'
                elif equipment_id.startswith('COMP'):
                    equipment_type = 'compressor'
                elif equipment_id.startswith('TURB'):
                    equipment_type = 'turbine'
                else:
                    equipment_type = random.choice(list(self.equipment_profiles.keys()))
            
            # Получаем профиль оборудования
            profile = self.equipment_profiles.get(
                equipment_type, 
                self.equipment_profiles['pump']
            )
            
            # Инициализация состояния оборудования если его нет
            if equipment_id not in self.equipment_states:
                self.equipment_states[equipment_id] = {
                    'start_time': time.time(),
                    'wear_level': 0.0,  # 0-1 уровень износа
                    'last_failure': None,
                    'operating_hours': 0,
                    'profile': profile,
                    'sine_phase': {k: random.random() * 2 * np.pi 
                                 for k in self.sine_frequencies.keys()}
                }
            
            state = self.equipment_states[equipment_id]
            
            # Обновление состояния
            current_time = time.time()
            elapsed = current_time - state['start_time']
            state['operating_hours'] = elapsed / 3600
            
            # Увеличение уровня износа (линейно со временем)
            state['wear_level'] = min(1.0, elapsed / (self.trend_duration * 10))
            
            # Генерация данных с датчиков
            sensor_data = {}
            
            # Температура
            temp_range = profile['temperature_range']
            base_temp = (temp_range[0] + temp_range[1]) / 2
            temp_trend = state['wear_level'] * 20  # нагрев при износе
            temp_noise = random.uniform(-5, 5)
            temp_sine = 5 * np.sin(
                self.sine_frequencies['temperature'] * elapsed + state['sine_phase']['temperature']
            )
            
            sensor_data['temperature'] = round(
                base_temp + temp_trend + temp_noise + temp_sine, 2
            )
            
            # Вибрация
            vib_range = profile['vibration_range']
            base_vib = (vib_range[0] + vib_range[1]) / 2
            vib_trend = state['wear_level'] * 15  # увеличение вибрации при износе
            vib_noise = random.uniform(-3, 3)
            vib_sine = 3 * np.sin(
                self.sine_frequencies['vibration'] * elapsed + state['sine_phase']['vibration']
            )
            
            sensor_data['vibration'] = round(
                base_vib + vib_trend + vib_noise + vib_sine, 2
            )
            
            # Давление
            press_range = profile['pressure_range']
            base_press = (press_range[0] + press_range[1]) / 2
            press_trend = state['wear_level'] * 10  # изменение давления при износе
            press_noise = random.uniform(-2, 2)
            press_sine = 2 * np.sin(
                self.sine_frequencies['pressure'] * elapsed + state['sine_phase']['pressure']
            )
            
            sensor_data['pressure'] = round(
                base_press + press_trend + press_noise + press_sine, 2
            )
            
            # Скорость вращения
            speed_range = profile['rotation_speed_range']
            base_speed = random.uniform(speed_range[0], speed_range[1])
            speed_noise = random.uniform(-50, 50)
            
            sensor_data['rotation_speed'] = round(base_speed + speed_noise, 1)
            
            # Потребление энергии
            power_base = 1000 + sensor_data['rotation_speed'] * 0.1
            power_noise = random.uniform(-20, 20)
            
            sensor_data['power_consumption'] = round(power_base + power_noise, 1)
            
            # Дополнительные метрики
            sensor_data['efficiency'] = round(
                sensor_data['rotation_speed'] / (sensor_data['power_consumption'] + 1e-10), 4
            )
            
            sensor_data['temperature_vibration_ratio'] = round(
                sensor_data['temperature'] / (sensor_data['vibration'] + 1e-10), 4
            )
            
            # Симуляция отказа
            if random.random() < profile['failure_probability']:
                sensor_data = self._simulate_failure(sensor_data, equipment_type)
                state['last_failure'] = current_time
                logger.warning(f"Failure simulated for {equipment_id}")
            
            # Добавляем метаданные
            sensor_data['equipment_type'] = equipment_type
            sensor_data['wear_level'] = round(state['wear_level'], 4)
            sensor_data['operating_hours'] = round(state['operating_hours'], 2)
            
            return sensor_data
            
        except Exception as e:
            logger.error(f"Failed to generate sensor data: {str(e)}")
            return self._generate_default_data()
    
    def _simulate_failure(self, sensor_data: Dict[str, Any], 
                         equipment_type: str) -> Dict[str, Any]:
        """Симуляция отказа оборудования"""
        failure_type = random.choice(['gradual', 'sudden', 'intermittent'])
        
        if failure_type == 'gradual':
            # Постепенное ухудшение
            sensor_data['temperature'] *= 1.3
            sensor_data['vibration'] *= 1.5
            sensor_data['pressure'] *= 0.8
        
        elif failure_type == 'sudden':
            # Внезапный отказ
            sensor_data['temperature'] = random.uniform(120, 150)
            sensor_data['vibration'] = random.uniform(80, 100)
            sensor_data['pressure'] = random.uniform(180, 200)
        
        elif failure_type == 'intermittent':
            # Прерывистый отказ
            if random.random() > 0.5:
                sensor_data['temperature'] *= 1.2
                sensor_data['vibration'] *= 1.3
        
        # Добавляем флаг отказа
        sensor_data['failure_indicator'] = 1
        sensor_data['failure_type'] = failure_type
        
        return sensor_data
    
    def _generate_default_data(self) -> Dict[str, Any]:
        """Генерация данных по умолчанию при ошибке"""
        return {
            'temperature': round(random.uniform(70, 90), 2),
            'vibration': round(random.uniform(30, 50), 2),
            'pressure': round(random.uniform(100, 140), 2),
            'rotation_speed': round(random.uniform(2000, 3000), 1),
            'power_consumption': round(random.uniform(1000, 1500), 1),
            'efficiency': round(random.uniform(1.5, 2.5), 4),
            'equipment_type': 'unknown',
            'wear_level': 0.0,
            'operating_hours': 0.0
        }
    
    def generate_batch_data(self, equipment_ids: List[str], 
                          count: int = 100) -> List[Dict[str, Any]]:
        """Генерация пакета данных"""
        batch = []
        
        for _ in range(count):
            eq_id = random.choice(equipment_ids)
            data = self.generate_sensor_data(eq_id)
            batch.append({
                'equipment_id': eq_id,
                'timestamp': datetime.utcnow().isoformat(),
                'data': data
            })
        
        return batch
    
    def simulate_real_time_stream(self, equipment_ids: List[str],
                                interval_seconds: float = 1.0,
                                duration_minutes: float = 5.0):
        """Симуляция реального потока данных"""
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        message_count = 0
        
        logger.info(f"Starting real-time simulation for {len(equipment_ids)} equipment")
        logger.info(f"Interval: {interval_seconds}s, Duration: {duration_minutes}min")
        
        try:
            while time.time() < end_time:
                for eq_id in equipment_ids:
                    data = self.generate_sensor_data(eq_id)
                    
                    # Создание сообщения
                    message = {
                        'equipment_id': eq_id,
                        'timestamp': datetime.utcnow().isoformat(),
                        'data': data
                    }
                    
                    yield message
                    message_count += 1
                
                # Пауза между отправками
                time.sleep(interval_seconds)
                
                # Логирование прогресса
                if message_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = message_count / elapsed
                    logger.info(f"Generated {message_count} messages, rate: {rate:.1f} msg/sec")
        
        except KeyboardInterrupt:
            logger.info("Simulation stopped by user")
        
        finally:
            total_time = time.time() - start_time
            logger.info(
                f"Simulation complete. Total: {message_count} messages, "
                f"Duration: {total_time:.1f}s, Rate: {message_count/total_time:.1f} msg/sec"
            )

# Пример использования
if __name__ == "__main__":
    simulator = SensorDataSimulator()
    
    # Генерация данных для одного оборудования
    data = simulator.generate_sensor_data('EQ-001', 'pump')
    print("Generated sensor data:")
    for key, value in data.items():
        print(f"  {key}: {value}")
    
    # Симуляция потока данных
    equipment_ids = [f'EQ-{i:03d}' for i in range(1, 6)]
    
    print("\nSimulating real-time stream...")
    for i, message in enumerate(simulator.simulate_real_time_stream(
        equipment_ids, interval_seconds=0.1, duration_minutes=0.1
    )):
        if i >= 10:  # Ограничиваем вывод
            break
        print(f"Message {i+1}: {message['equipment_id']} - {message['data']['temperature']}°C")