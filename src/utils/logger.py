import logging
import sys
from datetime import datetime

def setup_logger(name: str, log_file: str = None):
    """Настройка логгера"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Форматтер
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Консольный хендлер
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Файловый хендлер (если указан файл)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Глобальный логгер
logger = setup_logger('predictive_maintenance', 'logs/app.log')