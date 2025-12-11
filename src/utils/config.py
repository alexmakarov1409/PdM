import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Пути к данным
    DATA_PATH = 'data/raw/predictive_maintenance.csv'
    MODEL_PATH = 'models/final_model.pkl'
    
    # База данных
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'predictive_maintenance')
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'password')
    
    # Параметры модели
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    TARGET_COLUMN = 'failure_target'
    
    # Пороги
    RECALL_THRESHOLD = 0.85
    FPR_THRESHOLD = 0.15
    
    @property
    def DB_CONNECTION_STRING(self):
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

config = Config()