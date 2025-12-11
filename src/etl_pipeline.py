import pandas as pd
from src.data.loader import load_from_csv, save_to_postgres
from src.data.cleaner import clean_data, normalize_features
from src.data.feature_engineering import create_time_features, create_rolling_features
from src.utils.config import config
from src.utils.logger import logger

class ETLPipeline:
    def __init__(self):
        self.raw_df = None
        self.cleaned_df = None
        self.features_df = None
    
    def extract(self, source_type: str = 'csv', **kwargs):
        """Этап Extract"""
        logger.info("Starting extraction phase")
        
        if source_type == 'csv':
            self.raw_df = load_from_csv(config.DATA_PATH)
        elif source_type == 'postgres':
            query = kwargs.get('query', 'SELECT * FROM raw_data')
            self.raw_df = load_from_postgres(query, config.DB_CONNECTION_STRING)
        
        logger.info(f"Extracted {len(self.raw_df)} rows")
        return self
    
    def transform(self):
        """Этап Transform"""
        logger.info("Starting transformation phase")
        
        # Очистка данных
        self.cleaned_df = clean_data(self.raw_df)
        
        # Создание временных признаков
        self.cleaned_df = create_time_features(self.cleaned_df, 'timestamp')
        
        # Создание скользящих статистик
        numeric_cols = self.cleaned_df.select_dtypes(include=[np.number]).columns
        self.features_df = create_rolling_features(self.cleaned_df[numeric_cols])
        
        # Нормализация
        feature_cols = [col for col in self.features_df.columns if col != config.TARGET_COLUMN]
        self.features_df, scaler = normalize_features(self.features_df, feature_cols)
        
        logger.info(f"Transformed data shape: {self.features_df.shape}")
        return self
    
    def load(self):
        """Этап Load"""
        logger.info("Starting load phase")
        
        # Сохранение в PostgreSQL слоями
        save_to_postgres(self.raw_df, 'raw_data', config.DB_CONNECTION_STRING)
        save_to_postgres(self.cleaned_df, 'cleaned_data', config.DB_CONNECTION_STRING)
        save_to_postgres(self.features_df, 'features', config.DB_CONNECTION_STRING)
        
        logger.info("Data loaded to PostgreSQL")
        return self
    
    def run(self):
        """Запуск полного ETL пайплайна"""
        try:
            (self.extract('csv')
             .transform()
             .load())
            logger.info("ETL pipeline completed successfully")
        except Exception as e:
            logger.error(f"ETL pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    pipeline = ETLPipeline()
    pipeline.run()