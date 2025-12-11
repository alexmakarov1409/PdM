import pandas as pd
from sqlalchemy import create_engine
import logging

logger = logging.getLogger(__name__)

def load_from_csv(file_path: str) -> pd.DataFrame:
    """Загрузка данных из CSV"""
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows from {file_path}")
    return df

def load_from_postgres(query: str, connection_string: str) -> pd.DataFrame:
    """Загрузка данных из PostgreSQL"""
    engine = create_engine(connection_string)
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df)} rows from PostgreSQL")
    return df

def save_to_postgres(df: pd.DataFrame, table_name: str, connection_string: str):
    """Сохранение DataFrame в PostgreSQL"""
    engine = create_engine(connection_string)
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    logger.info(f"Saved {len(df)} rows to {table_name}")