import pandas as pd
import json

def validate_data_quality():
    """Проверка качества данных"""
    
    validation_report = {
        "raw_data": validate_raw_data(),
        "processed_data": validate_processed_data(),
        "samples": validate_samples()
    }
    
    # Сохранение отчета
    with open('data/validation_report.json', 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    return validation_report

def validate_raw_data():
    """Проверка сырых данных"""
    df = pd.read_csv('data/raw/predictive_maintenance.csv')
    
    report = {
        "total_records": len(df),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "value_ranges": {
            "air_temperature": {"min": df['Air temperature [K]'].min(), "max": df['Air temperature [K]'].max()},
            "torque": {"min": df['Torque [Nm]'].min(), "max": df['Torque [Nm]'].max()}
        }
    }
    
    return report