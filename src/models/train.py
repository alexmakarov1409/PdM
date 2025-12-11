import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix

def train_baseline(X_train: pd.DataFrame, y_train: pd.Series):
    """Baseline модель (Logistic Regression или Decision Tree)"""
    from sklearn.linear_model import LogisticRegression
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 100):
    """Random Forest с гиперпараметрами"""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series):
    """XGBoost модель"""
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, path: str):
    """Сохранение обученной модели"""
    joblib.dump(model, path)
    print(f"Model saved to {path}")