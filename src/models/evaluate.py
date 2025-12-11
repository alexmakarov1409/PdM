from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def evaluate_model(y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series):
    """Полная оценка модели"""
    # Основные метрики
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC-AUC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    
    metrics = {
        'accuracy': report['accuracy'],
        'recall': report['1']['recall'],
        'precision': report['1']['precision'],
        'f1_score': report['1']['f1-score'],
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }
    
    return metrics

def plot_confusion_matrix(cm, model_name: str):
    """Визуализация confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.show()