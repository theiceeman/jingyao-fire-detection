"""
Custom metrics for fire detection evaluation.
"""
import numpy as np
from sklearn.metrics import confusion_matrix


def fire_detection_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Fire Detection Rate = TP / (TP + FN)
    
    Args:
        y_true: True labels (0 = non-fire, 1 = fire)
        y_pred: Predicted labels (0 = non-fire, 1 = fire)
    
    Returns:
        Fire Detection Rate
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    if tp + fn == 0:
        return 0.0
    
    return tp / (tp + fn)


def error_warning_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Error Warning Rate = FP / (FP + TN)
    
    Args:
        y_true: True labels (0 = non-fire, 1 = fire)
        y_pred: Predicted labels (0 = non-fire, 1 = fire)
    
    Returns:
        Error Warning Rate
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    if fp + tn == 0:
        return 0.0
    
    return fp / (fp + tn)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate all metrics for fire detection.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary with all metrics
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    fdr = fire_detection_rate(y_true, y_pred)
    ewr = error_warning_rate(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'confusion_matrix': np.array([[tn, fp], [fn, tp]]),
        'fire_detection_rate': fdr,
        'error_warning_rate': ewr,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }

