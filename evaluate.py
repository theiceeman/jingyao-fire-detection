"""
Evaluation script for fire detection models.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import calculate_metrics


def evaluate_pytorch_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device = None
) -> dict:
    """
    Evaluate a PyTorch model on test data.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to run evaluation on
    
    Returns:
        Dictionary with metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    metrics = calculate_metrics(y_true, y_pred)
    return metrics


def evaluate_svm_model(svm_model, test_loader: DataLoader) -> dict:
    """
    Evaluate SVM model on test data.
    
    Args:
        svm_model: Trained SVM model
        test_loader: Test data loader
    
    Returns:
        Dictionary with metrics
    """
    y_pred, y_true = svm_model.predict(test_loader)
    metrics = calculate_metrics(y_true, y_pred)
    return metrics

