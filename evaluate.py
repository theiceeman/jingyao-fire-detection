"""
Evaluation script for fire detection models.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import calculate_metrics
from utils.visualization import plot_confusion_matrix


def evaluate_pytorch_model(
    model: torch.nn.Module, test_loader: DataLoader, device: torch.device = None
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
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


def print_metrics(metrics: dict, test_set_name: str = ""):
    """
    Print formatted metrics for a test set.

    Args:
        metrics: Dictionary containing metrics
        test_set_name: Optional name of the test set for display
    """
    if test_set_name:
        print(f"\n{test_set_name} Results:")
    print(f"  Fire Detection Rate: {metrics['fire_detection_rate']:.4f}")
    print(f"  Error Warning Rate: {metrics['error_warning_rate']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")


def evaluate_pytorch_model_on_test_sets(
    model: torch.nn.Module,
    model_name: str,
    test_data_loader: DataLoader,
    device: torch.device,
    results_dict: dict,
    firelike_amount: int = None,
) -> tuple:
    """
    Evaluate a PyTorch model on both baseline and BCST test sets.
    Handles evaluation, plotting, printing, and storing results.

    Args:
        model: Trained PyTorch model
        model_name: Name of the model (e.g., "EfficientNet", "MobileNet")
        test_data_loader: DataLoader for test set
        device: Device to run evaluation on
        results_dict: Dictionary to store results in
        bcst_amount: Optional BCST amount used in training (for result storage)

    Returns:
        Tuple of (test_metrics)
    """

    print(f"\nEvaluating {model_name} on test set...")
    metrics_test = evaluate_pytorch_model(model, test_data_loader, device)

    # Store results by BCST amount if provided
    if str(firelike_amount) not in results_dict[model_name]:
        results_dict[model_name][str(firelike_amount)] = {}

    results_dict[model_name][str(firelike_amount)]["test"] = metrics_test

    save_suffix = f"_firelike{firelike_amount}" if firelike_amount is not None else ""
    plot_confusion_matrix(
        metrics_test["confusion_matrix"],
        model_name,
        save_path=f"results/{model_name.lower()}_baseline{save_suffix}_confusion_matrix.png",
        title=(f"{model_name} - Test Set (Firelike Amount {firelike_amount})"),
    )
    print_metrics(metrics_test)


def evaluate_svm_model_on_test_sets(
    svm_model,
    test_loader_baseline: DataLoader,
    test_loader_bcst: DataLoader,
    results_dict: dict,
) -> tuple:
    """
    Evaluate an SVM model on both baseline and BCST test sets.
    Handles evaluation, plotting, printing, and storing results.

    Args:
        svm_model: Trained SVM model
        test_loader_baseline: DataLoader for baseline test set
        test_loader_bcst: DataLoader for BCST test set
        results_dict: Dictionary to store results in

    Returns:
        Tuple of (baseline_metrics, bcst_metrics)
    """
    # Evaluate on baseline test set
    print("\nEvaluating SVM on baseline test set...")
    metrics_baseline = evaluate_svm_model(svm_model, test_loader_baseline)
    results_dict["SVM"]["Baseline"] = metrics_baseline

    plot_confusion_matrix(
        metrics_baseline["confusion_matrix"],
        "SVM",
        save_path="results/svm_baseline_confusion_matrix.png",
        title="SVM - Baseline Test Set",
    )
    print_metrics(metrics_baseline)

    # Evaluate on BCST test set
    print("\nEvaluating SVM on test set with 200 BCST...")
    metrics_bcst = evaluate_svm_model(svm_model, test_loader_bcst)
    results_dict["SVM"]["200_BCST"] = metrics_bcst

    plot_confusion_matrix(
        metrics_bcst["confusion_matrix"],
        "SVM",
        save_path="results/svm_200bcst_confusion_matrix.png",
        title="SVM - Test Set with 200 BCST",
    )
    print_metrics(metrics_bcst)

    return metrics_baseline, metrics_bcst
