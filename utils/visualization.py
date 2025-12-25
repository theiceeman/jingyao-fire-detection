"""
Visualization utilities for confusion matrices and results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    save_path: str = None,
    title: str = None
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix as 2x2 array [[TN, FP], [FN, TP]]
        model_name: Name of the model
        save_path: Path to save the figure (optional)
        title: Title for the plot (optional)
    """
    plt.figure(figsize=(8, 6))
    
    # Reorder to standard format: [[TN, FP], [FN, TP]]
    labels = ['Non-Fire', 'Fire']
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Confusion Matrix - {model_name}')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def create_comparison_table(results: dict, save_path: str = None) -> str:
    """
    Create a comparison table from experiment results.
    
    Args:
        results: Dictionary with results for each model and scenario
        save_path: Path to save the table as CSV (optional)
    
    Returns:
        Formatted table as string
    """
    import pandas as pd
    
    rows = []
    for model_name, scenarios in results.items():
        for scenario_name, metrics in scenarios.items():
            rows.append({
                'Model': model_name,
                'Scenario': scenario_name,
                'Fire Detection Rate': f"{metrics['fire_detection_rate']:.4f}",
                'Error Warning Rate': f"{metrics['error_warning_rate']:.4f}",
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'TP': metrics['tp'],
                'TN': metrics['tn'],
                'FP': metrics['fp'],
                'FN': metrics['fn']
            })
    
    df = pd.DataFrame(rows)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Comparison table saved to {save_path}")
    
    return df.to_string(index=False)

