"""
Main experiment orchestrator for fire detection.
Runs baseline and BCST experiments for all three models.
"""
import torch
import numpy as np
from pathlib import Path
import json

from models.efficientnet import create_efficientnet_model
from models.mobilenet import create_mobilenet_model
from models.svm_model import create_svm_model
from utils.data_loader import (
    prepare_baseline_dataset,
    prepare_test_dataset,
    create_dataloaders,
    get_train_transforms,
    get_val_transforms
)
from utils.metrics import calculate_metrics
from utils.visualization import plot_confusion_matrix, create_comparison_table
from train import train_pytorch_model, train_svm_model
from evaluate import evaluate_pytorch_model, evaluate_svm_model


def run_experiment(
    data_root: str = 'data',
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: torch.device = None
):
    """
    Run the complete fire detection experiment.
    
    Args:
        data_root: Root directory containing data folders
        num_epochs: Number of training epochs for PyTorch models
        batch_size: Batch size for training
        learning_rate: Learning rate for PyTorch models
        device: Device to run on
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("FOREST FIRE DETECTION EXPERIMENT")
    print("=" * 80)
    print(f"Device: {device}")
    print()
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dir = Path(data_root) / 'train'
    test_dir = Path(data_root) / 'test'
    bcst_dir = Path(data_root) / 'bcst'
    
    # Baseline training dataset: 600 fire + 600 non-fire
    print("Loading baseline training dataset (600 fire + 600 non-fire)...")
    train_paths, train_labels = prepare_baseline_dataset(
        str(train_dir),
        num_fire=600,
        num_non_fire=600
    )
    print(f"  Loaded {len(train_paths)} training images")
    
    # Split training data for validation (80/20 split)
    from sklearn.model_selection import train_test_split
    train_paths_split, val_paths_split, train_labels_split, val_labels_split = train_test_split(
        train_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    # Create data loaders
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    train_loader = create_dataloaders(
        train_paths_split, train_labels_split,
        batch_size=batch_size, shuffle=True, transform=train_transform
    )
    val_loader = create_dataloaders(
        val_paths_split, val_labels_split,
        batch_size=batch_size, shuffle=False, transform=val_transform
    )
    
    # Prepare test datasets: baseline and 200 BCST
    print("Loading baseline test dataset...")
    test_paths_baseline, test_labels_baseline = prepare_test_dataset(
        str(test_dir),
        use_bcst=False
    )
    print(f"  Loaded {len(test_paths_baseline)} baseline test images")
    
    print("Loading test dataset (with 200 BCST images)...")
    test_paths_bcst, test_labels_bcst = prepare_test_dataset(
        str(test_dir),
        bcst_dir=str(bcst_dir) if bcst_dir.exists() else None,
        num_bcst=200,
        num_forest_to_remove=200,
        use_bcst=True
    )
    print(f"  Loaded {len(test_paths_bcst)} test images with BCST")
    
    test_loader_baseline = create_dataloaders(
        test_paths_baseline, test_labels_baseline,
        batch_size=batch_size, shuffle=False, transform=val_transform
    )
    test_loader_bcst = create_dataloaders(
        test_paths_bcst, test_labels_bcst,
        batch_size=batch_size, shuffle=False, transform=val_transform
    )
    
    # Results storage
    results = {
        'EfficientNet': {},
        'MobileNet': {},
        'SVM': {}
    }
    
    # ========== EfficientNet ==========
    print("\n" + "=" * 80)
    print("MODEL 1: EfficientNet")
    print("=" * 80)
    
    # Train EfficientNet
    print("\nTraining EfficientNet...")
    efficientnet = create_efficientnet_model(num_classes=2, model_name='efficientnet_b0')
    efficientnet = train_pytorch_model(
        efficientnet, train_loader, val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        save_dir='checkpoints/efficientnet'
    )
    
    # Evaluate on baseline test set
    print("\nEvaluating EfficientNet on baseline test set...")
    efficientnet_metrics_baseline = evaluate_pytorch_model(efficientnet, test_loader_baseline, device)
    results['EfficientNet']['Baseline'] = efficientnet_metrics_baseline
    
    plot_confusion_matrix(
        efficientnet_metrics_baseline['confusion_matrix'],
        'EfficientNet',
        save_path='results/efficientnet_baseline_confusion_matrix.png',
        title='EfficientNet - Baseline Test Set'
    )
    
    print(f"  Fire Detection Rate: {efficientnet_metrics_baseline['fire_detection_rate']:.4f}")
    print(f"  Error Warning Rate: {efficientnet_metrics_baseline['error_warning_rate']:.4f}")
    print(f"  Accuracy: {efficientnet_metrics_baseline['accuracy']:.4f}")
    
    # Evaluate on 200 BCST test set
    print("\nEvaluating EfficientNet on test set with 200 BCST...")
    efficientnet_metrics_bcst = evaluate_pytorch_model(efficientnet, test_loader_bcst, device)
    results['EfficientNet']['200_BCST'] = efficientnet_metrics_bcst
    
    plot_confusion_matrix(
        efficientnet_metrics_bcst['confusion_matrix'],
        'EfficientNet',
        save_path='results/efficientnet_200bcst_confusion_matrix.png',
        title='EfficientNet - Test Set with 200 BCST'
    )
    
    print(f"  Fire Detection Rate: {efficientnet_metrics_bcst['fire_detection_rate']:.4f}")
    print(f"  Error Warning Rate: {efficientnet_metrics_bcst['error_warning_rate']:.4f}")
    print(f"  Accuracy: {efficientnet_metrics_bcst['accuracy']:.4f}")
    
    # ========== MobileNet ==========
    print("\n" + "=" * 80)
    print("MODEL 2: MobileNet")
    print("=" * 80)
    
    # Train MobileNet
    print("\nTraining MobileNet...")
    mobilenet = create_mobilenet_model(num_classes=2, model_name='mobilenet_v2')
    mobilenet = train_pytorch_model(
        mobilenet, train_loader, val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        save_dir='checkpoints/mobilenet'
    )
    
    # Evaluate on baseline test set
    print("\nEvaluating MobileNet on baseline test set...")
    mobilenet_metrics_baseline = evaluate_pytorch_model(mobilenet, test_loader_baseline, device)
    results['MobileNet']['Baseline'] = mobilenet_metrics_baseline
    
    plot_confusion_matrix(
        mobilenet_metrics_baseline['confusion_matrix'],
        'MobileNet',
        save_path='results/mobilenet_baseline_confusion_matrix.png',
        title='MobileNet - Baseline Test Set'
    )
    
    print(f"  Fire Detection Rate: {mobilenet_metrics_baseline['fire_detection_rate']:.4f}")
    print(f"  Error Warning Rate: {mobilenet_metrics_baseline['error_warning_rate']:.4f}")
    print(f"  Accuracy: {mobilenet_metrics_baseline['accuracy']:.4f}")
    
    # Evaluate on 200 BCST test set
    print("\nEvaluating MobileNet on test set with 200 BCST...")
    mobilenet_metrics_bcst = evaluate_pytorch_model(mobilenet, test_loader_bcst, device)
    results['MobileNet']['200_BCST'] = mobilenet_metrics_bcst
    
    plot_confusion_matrix(
        mobilenet_metrics_bcst['confusion_matrix'],
        'MobileNet',
        save_path='results/mobilenet_200bcst_confusion_matrix.png',
        title='MobileNet - Test Set with 200 BCST'
    )
    
    print(f"  Fire Detection Rate: {mobilenet_metrics_bcst['fire_detection_rate']:.4f}")
    print(f"  Error Warning Rate: {mobilenet_metrics_bcst['error_warning_rate']:.4f}")
    print(f"  Accuracy: {mobilenet_metrics_bcst['accuracy']:.4f}")
    
    # ========== SVM ==========
    print("\n" + "=" * 80)
    print("MODEL 3: SVM")
    print("=" * 80)
    
    # Train SVM
    print("\nTraining SVM...")
    svm_model = create_svm_model(kernel='rbf', C=1.0, gamma='scale')
    train_svm_model(svm_model, train_loader)
    
    # Evaluate on baseline test set
    print("\nEvaluating SVM on baseline test set...")
    svm_metrics_baseline = evaluate_svm_model(svm_model, test_loader_baseline)
    results['SVM']['Baseline'] = svm_metrics_baseline
    
    plot_confusion_matrix(
        svm_metrics_baseline['confusion_matrix'],
        'SVM',
        save_path='results/svm_baseline_confusion_matrix.png',
        title='SVM - Baseline Test Set'
    )
    
    print(f"  Fire Detection Rate: {svm_metrics_baseline['fire_detection_rate']:.4f}")
    print(f"  Error Warning Rate: {svm_metrics_baseline['error_warning_rate']:.4f}")
    print(f"  Accuracy: {svm_metrics_baseline['accuracy']:.4f}")
    
    # Evaluate on 200 BCST test set
    print("\nEvaluating SVM on test set with 200 BCST...")
    svm_metrics_bcst = evaluate_svm_model(svm_model, test_loader_bcst)
    results['SVM']['200_BCST'] = svm_metrics_bcst
    
    plot_confusion_matrix(
        svm_metrics_bcst['confusion_matrix'],
        'SVM',
        save_path='results/svm_200bcst_confusion_matrix.png',
        title='SVM - Test Set with 200 BCST'
    )
    
    print(f"  Fire Detection Rate: {svm_metrics_bcst['fire_detection_rate']:.4f}")
    print(f"  Error Warning Rate: {svm_metrics_bcst['error_warning_rate']:.4f}")
    print(f"  Accuracy: {svm_metrics_bcst['accuracy']:.4f}")
    
    # ========== Generate Results ==========
    print("\n" + "=" * 80)
    print("GENERATING RESULTS")
    print("=" * 80)
    
    # Create output directory
    Path('results').mkdir(exist_ok=True)
    
    # Create comparison table
    print("\nComparison Table:")
    table_str = create_comparison_table(results, save_path='results/comparison_table.csv')
    print(table_str)
    
    # Save detailed results as JSON
    results_json = {}
    for model_name, scenarios in results.items():
        results_json[model_name] = {}
        for scenario_name, metrics in scenarios.items():
            results_json[model_name][scenario_name] = {
                'fire_detection_rate': float(metrics['fire_detection_rate']),
                'error_warning_rate': float(metrics['error_warning_rate']),
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score']),
                'tp': int(metrics['tp']),
                'tn': int(metrics['tn']),
                'fp': int(metrics['fp']),
                'fn': int(metrics['fn'])
            }
    
    with open('results/detailed_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print("\nResults saved to:")
    print("  - results/comparison_table.csv")
    print("  - results/detailed_results.json")
    print("  - results/*_confusion_matrix.png")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED!")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run fire detection experiment')
    parser.add_argument('--data-root', type=str, default='data',
                       help='Root directory containing data folders')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    run_experiment(
        data_root=args.data_root,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

