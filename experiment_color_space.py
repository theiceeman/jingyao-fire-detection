# Q2 Experiment color space, for fire detection.
# Runs experiments with different color spaces .

import torch
import numpy as np
from pathlib import Path
import json

from models.efficientnet import create_efficientnet_model
from models.mobilenet import create_mobilenet_model
from models.svm_model import create_svm_model
from utils.data_loader import (
    prepare_dataset,
    load_image_paths,
    create_dataloaders,
    get_train_transforms,
    get_val_transforms,
)
from train import train_pytorch_model, train_svm_model
from evaluate import (
    evaluate_pytorch_model_for_color_space,
    evaluate_svm_model_for_color_space,
)
from sklearn.model_selection import train_test_split


def run_experiment_color_space(
    data_root: str = "data",
    num_epochs: int = 20,
    learning_rate: float = 0.001,
    device: torch.device = None,
    patience: int = None,
):
    """
    Run fire detection experiment with different color spaces.

    Args:
        data_root: Root directory containing data folders
        num_epochs: Number of training epochs
        learning_rate: Learning rate for PyTorch models
        device: Device to run on
        patience: Early stopping patience (None = no early stopping)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== SETUP ==========
    print("=" * 80)
    print("SETTING UP EXPERIMENT")
    print("=" * 80)

    train_dir = Path(data_root) / "train"
    test_dir = Path(data_root) / "test"
    firelike_dir = Path(data_root) / "bcst"

    # Prepare test sets.
    # These re used for baseline tests(before training) and for evaluation (after training has been completed).
    print("\nPreparing test sets...")
    test_fire_directories = load_image_paths(str(test_dir), "fire")
    test_nonfire_directories = load_image_paths(str(test_dir), "non_fire")

    # combine all fire and non fire directories
    test_directories = test_fire_directories + test_nonfire_directories
    # label fire directories as 1 and non fire directories as 0 for evaluation purposes
    test_labels = [1] * len(test_fire_directories) + [0] * len(test_nonfire_directories)

    print(f"  evaluation test set: {len(test_directories)} images")

    # ========== EXPERIMENT LOOP ==========
    color_spaces = ["HSV", 
                    # "LAB", "YCbCr"
                    ]
    firelike_amounts = [0, 50, 100, 150]
    results = {"EfficientNet": {}, "MobileNet": {}, "SVM": {}}

    for color_space in color_spaces:
        print("\n" + "=" * 80)
        print(f"COLOR SPACE: {color_space}")
        print("=" * 80)

        # Initialize color space entries in results dict
        if color_space not in results["EfficientNet"]:
            results["EfficientNet"][color_space] = {}
        if color_space not in results["MobileNet"]:
            results["MobileNet"][color_space] = {}
        if color_space not in results["SVM"]:
            results["SVM"][color_space] = {}

        # Create test data loader with current color space
        val_transform = get_val_transforms(color_space=color_space)
        test_data_loader = create_dataloaders(
            test_directories,
            test_labels,
            shuffle=False,
            transform=val_transform,
        )

        for firelike_amount in firelike_amounts:
            print("\n" + "=" * 80)
            print(f"EXPERIMENT: {color_space} - Training with {firelike_amount} Fire-like images")
            print("=" * 80)

            # prepares training dataset for each batch of fire-like images.
            # it handles adding of the fire-like image no, and removing the non-fire equivalent.
            (
                fire_paths,
                fire_labels,
                nonfire_paths,
                nonfire_labels,
                firelike_paths,
                firelike_labels,
            ) = prepare_dataset(
                str(train_dir),
                firelike_dir=str(firelike_dir),
                num_firelike=firelike_amount,
            )

            print(f"\nTraining data composition:")
            print(f"  Fire images: {len(fire_paths)}")
            print(f"  Forest images: {len(nonfire_paths)}")
            print(f"  Fire-like images: {len(firelike_paths)}")
            print(f"  Total: {len(fire_paths) + len(nonfire_paths) + len(firelike_paths)}")

            # Split each class 80/20
            fire_train, fire_val, fire_labels_train, fire_labels_val = train_test_split(
                fire_paths, fire_labels, test_size=0.2, random_state=42
            )
            nonfire_train, nonfire_val, nonfire_labels_train, nonfire_labels_val = (
                train_test_split(
                    nonfire_paths, nonfire_labels, test_size=0.2, random_state=42
                )
            )

            if firelike_paths:
                firelike_train, firelike_val, firelike_labels_train, firelike_labels_val = (
                    train_test_split(
                        firelike_paths, firelike_labels, test_size=0.2, random_state=42
                    )
                )
                train_paths = fire_train + nonfire_train + firelike_train
                train_labels = (
                    fire_labels_train + nonfire_labels_train + firelike_labels_train
                )
                val_paths = fire_val + nonfire_val + firelike_val
                val_labels = fire_labels_val + nonfire_labels_val + firelike_labels_val
            else:
                train_paths = fire_train + nonfire_train
                train_labels = fire_labels_train + nonfire_labels_train
                val_paths = fire_val + nonfire_val
                val_labels = fire_labels_val + nonfire_labels_val

            print(f"\nTrain/Val split:")
            print(f"  Training: {len(train_paths)} images")
            print(f"  Validation: {len(val_paths)} images")

            # Create data loaders with color space
            train_transform = get_train_transforms(color_space=color_space)
            val_transform_cs = get_val_transforms(color_space=color_space)
            train_loader = create_dataloaders(
                train_paths,
                train_labels,
                shuffle=True,
                transform=train_transform,
            )
            val_loader = create_dataloaders(
                val_paths,
                val_labels,
                shuffle=False,
                transform=val_transform_cs,
            )

            # Train EfficientNet
            print(f"\nTraining EfficientNet ({color_space}, FIRE-LIKE={firelike_amount})...")
            efficientnet = create_efficientnet_model(
                num_classes=2, model_name="efficientnet_b0"
            )
            efficientnet = train_pytorch_model(
                efficientnet,
                train_loader,
                val_loader,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                device=device,
                save_dir=f"checkpoints/efficientnet/{color_space}_firelike_{firelike_amount}",
                patience=patience,
            )

            # Evaluate on test set
            evaluate_pytorch_model_for_color_space(
                efficientnet,
                "EfficientNet",
                test_data_loader,
                device,
                results["EfficientNet"][color_space],
                firelike_amount,
                color_space,
            )

            # ========== MobileNet ==========
            print(f"\nTraining MobileNet ({color_space}, FIRE-LIKE={firelike_amount})...")
            mobilenet = create_mobilenet_model(num_classes=2, model_name="mobilenet_v2")
            mobilenet = train_pytorch_model(
                mobilenet,
                train_loader,
                val_loader,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                device=device,
                save_dir=f"checkpoints/mobilenet/{color_space}_firelike_{firelike_amount}",
                patience=patience,
            )

            # Evaluate on test set
            evaluate_pytorch_model_for_color_space(
                mobilenet,
                "MobileNet",
                test_data_loader,
                device,
                results["MobileNet"][color_space],
                firelike_amount,
                color_space,
            )

            # ========== SVM ==========
            print(f"\nTraining SVM ({color_space}, FIRE-LIKE={firelike_amount})...")
            svm_model = create_svm_model(kernel="rbf", C=1.0, gamma="scale")
            train_svm_model(svm_model, train_loader)

            # Evaluate on test set
            evaluate_svm_model_for_color_space(
                svm_model,
                test_data_loader,
                results["SVM"][color_space],
                firelike_amount,
                color_space,
            )

    # ========== SAVE RESULTS ==========
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    Path("results").mkdir(exist_ok=True)

    # Save detailed results as JSON
    results_json = {}
    for model_name, color_space_results in results.items():
        results_json[model_name] = {}
        for color_space, bcst_results in color_space_results.items():
            results_json[model_name][color_space] = {}
            for bcst_key, test_results in bcst_results.items():
                results_json[model_name][color_space][bcst_key] = {}
                for test_set_name, metrics in test_results.items():
                    results_json[model_name][color_space][bcst_key][test_set_name] = {
                        "fire_detection_rate": float(metrics["fire_detection_rate"]),
                        "error_warning_rate": float(metrics["error_warning_rate"]),
                        "accuracy": float(metrics["accuracy"]),
                        "precision": float(metrics["precision"]),
                        "recall": float(metrics["recall"]),
                        "f1_score": float(metrics["f1_score"]),
                        "tp": int(metrics["tp"]),
                        "tn": int(metrics["tn"]),
                        "fp": int(metrics["fp"]),
                        "fn": int(metrics["fn"]),
                    }

    with open("results/detailed_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print("\nResults saved to:")
    print("  - results/detailed_results.json")
    print("  - results/*_confusion_matrix.png")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run fire detection experiment")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory containing data folders",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--patience", type=int, default=None, help="Early stopping patience"
    )

    args = parser.parse_args()

    run_experiment_color_space(
        data_root=args.data_root,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
    )
