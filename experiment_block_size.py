# Q2 Experiment 1: Block size experiment for fire detection.
# Trains models on small image blocks (8x8, 16x16, 32x32) and tests on a single test image.

import torch
import numpy as np
from pathlib import Path
import json
from PIL import Image

from models.efficientnet import create_efficientnet_model
from models.mobilenet import create_mobilenet_model
from models.svm_model import create_svm_model
from utils.data_loader import (
    prepare_dataset,
    load_image_paths,
    create_dataloaders,
    get_train_transforms,
    get_val_transforms,
    FireDataset,
)
from utils.block_utils import (
    extract_blocks_from_dataset,
    extract_blocks_from_image,
    predict_image_blocks,
    visualize_block_predictions,
)
from train import train_pytorch_model, train_svm_model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def run_experiment_block_size(
    data_root: str = "data",
    test_image_path: str = "data/test/non_fire/block-test.png",
    num_epochs: int = 20,
    learning_rate: float = 0.001,
    device: torch.device = None,
    patience: int = None,
    max_training_blocks: int = 600,
):
    """
    Run block size experiment: train on small blocks, test on single image.

    Args:
        data_root: Root directory containing data folders
        test_image_path: Path to single test image
        num_epochs: Number of training epochs
        learning_rate: Learning rate for PyTorch models
        device: Device to run on
        patience: Early stopping patience
        max_training_blocks: Maximum blocks per class for training (600 total = 300 fire + 300 non-fire)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== SETUP ==========
    print("=" * 80)
    print("SETTING UP BLOCK SIZE EXPERIMENT")
    print("=" * 80)

    train_dir = Path(data_root) / "train"
    firelike_dir = Path(data_root) / "bcst"

    # Verify test image exists
    if not Path(test_image_path).exists():
        raise FileNotFoundError(f"Test image not found: {test_image_path}")

    # ========== EXPERIMENT LOOP ==========
    block_sizes = [8, 16, 32]
    bcst_amounts = [0, 50, 100, 150]  # Match paper
    results = {"EfficientNet": {}, "MobileNet": {}, "SVM": {}}
    trained_models = {}  # Store trained models: {block_size: {bcst: {model_name: model}}}

    for block_size in block_sizes:
        print("\n" + "=" * 80)
        print(f"BLOCK SIZE: {block_size}x{block_size}")
        print("=" * 80)

        trained_models[block_size] = {}

        for bcst_amount in bcst_amounts:
            print("\n" + "=" * 80)
            print(f"EXPERIMENT: Block Size {block_size} - Training with {bcst_amount} Fire-like images")
            print("=" * 80)

            # Prepare dataset (full images)
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
                num_firelike=bcst_amount,
            )

            print(f"\nFull image dataset:")
            print(f"  Fire images: {len(fire_paths)}")
            print(f"  Forest images: {len(nonfire_paths)}")
            print(f"  Fire-like images: {len(firelike_paths)}")

            # Combine all image paths and labels
            all_paths = fire_paths + nonfire_paths + firelike_paths
            all_labels = fire_labels + nonfire_labels + firelike_labels

            # Extract blocks from all images
            print(f"\nExtracting {block_size}x{block_size} blocks...")
            block_images, block_labels = extract_blocks_from_dataset(
                all_paths,
                all_labels,
                block_size=block_size,
                max_samples_per_class=max_training_blocks // 2,  # 300 per class
            )

            print(f"  Total blocks extracted: {len(block_images)}")
            print(f"  Fire blocks: {sum(block_labels)}")
            print(f"  Non-fire blocks: {len(block_labels) - sum(block_labels)}")

            # Split blocks into train/val (80/20)
            block_train, block_val, label_train, label_val = train_test_split(
                block_images,
                block_labels,
                test_size=0.2,
                random_state=42,
                stratify=block_labels,
            )

            print(f"\nBlock train/val split:")
            print(f"  Training blocks: {len(block_train)}")
            print(f"  Validation blocks: {len(block_val)}")

            # Create transforms (resize blocks to 224x224 for model input)
            train_transform = get_train_transforms()
            val_transform = get_val_transforms()

            # Create datasets and loaders
            train_dataset = FireDataset(block_train, label_train, transform=train_transform)
            val_dataset = FireDataset(block_val, label_val, transform=val_transform)

            train_loader = DataLoader(
                train_dataset,
                batch_size=32,
                shuffle=True,
                num_workers=0,  # Blocks are already in memory, no need for workers
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            trained_models[block_size][bcst_amount] = {}

            # ========== Train EfficientNet ==========
            print(f"\nTraining EfficientNet (Block Size {block_size}, BCST {bcst_amount})...")
            efficientnet = create_efficientnet_model(num_classes=2, model_name="efficientnet_b0")
            efficientnet = train_pytorch_model(
                efficientnet,
                train_loader,
                val_loader,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                device=device,
                save_dir=f"checkpoints/block_size/efficientnet/{block_size}_bcst_{bcst_amount}",
                patience=patience,
            )
            trained_models[block_size][bcst_amount]["EfficientNet"] = efficientnet

            # ========== Train MobileNet ==========
            print(f"\nTraining MobileNet (Block Size {block_size}, BCST {bcst_amount})...")
            mobilenet = create_mobilenet_model(num_classes=2, model_name="mobilenet_v2")
            mobilenet = train_pytorch_model(
                mobilenet,
                train_loader,
                val_loader,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                device=device,
                save_dir=f"checkpoints/block_size/mobilenet/{block_size}_bcst_{bcst_amount}",
                patience=patience,
            )
            trained_models[block_size][bcst_amount]["MobileNet"] = mobilenet

            # ========== Train SVM ==========
            print(f"\nTraining SVM (Block Size {block_size}, BCST {bcst_amount})...")
            svm_model = create_svm_model(kernel="rbf", C=1.0, gamma="scale")
            train_svm_model(svm_model, train_loader)
            trained_models[block_size][bcst_amount]["SVM"] = svm_model

    # ========== TESTING & VISUALIZATION ==========
    print("\n" + "=" * 80)
    print("TESTING ON SINGLE TEST IMAGE")
    print("=" * 80)

    val_transform = get_val_transforms()

    for block_size in block_sizes:
        print(f"\n{'=' * 80}")
        print(f"BLOCK SIZE: {block_size}x{block_size}")
        print("=" * 80)

        for bcst_amount in bcst_amounts:
            print(f"\nTesting with BCST {bcst_amount}...")

            for model_name in ["EfficientNet", "MobileNet", "SVM"]:
                model = trained_models[block_size][bcst_amount][model_name]

                # Predict blocks in test image
                if model_name == "SVM":
                    # SVM uses its own predict method with dataloader
                    test_image = Image.open(test_image_path).convert("RGB")
                    blocks, positions = extract_blocks_from_image(test_image, block_size)

                    max_row = max(pos[0] for pos in positions) + 1
                    max_col = max(pos[1] for pos in positions) + 1
                    matrix = np.ones((max_row, max_col), dtype=int)

                    # Create dataloader for blocks
                    block_dataset = FireDataset(blocks, [0] * len(blocks), transform=val_transform)
                    block_loader = DataLoader(block_dataset, batch_size=32, shuffle=False, num_workers=0)

                    # Use SVM's predict method
                    predictions, _ = model.predict(block_loader)

                    # Fill matrix (SVM returns 0=non-fire, 1=fire, but we want 0=fire, 1=non-fire)
                    for (row, col), pred in zip(positions, predictions):
                        matrix[row, col] = 1 - pred  # Invert: 0=fire, 1=non-fire

                    original_image = test_image

                else:
                    # PyTorch models
                    matrix, original_image = predict_image_blocks(
                        model, test_image_path, block_size, device, val_transform
                    )

                # Visualize results
                save_path = f"results/block_size/{block_size}/{model_name}/bcst_{bcst_amount}_visualization.png"
                visualize_block_predictions(original_image, matrix, block_size, save_path)

                # Store results
                if block_size not in results[model_name]:
                    results[model_name][block_size] = {}
                if bcst_amount not in results[model_name][block_size]:
                    results[model_name][block_size][bcst_amount] = {}

                # Calculate metrics from matrix (if we had ground truth)
                # For now, just store the matrix
                results[model_name][block_size][bcst_amount] = {
                    "matrix_shape": matrix.shape,
                    "fire_blocks": int(np.sum(matrix == 0)),
                    "non_fire_blocks": int(np.sum(matrix == 1)),
                    "visualization_path": save_path,
                }

    # ========== SAVE RESULTS ==========
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    Path("results/block_size").mkdir(parents=True, exist_ok=True)

    # Save results as JSON
    results_json = {}
    for model_name, block_results in results.items():
        results_json[model_name] = {}
        for block_size, bcst_results in block_results.items():
            results_json[model_name][str(block_size)] = {}
            for bcst_amount, metrics in bcst_results.items():
                results_json[model_name][str(block_size)][str(bcst_amount)] = metrics

    with open("results/block_size/block_size_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print("\nResults saved to:")
    print("  - results/block_size/block_size_results.json")
    print("  - results/block_size/*/visualization.png")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run block size experiment")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory containing data folders",
    )
    parser.add_argument(
        "--test-image",
        type=str,
        default="data/test/non_fire/block-test.png",
        help="Path to test image",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--patience", type=int, default=None, help="Early stopping patience"
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=600,
        help="Maximum training blocks total (300 per class)",
    )

    args = parser.parse_args()

    run_experiment_block_size(
        data_root=args.data_root,
        test_image_path=args.test_image,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        max_training_blocks=args.max_blocks,
    )

