"""
Utilities for block-based image processing for fire detection.
"""
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import List, Tuple
import random


def extract_blocks_from_image(
    image: Image.Image, block_size: int, stride: int = None
) -> Tuple[List[Image.Image], List[Tuple[int, int]]]:
    """
    Extract blocks from a single image.

    Args:
        image: PIL Image
        block_size: Size of each block (e.g., 8, 16, 32)
        stride: Step size (default: block_size for non-overlapping)

    Returns:
        blocks: List of PIL Images (blocks)
        positions: List of (row, col) tuples indicating block position
    """
    if stride is None:
        stride = block_size

    blocks = []
    positions = []

    width, height = image.size

    for row in range(0, height - block_size + 1, stride):
        for col in range(0, width - block_size + 1, stride):
            # Extract block
            block = image.crop((col, row, col + block_size, row + block_size))
            blocks.append(block)
            # Store position (in block coordinates)
            positions.append((row // block_size, col // block_size))

    return blocks, positions


def extract_blocks_from_dataset(
    image_paths: List[str],
    labels: List[int],
    block_size: int,
    max_samples_per_class: int = None,
    random_seed: int = 42,
) -> Tuple[List[Image.Image], List[int]]:
    """
    Extract blocks from multiple images and sample to fixed number.

    Args:
        image_paths: List of paths to full images
        labels: List of labels (0=non-fire, 1=fire)
        block_size: Size of blocks to extract
        max_samples_per_class: Maximum blocks per class (None = use all)
        random_seed: Random seed for sampling

    Returns:
        block_images: List of PIL Images (blocks)
        block_labels: List of labels for each block
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    all_blocks = {0: [], 1: []}  # Separate by class

    # Extract blocks from all images
    for image_path, label in zip(image_paths, labels):
        try:
            image = Image.open(image_path).convert("RGB")
            blocks, _ = extract_blocks_from_image(image, block_size)
            all_blocks[label].extend(blocks)
        except Exception as e:
            print(f"Warning: Could not load {image_path}: {e}")
            continue

    # Sample to fixed number per class if specified
    block_images = []
    block_labels = []

    for label in [0, 1]:
        blocks = all_blocks[label]
        if max_samples_per_class and len(blocks) > max_samples_per_class:
            # Randomly sample
            sampled_blocks = random.sample(blocks, max_samples_per_class)
        else:
            sampled_blocks = blocks

        block_images.extend(sampled_blocks)
        block_labels.extend([label] * len(sampled_blocks))

    return block_images, block_labels


def predict_image_blocks(
    model, test_image_path: str, block_size: int, device, transform
) -> Tuple[np.ndarray, Image.Image]:
    """
    Predict all blocks in test image and return 2D matrix.

    Args:
        model: Trained PyTorch model
        test_image_path: Path to test image
        block_size: Size of blocks
        device: Device to run inference on
        transform: Transform to apply to blocks

    Returns:
        matrix: 2D numpy array (0=fire, 1=non-fire)
        original_image: PIL Image
    """
    # Load test image
    original_image = Image.open(test_image_path).convert("RGB")
    blocks, positions = extract_blocks_from_image(original_image, block_size)

    # Calculate matrix dimensions
    max_row = max(pos[0] for pos in positions) + 1
    max_col = max(pos[1] for pos in positions) + 1
    matrix = np.ones((max_row, max_col), dtype=int)  # Initialize to 1 (non-fire)

    model.eval()
    with torch.no_grad():
        for block, (row, col) in zip(blocks, positions):
            # Transform block
            block_tensor = transform(block).unsqueeze(0).to(device)

            # Predict
            outputs = model(block_tensor)
            _, predicted = torch.max(outputs, 1)
            prediction = predicted.cpu().item()

            # Store in matrix (0=fire, 1=non-fire)
            # Model outputs: 0=non-fire, 1=fire, so invert
            matrix[row, col] = 1 - prediction

    return matrix, original_image


def visualize_block_predictions(
    original_image: Image.Image,
    matrix: np.ndarray,
    block_size: int,
    save_path: str = None,
) -> Image.Image:
    """
    Create visualization where fire blocks keep original colors, non-fire blocks are black.

    Args:
        original_image: Original test image
        matrix: 2D array (0=fire, 1=non-fire)
        block_size: Size of blocks
        save_path: Path to save visualization (optional)

    Returns:
        Visualization image
    """
    import torch

    output_array = np.array(original_image).copy()
    height, width = output_array.shape[:2]

    # For each block position in matrix
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            start_row = row * block_size
            start_col = col * block_size
            end_row = min(start_row + block_size, height)
            end_col = min(start_col + block_size, width)

            if matrix[row, col] == 1:  # Non-fire detected
                # Make this block black
                output_array[start_row:end_row, start_col:end_col] = [0, 0, 0]

            # If matrix[row, col] == 0 (fire), keep original colors (no change)

    output_image = Image.fromarray(output_array.astype(np.uint8))

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        output_image.save(save_path)
        print(f"Visualization saved to {save_path}")

    return output_image

