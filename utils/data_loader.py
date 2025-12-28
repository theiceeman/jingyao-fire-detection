"""
Data loading and preprocessing utilities for fire detection experiment.
"""

import os
from pathlib import Path
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


class FireDataset(Dataset):
    """Custom dataset for fire/non-fire images."""

    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        """
        Args:
            image_paths: List of paths to images
            labels: List of labels (0 for non-fire, 1 for fire)
            transform: Optional transform to be applied on images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transforms():
    """Get training transforms with augmentation."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_val_transforms():
    """Get validation/test transforms (no augmentation)."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_image_paths(data_dir: str, class_name: str) -> List[str]:
    """Load all image paths from a directory."""
    class_dir = Path(data_dir) / class_name
    if not class_dir.exists():
        return []

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    image_paths = [
        str(p) for p in class_dir.iterdir() if p.suffix.lower() in image_extensions
    ]
    return sorted(image_paths)


def prepare_dataset(
    train_dir: str,
    firelike_dir: str = None,
    num_firelike: int = 0,
):
    """
    Prepare training dataset with optional Fire-like images.
    
    Loads all images from train/fire and train/non_fire folders.
    If num_bcst > 0, adjusts non_fire count to keep total non_fire at 600.
    
    Args:
        train_dir: Directory containing fire and non-fire images
        bcst_dir: Directory containing Fire-like images (optional)
        num_bcst: Number of Fire-like images to add (0, 50, 100, 150...)
    
    Returns:
        (fire_paths, fire_labels, non_fire_paths, non_fire_labels, bcst_paths, bcst_labels)
        Always returns separately for proper stratification
    """
    fire_dir = Path(train_dir) / "fire"
    non_fire_dir = Path(train_dir) / "non_fire"
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    
    # Loads and labels all fire images
    fire_paths = [
        str(p)
        for p in fire_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]
    fire_labels = [1] * len(fire_paths)
    
    # Loads and labels all non-fire images
    all_non_fire_paths = [
        str(p)
        for p in non_fire_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]
    all_non_fire_labels = [0] * len(all_non_fire_paths)
    
    # Load BCST images if specified
    bcst_paths = []
    bcst_labels = []
    if firelike_dir and num_firelike > 0:
        bcst_dir_path = Path(firelike_dir)
        if bcst_dir_path.exists():
            all_bcst_paths = [
                str(p)
                for p in bcst_dir_path.iterdir()
                if p.is_file() and p.suffix.lower() in image_extensions
            ]
            # Take first num_bcst BCST images
            bcst_paths = sorted(all_bcst_paths)[:num_firelike]
            bcst_labels = [0] * len(bcst_paths)
    
    # Adjust non_fire count: if num_bcst > 0, take (600 - num_bcst) non_fire images
    # This keeps total non_fire at 600 (non_fire + firelike = 600)
    if num_firelike > 0:
        target_non_fire = 600 - num_firelike
        non_fire_paths = sorted(all_non_fire_paths)[:target_non_fire]
        non_fire_labels = [0] * len(non_fire_paths)
    else:
        # If no firelike, return all non_fire images
        non_fire_paths = sorted(all_non_fire_paths)
        non_fire_labels = all_non_fire_labels
    
    return (
        fire_paths,
        fire_labels,
        non_fire_paths,
        non_fire_labels,
        bcst_paths,
        bcst_labels,
    )


def create_dataloaders(
    image_paths: List[str], labels: List[int], shuffle: bool = True, transform=None
) -> DataLoader:
    """Create a DataLoader from image paths and labels."""
    dataset = FireDataset(image_paths, labels, transform=transform)
    return DataLoader(
        dataset, batch_size=32, shuffle=shuffle, num_workers=4, pin_memory=True
    )


def extract_features_for_svm(
    model, dataloader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from images using a pre-trained model for SVM.

    Args:
        model: Pre-trained PyTorch model (feature extractor)
        dataloader: DataLoader with images
        device: Device to run inference on

    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)
            batch_features = model(images)
            features.append(batch_features.cpu().numpy())
            labels.append(batch_labels.numpy())

    features = np.vstack(features)
    labels = np.hstack(labels)

    return features, labels
