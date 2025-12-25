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
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_train_transforms():
    """Get training transforms with augmentation."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms():
    """Get validation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def load_image_paths(data_dir: str, class_name: str) -> List[str]:
    """Load all image paths from a directory."""
    class_dir = Path(data_dir) / class_name
    if not class_dir.exists():
        return []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_paths = [
        str(p) for p in class_dir.iterdir() 
        if p.suffix.lower() in image_extensions
    ]
    return sorted(image_paths)


def prepare_baseline_dataset(
    train_dir: str,
    num_fire: int = 600,
    num_non_fire: int = 600
) -> Tuple[List[str], List[int]]:
    """
    Prepare baseline training dataset.
    
    Args:
        train_dir: Directory containing train/fire and train/non_fire folders
        num_fire: Number of fire images to use
        num_non_fire: Number of non-fire images to use
    
    Returns:
        Tuple of (image_paths, labels)
    """
    fire_paths = load_image_paths(train_dir, 'fire')[:num_fire]
    non_fire_paths = load_image_paths(train_dir, 'non_fire')[:num_non_fire]
    
    image_paths = fire_paths + non_fire_paths
    labels = [1] * len(fire_paths) + [0] * len(non_fire_paths)
    
    return image_paths, labels


def prepare_test_dataset(
    test_dir: str,
    bcst_dir: str = None,
    num_bcst: int = 200,
    num_forest_to_remove: int = 200,
    use_bcst: bool = False
) -> Tuple[List[str], List[int]]:
    """
    Prepare test dataset. Can create baseline or BCST-modified version.
    
    Args:
        test_dir: Directory containing test/fire and test/non_fire folders
        bcst_dir: Directory containing BCST images
        num_bcst: Number of BCST images to add (if use_bcst=True)
        num_forest_to_remove: Number of forest images to remove (if use_bcst=True)
        use_bcst: If True, add BCST images and remove forest images. If False, use baseline test set.
    
    Returns:
        Tuple of (image_paths, labels)
    """
    fire_paths = load_image_paths(test_dir, 'fire')
    non_fire_paths = load_image_paths(test_dir, 'non_fire')
    
    if use_bcst:
        # Remove some forest images
        if len(non_fire_paths) > num_forest_to_remove:
            non_fire_paths = non_fire_paths[num_forest_to_remove:]
        
        # Add BCST images
        if bcst_dir:
            bcst_path = Path(bcst_dir)
            if bcst_path.exists():
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
                bcst_paths = [
                    str(p) for p in bcst_path.iterdir() 
                    if p.is_file() and p.suffix.lower() in image_extensions
                ]
                bcst_paths = sorted(bcst_paths)[:num_bcst]
                non_fire_paths.extend(bcst_paths)
    
    image_paths = fire_paths + non_fire_paths
    labels = [1] * len(fire_paths) + [0] * len(non_fire_paths)
    
    return image_paths, labels


def create_dataloaders(
    image_paths: List[str],
    labels: List[int],
    batch_size: int = 32,
    shuffle: bool = True,
    transform=None
) -> DataLoader:
    """Create a DataLoader from image paths and labels."""
    dataset = FireDataset(image_paths, labels, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )


def extract_features_for_svm(
    model,
    dataloader: DataLoader,
    device: torch.device
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

