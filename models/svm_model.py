"""
SVM model implementation for fire detection using CNN features.
"""
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
from torchvision import models


class FeatureExtractor(nn.Module):
    """Feature extractor using pre-trained ResNet for SVM."""
    
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Use ResNet50 as feature extractor (can be changed)
        # Try new API first, fall back to old API
        try:
            from torchvision.models import ResNet50_Weights
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except (ImportError, AttributeError):
            resnet = models.resnet50(pretrained=True)
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class SVMFireDetector:
    """SVM-based fire detector using CNN-extracted features."""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale'):
        """
        Args:
            kernel: SVM kernel type ('rbf', 'linear', 'poly', etc.)
            C: Regularization parameter
            gamma: Kernel coefficient ('scale', 'auto', or float)
        """
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.svm = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
    
    def extract_features(self, dataloader):
        """Extract features from images using the feature extractor."""
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                features = self.feature_extractor(images)
                features_list.append(features.cpu().numpy())
                labels_list.append(labels.numpy())
        
        features = np.vstack(features_list)
        labels = np.hstack(labels_list)
        
        return features, labels
    
    def train(self, train_dataloader):
        """Train the SVM model."""
        print("Extracting features for SVM training...")
        X_train, y_train = self.extract_features(train_dataloader)
        
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print("Training SVM...")
        self.svm.fit(X_train_scaled, y_train)
        print("SVM training completed!")
    
    def predict(self, dataloader):
        """Predict labels for given dataloader."""
        print("Extracting features for SVM prediction...")
        X, y_true = self.extract_features(dataloader)
        
        X_scaled = self.scaler.transform(X)
        y_pred = self.svm.predict(X_scaled)
        
        return y_pred, y_true
    
    def predict_proba(self, dataloader):
        """Predict probabilities for given dataloader."""
        X, y_true = self.extract_features(dataloader)
        X_scaled = self.scaler.transform(X)
        y_proba = self.svm.predict_proba(X_scaled)
        return y_proba, y_true


def create_svm_model(kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale'):
    """Factory function to create SVM model."""
    return SVMFireDetector(kernel=kernel, C=C, gamma=gamma)

