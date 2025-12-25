"""
EfficientNet model implementation for fire detection.
"""
import torch
import torch.nn as nn
from torchvision import models


class EfficientNetFireDetector(nn.Module):
    """EfficientNet-based fire detector with pre-trained weights."""
    
    def __init__(self, num_classes: int = 2, model_name: str = 'efficientnet_b0'):
        """
        Args:
            num_classes: Number of output classes (2 for binary classification)
            model_name: EfficientNet variant (b0, b1, b2, etc.)
        """
        super(EfficientNetFireDetector, self).__init__()
        
        # Load pre-trained EfficientNet
        # Try new API first (torchvision >= 0.13), fall back to old API
        try:
            from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights
            if model_name == 'efficientnet_b0':
                self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            elif model_name == 'efficientnet_b1':
                self.backbone = models.efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
            elif model_name == 'efficientnet_b2':
                self.backbone = models.efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
            else:
                raise ValueError(f"Unsupported EfficientNet variant: {model_name}")
        except (ImportError, AttributeError):
            # Fall back to old API
            if model_name == 'efficientnet_b0':
                self.backbone = models.efficientnet_b0(pretrained=True)
            elif model_name == 'efficientnet_b1':
                self.backbone = models.efficientnet_b1(pretrained=True)
            elif model_name == 'efficientnet_b2':
                self.backbone = models.efficientnet_b2(pretrained=True)
            else:
                raise ValueError(f"Unsupported EfficientNet variant: {model_name}")
        
        # Replace classifier head
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def extract_features(self, x):
        """Extract features before the classifier for SVM."""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def create_efficientnet_model(num_classes: int = 2, model_name: str = 'efficientnet_b0'):
    """Factory function to create EfficientNet model."""
    return EfficientNetFireDetector(num_classes=num_classes, model_name=model_name)

