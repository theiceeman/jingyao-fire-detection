"""
MobileNet model implementation for fire detection.
"""
import torch
import torch.nn as nn
from torchvision import models


class MobileNetFireDetector(nn.Module):
    """MobileNet-based fire detector with pre-trained weights."""
    
    def __init__(self, num_classes: int = 2, model_name: str = 'mobilenet_v2'):
        """
        Args:
            num_classes: Number of output classes (2 for binary classification)
            model_name: MobileNet variant (mobilenet_v2 or mobilenet_v3)
        """
        super(MobileNetFireDetector, self).__init__()
        
        # Load pre-trained MobileNet
        # Try new API first (torchvision >= 0.13), fall back to old API
        try:
            from torchvision.models import MobileNet_V2_Weights, MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights
            if model_name == 'mobilenet_v2':
                self.backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
                num_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(p=0.2, inplace=False),
                    nn.Linear(num_features, num_classes)
                )
            elif model_name == 'mobilenet_v3_small':
                self.backbone = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
                num_features = self.backbone.classifier[3].in_features
                self.backbone.classifier = nn.Sequential(
                    nn.Linear(num_features, 512),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(512, num_classes)
                )
            elif model_name == 'mobilenet_v3_large':
                self.backbone = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
                num_features = self.backbone.classifier[3].in_features
                self.backbone.classifier = nn.Sequential(
                    nn.Linear(num_features, 512),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(512, num_classes)
                )
            else:
                raise ValueError(f"Unsupported MobileNet variant: {model_name}")
        except (ImportError, AttributeError):
            # Fall back to old API
            if model_name == 'mobilenet_v2':
                self.backbone = models.mobilenet_v2(pretrained=True)
                num_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(p=0.2, inplace=False),
                    nn.Linear(num_features, num_classes)
                )
            elif model_name == 'mobilenet_v3_small':
                self.backbone = models.mobilenet_v3_small(pretrained=True)
                num_features = self.backbone.classifier[3].in_features
                self.backbone.classifier = nn.Sequential(
                    nn.Linear(num_features, 512),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(512, num_classes)
                )
            elif model_name == 'mobilenet_v3_large':
                self.backbone = models.mobilenet_v3_large(pretrained=True)
                num_features = self.backbone.classifier[3].in_features
                self.backbone.classifier = nn.Sequential(
                    nn.Linear(num_features, 512),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(512, num_classes)
                )
            else:
                raise ValueError(f"Unsupported MobileNet variant: {model_name}")
    
    def forward(self, x):
        return self.backbone(x)
    
    def extract_features(self, x):
        """Extract features before the classifier for SVM."""
        if hasattr(self.backbone, 'features'):
            x = self.backbone.features(x)
            x = nn.AdaptiveAvgPool2d(1)(x)
            x = torch.flatten(x, 1)
        else:
            # For MobileNetV3, extract features differently
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
        return x


def create_mobilenet_model(num_classes: int = 2, model_name: str = 'mobilenet_v2'):
    """Factory function to create MobileNet model."""
    return MobileNetFireDetector(num_classes=num_classes, model_name=model_name)

