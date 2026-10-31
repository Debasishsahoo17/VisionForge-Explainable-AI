"""Neural architectures for the benchmark and optional ImageNet transfer study."""
from __future__ import annotations
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class CustomCNN(nn.Module):
    """Compact convolutional baseline trained from random initialization."""
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(.15), nn.Linear(96, num_classes))
    @property
    def cam_layer(self):
        return self.features[12]  # final convolutional layer
    def forward(self,x):
        return self.head(self.pool(self.features(x)))

def make_resnet18(num_classes=10, mode='digits', pretrained=False):
    """Digits: 1-channel 32px ResNet18; ImageNet: original 3-channel architecture.
    ImageNet weights are NOT claimed for the offline digits model.
    """
    if mode not in ('digits','imagenet'):
        raise ValueError('mode must be digits or imagenet')
    if pretrained and mode != 'imagenet':
        raise ValueError('ImageNet pretrained weights require mode=imagenet')
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    if mode == 'digits':
        model.conv1=nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1,bias=False)
        model.maxpool=nn.Identity()
    model.fc=nn.Linear(model.fc.in_features,num_classes)
    return model

def get_cam_layer(model):
    if isinstance(model,CustomCNN): return model.cam_layer
    return model.layer4[-1].conv2
