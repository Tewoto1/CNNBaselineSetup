import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetBaseline(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__()
        self.net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        if num_classes != 1000:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x): return self.net(x)
