import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class CustomResNet50(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()

        self.backbone = models.resnet50(pretrained=True)

        # Кастомный avgpool
        self.backbone.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        in_features = self.backbone.fc.in_features

        # Кастомный классификатор
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Заморозка backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Размораживаем только fc
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [B, 16, H/2, W/2]
        x = self.pool(F.relu(self.conv2(x)))   # [B, 32, H/4, W/4]

        x = self.gap(x)                         # [B, 32, 1, 1]
        x = x.view(x.size(0), -1)               # [B, 32]

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
