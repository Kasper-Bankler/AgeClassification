import torch.nn as nn
from torchvision import models


class TransferAgeModel(nn.Module):
    def __init__(self):
        super(TransferAgeModel, self).__init__()

        # 1. Load Pretrained MobileNet
        self.net = models.mobilenet_v2(weights='DEFAULT')

        # 2. Freeze all first
        for param in self.net.features.parameters():
            param.requires_grad = False

        # 3. Unfreeze the last 4 blocks for fine-tuning
        for param in self.net.features[-4:].parameters():
            param.requires_grad = True

        # 4. New Classifier Head
        last_channel = self.net.last_channel  # 1280

        self.net.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(last_channel, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            # CHANGED: Output 4 classes instead of 1
            nn.Linear(512, 4)
        )

    def forward(self, x):
        return self.net(x)
