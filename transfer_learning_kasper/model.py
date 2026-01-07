import torch.nn as nn
from torchvision import models


class TransferAgeModel(nn.Module):
    def __init__(self):
        super(TransferAgeModel, self).__init__()

        # 1. Load Pretrained MobileNet
        self.net = models.mobilenet_v2(weights='DEFAULT')

        # 2. FREEZE start, UNFREEZE end
        # Freeze all first
        for param in self.net.features.parameters():
            param.requires_grad = False

        # Unfreeze the last 3 major blocks (approx last 30 layers)
        # This allows the model to relearn "high level" shapes (faces)
        for param in self.net.features[-4:].parameters():
            param.requires_grad = True

        # 3. Improved Classifier Head
        # We add a hidden layer + Dropout to match the complexity of your Simple Model's head
        last_channel = self.net.last_channel  # 1280

        self.net.classifier = nn.Sequential(
            nn.Dropout(0.3),  # Helps prevent overfitting
            nn.Linear(last_channel, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # More dropout
            nn.Linear(512, 1)  # Output Age
        )

    def forward(self, x):
        return self.net(x)
