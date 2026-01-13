# Import necessary modules
import torch.nn as nn
from torchvision import models

# Class defining the transfer learning model for age classification


class TransferAgeModel(nn.Module):
    def __init__(self):
        # Inherit from nn.Module
        super(TransferAgeModel, self).__init__()

        # Load Pretrained MobileNet
        self.net = models.mobilenet_v2(weights='DEFAULT')

        # Freeze all layers
        for param in self.net.features.parameters():
            param.requires_grad = False

        # Unfreeze the last 4 blocks for fine-tuning
        for param in self.net.features[-4:].parameters():
            param.requires_grad = True

        # New Classifier Head
        last_channel = self.net.last_channel  # 1280

        # Overwrite the classifier with a new one for 3 classes
        self.net.classifier = nn.Sequential(
            # Dropout to prevent overfitting
            nn.Dropout(0.3),
            nn.Linear(last_channel, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Output 3 classes
            nn.Linear(512, 3)
        )

    # Function for forward pass
    def forward(self, x):
        return self.net(x)
