import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineAgeModel(nn.Module):
    def __init__(self):
        super(BaselineAgeModel, self).__init__()
        
        # Input: 3 Channels (RGB), 224x224
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2) # 224 -> 112
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2) # 112 -> 56
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2) # 56 -> 28
        
        # Block 4 (Added for 224x224 images)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2) # 28 -> 14
        
        # Fully Connected Layers
        # 14 * 14 image size * 256 channels = 50,176 inputs
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 1) # Output: Age

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        
        # Flatten
        x = x.view(-1, 256 * 14 * 14)
        
        # Classification/Regression Head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x