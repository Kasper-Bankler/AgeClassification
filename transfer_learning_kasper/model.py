import torch.nn as nn
from torchvision import models

class TransferAgeModel(nn.Module):
    def __init__(self):
        super(TransferAgeModel, self).__init__()
        
        # 1. Load pre-trained MobileNetV2
        # Weights='DEFAULT' is the modern way to load ImageNet weights
        self.net = models.mobilenet_v2(weights='DEFAULT')
        
        # 2. Freeze the "features" (the backbone)
        # This prevents the pre-trained layers from being destroyed by large initial gradients
        for param in self.net.features.parameters():
            param.requires_grad = False
        
        # 3. Modify the Classifier
        # MobileNetV2 classifier is: Sequential(Dropout, Linear(1280, 1000))
        # We replace the Linear layer to output 1 value (Age) instead of 1000 classes
        self.net.classifier[1] = nn.Linear(in_features=1280, out_features=1)

    def forward(self, x):
        # We just call the modified network
        return self.net(x)