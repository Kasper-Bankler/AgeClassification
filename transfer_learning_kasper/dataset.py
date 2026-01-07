# Import necessary libraries
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import warnings


# Ignore warnings
warnings.filterwarnings("ignore")

# Class to load images from UTKFace dataset


class UTKFaceImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # Path to images
        self.image_paths = []
        # Ages
        self.labels = []
        # Path to directories with images
        self.root_dir = root_dir
        # Transformations to apply to images
        self.transform = transform

        # Loop through all images in the dataset folder
        for filename in os.listdir(root_dir):
            if filename.endswith(".jpg"):
                # Split filename: "1_0_0_...". The first part is the age, second part is gender, third part is ethnicity and fourth part is a timestamp
                parts = filename.split('_')
                if len(parts) == 4:  # specific format check. If a file does not follow the format, skip it
                    # Extract age
                    age = int(parts[0])
                    # Update lists
                    self.image_paths.append(os.path.join(root_dir, filename))
                    self.labels.append(age)

    # Function to get the length of the dataset
    def __len__(self):
        # Return the total number of samples
        return len(self.image_paths)

    # Function to get an image and its label
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Color image
        # Apply transforms (Resize to 224 for MobileNet)
        if self.transform:
            image = self.transform(image)
        # Get label
        age = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, age


# 1. Rename your existing transform to 'val_transforms' (Clean)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Add a NEW 'train_transforms' (With Randomness)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip left-right
    transforms.RandomRotation(degrees=10),   # Rotate slightly (-10 to 10 deg)
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Random lighting
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
