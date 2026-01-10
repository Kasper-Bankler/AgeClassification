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
        self.ages = []
        # Labels (groups)
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
                if len(parts) == 4: # specific format check. If a file does not follow the format, skip it
                    # Extract age
                    age = int(parts[0])
                    # 3-Class System
                    if age < 16:
                        label = 0  # <16
                    elif age <= 25:
                        label = 1  # 16-25
                    else:
                        label = 2  # 25+
                    # Update lists
                    self.image_paths.append(os.path.join(root_dir, filename))
                    self.labels.append(label)
                    self.ages.append(age)

    # Function to get the length of the dataset
    def __len__(self):
        # Return the total number of samples
        return len(self.image_paths)
    
    # Function to get an image and its label
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB") # Color image
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        # Get label and age
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        age = torch.tensor(self.ages[idx], dtype=torch.float32)
        return image, label, age
    

# Define the transforms MobileNet expects
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # Resize 200 -> 224. This is the size expected by MobileNet
    transforms.ToTensor(), # Convert the image to a PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Adjust the color channels to match what MobileNet was trained on
])
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # Resize 200 -> 224. This is the size expected by MobileNet
    # Data augmentation
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),# Convert the image to a PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Adjust the color channels to match what MobileNet was trained on
])