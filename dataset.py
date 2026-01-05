import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

class UTKFaceImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Loop through all files in the folder
        for filename in os.listdir(root_dir):
            if filename.endswith(".jpg"):
                # Parse filename: "26_0_2_2017..."
                parts = filename.split('_')
                if len(parts) == 4: # specific format check
                    age = int(parts[0])
                    self.image_paths.append(os.path.join(root_dir, filename))
                    self.labels.append(age)
                    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image (Jpg -> PIL Image)
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB") # Color image
        
        # Apply transforms (Resize to 224 for MobileNet)
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

# --- USAGE ON YOUR DESKTOP ---

# Define the transforms MobileNet expects
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # Resize 200 -> 224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])