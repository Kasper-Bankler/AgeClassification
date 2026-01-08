import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class UTKFaceImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.root_dir = root_dir
        self.transform = transform

        for filename in os.listdir(root_dir):
            if filename.endswith(".jpg"):
                parts = filename.split('_')
                if len(parts) == 4:
                    age = int(parts[0])

                    # --- NEW: Convert Age to Class Index ---
                    if age < 16:
                        label = 0  # Class 0: Block (<16)
                    elif age <= 25:
                        label = 1  # Class 1: Manual ID Check (16-25)
                    else:
                        label = 2  # Class 2: Auto-Approve (25+)

                    self.image_paths.append(os.path.join(root_dir, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Return label as LongTensor (Standard for Classification)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


# --- TRANSFORMS (Keep these the same) ---
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
