import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from dataset import UTKFaceImageDataset, train_transforms, val_transforms
from camera_inference2 import run_webcam  # Webcam + face detection

# ---------------- CONFIG ---------------- #
DATA_DIR = "./data/UTKFace"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 7
LEARNING_RATE = 0.0001
VAL_SPLIT = 0.2
MODEL_PATH = "final_age_model.pth"

CLASS_NAMES = {0: "0-15", 1: "16-25", 2: "25+"}

# ---------------- DEVICE ---------------- #
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

# ---------------- CNN MODEL ---------------- #
class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 28, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(28, 56, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(56, 112, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(112, 224, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Freeze første lag
        for param in self.features[0].parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224 * 14 * 14, 224),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(224, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ---------------- CLASS WEIGHTS ---------------- #
def calculate_class_weights(dataset):
    labels = dataset.labels
    counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)

# ---------------- TRAIN FUNCTION ---------------- #
def train():
    plt.ion()
    plt.figure(figsize=(15,5))

    dataset = UTKFaceImageDataset(DATA_DIR, transform=train_transforms)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    val_ds.dataset.transform = val_transforms

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {train_size}, Validation samples: {val_size}")

    model = CNN().to(device)
    weights = calculate_class_weights(dataset)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_illegal = 100.0

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        minors = illegal = adults = annoyed = 0
        with torch.no_grad():
            for imgs, labels, ages in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                for i in range(len(ages)):
                    age = ages[i].item()
                    pred = preds[i].item()
                    if age < 18:
                        minors += 1
                        if pred == 2:
                            illegal += 1
                    if age > 25:
                        adults += 1
                        if pred < 2:
                            annoyed += 1

        val_loss /= len(val_loader)
        illegal_pct = 100*illegal/minors if minors else 0
        annoy_pct = 100*annoyed/adults if adults else 0

        print(f"EPOCH {epoch+1}/{EPOCHS} | Train Loss {train_loss:.3f} | Val Loss {val_loss:.3f} | Illegal {illegal_pct:.1f}% | Annoyance {annoy_pct:.1f}%")

        # Save best model
        if illegal_pct < best_illegal:
            best_illegal = illegal_pct
            torch.save({
                "model_state": model.state_dict(),
                "img_size": IMG_SIZE,
                "class_names": CLASS_NAMES
            }, MODEL_PATH)
            print("✔ Model saved")

    plt.ioff()
    plt.savefig("training_history.png")
    print("✔ Training complete, plot saved as 'training_history.png'")

    # ✅ Final class evaluation
    print_final_class_accuracy(model, val_loader, device)

# ---------------- FINAL CLASS ACCURACY ---------------- #
def print_final_class_accuracy(model, loader, device):
    """
    Runs a final pass to calculate and print accuracy per class.
    """
    print("\n--- Final Evaluation by Class ---")
    model.eval()
    class_correct = [0]*len(CLASS_NAMES)
    class_total = [0]*len(CLASS_NAMES)

    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            for i in range(len(labels)):
                label = labels[i].item()
                pred = preds[i].item()
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

    for idx, cname in CLASS_NAMES.items():
        total = class_total[idx]
        correct = class_correct[idx]
        acc = 100*correct/total if total else 0
        print(f"Class {cname}: {acc:.2f}% ({correct}/{total})")

# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    train()
    run_webcam()  # Kører webcam direkte efter træning
