import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# ---------------- IMPORTS FRA DINE MODULER ---------------- #
from dataset import UTKFaceImageDataset, train_transforms, val_transforms
from camera_inference import run_webcam   # <-- webcam + face detection

# ---------------- CONFIG ---------------- #
DATA_DIR = "./data/UTKFace"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0005
VAL_SPLIT = 0.2
MODEL_PATH = "final_age_model.pth"

# Alders-klasser
CLASS_NAMES = {
    0: "0-15",
    1: "16-25",
    2: "25+"
}
# ---------------- DEVICE ---------------- #
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print("Using device:", device)

# ==========================================================
# 🔹 CNN MODEL (LIGGER HER – IKKE IMPORTERET)
# ==========================================================
class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 🔒 Freeze første lag (transfer-agtig stabilitet)
        for param in self.features[0].parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (14) * (14), 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    
# ==========================================================
# 🔹 CLASS WEIGHTS
# ==========================================================
def calculate_class_weights(dataset):
    labels = dataset.labels
    counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)

# ==========================================================
# 🔹 TRAINING
# ==========================================================
# ------------------ TRAIN FUNCTION ------------------ #
def train():
    plt.ion()  # interaktiv mode
    plt.figure(figsize=(15, 5))

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

    history = {"train_loss": [], "val_loss": [], "illegal": [], "annoyance": []}
    best_illegal = 100.0

    for epoch in range(EPOCHS):
        # -------- TRAIN -------- #
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

        # -------- VALIDATION -------- #
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
        illegal_pct = 100 * illegal / minors if minors else 0
        annoy_pct = 100 * annoyed / adults if adults else 0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["illegal"].append(illegal_pct)
        history["annoyance"].append(annoy_pct)

        print(
            f"EPOCH {epoch+1}/{EPOCHS} | "
            f"Train Loss {train_loss:.3f} | "
            f"Val Loss {val_loss:.3f} | "
            f"Illegal {illegal_pct:.1f}% | "
            f"Annoyance {annoy_pct:.1f}%"
        )

        # Save best model
        if illegal_pct < best_illegal:
            best_illegal = illegal_pct
            torch.save({
                "model_state": model.state_dict(),
                "img_size": IMG_SIZE,
                "class_names": CLASS_NAMES
            }, MODEL_PATH)
            print("✔ Model saved")

        # -------- PLOT -------- #
        plt.clf()
        x = range(1, len(history["train_loss"]) + 1)

        plt.subplot(1, 3, 1)
        plt.plot(x, history["train_loss"], label="Train")
        plt.plot(x, history["val_loss"], label="Val")
        plt.title("Loss")
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(x, history["illegal"], "r-o")
        plt.title("Illegal sales (%)")

        plt.subplot(1, 3, 3)
        plt.plot(x, history["annoyance"], "g-o")
        plt.title("Customer annoyance (%)")

        plt.tight_layout()
        plt.pause(0.1)  # kort pause, ikke blokering

    plt.ioff()
    plt.savefig("training_history.png")  # gem som fil
    print("✔ Training complete, plot saved as 'training_history.png'")

# ==========================================================
# 🔹 MAIN
# ==========================================================
if __name__ == "__main__":
    run_webcam()