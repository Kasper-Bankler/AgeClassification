import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torchvision import datasets, transforms
from PIL import Image

import cv2
import time
import os

# ---------------- Configuration ---------------- #
DATA_DIR = "./data/age_groups"
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.002
VAL_SPLIT = 0.2
MODEL_PATH = "./simple_cnn_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------------- CNN Model ---------------- #
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------- Data Transforms ---------------- #
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------- Load Dataset ---------------- #
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
if len(full_dataset) == 0:
    raise ValueError(f"No images found in {DATA_DIR}!")

class_names = full_dataset.classes
num_class = len(class_names)
print(f"Classes: {class_names}")

val_count = int(len(full_dataset)*VAL_SPLIT)
train_count = len(full_dataset) - val_count
train, val = random_split(full_dataset, [train_count, val_count])
val.dataset.transform = val_transform

print(f"Train samples: {train_count}, Validation samples: {val_count}")

train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ---------------- Initialize Model ---------------- #
model = CNN(num_classes=num_class).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------------- Training Function ---------------- #
def train():
    best_acc = 0.0

    for epoch in range(1, EPOCHS+1):
        model.train()
        avg_loss = 0.0
        avg_correct = 0
        total = 0
        t0 = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            avg_correct += (preds == labels).sum().item()
            total += images.size(0)

        avg_loss /= total
        avg_acc = avg_correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        elapsed = time.time() - t0

        print(f"EPOCH {epoch}/{EPOCHS} | "
              f"Train Loss: {avg_loss:.3f}, Train Acc: {avg_acc:.3f} | "
              f"Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f} | "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "class_names": class_names,
                "img_size": IMG_SIZE
            }, MODEL_PATH)
            print(f"Saved best model with val_acc={best_acc:.3f} to {MODEL_PATH}")

    print("Training completed.")

# ---------------- Inference Function ---------------- #
def inference():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Train first.")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    class_names = checkpoint["class_names"]

    print("Webcam starting... press 'q' to quit.")
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        # Preprocess frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame).resize((IMG_SIZE, IMG_SIZE))
        tensor = transforms.functional.to_tensor(pil_img)
        tensor = transforms.functional.normalize(tensor,[0.485,0.456,0.406],[0.229,0.224,0.225])
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(tensor)
            probs = torch.nn.functional.softmax(out, dim=1)
            top_prob, prob_idx = torch.max(probs, dim=1)
            label = class_names[prob_idx.item()]
            conf = top_prob.item()

        # Display prediction
        cv2.putText(frame, f"{label} {conf:.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.imshow("Age Detector (q to quit)", frame)

        # Print to terminal
        print(f"Predicted: {label}, Confidence: {conf:.2f}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    MODE = "run"   # "train" eller "run"

    if MODE == "train":
        train()
    elif MODE == "run":
        inference()
