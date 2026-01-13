import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torchvision import datasets, transforms
from PIL import Image

import cv2
import time
import os

FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

if face_cascade.empty():
    raise IOError("❌ Could not load haarcascade_frontalface_default.xml")


# ---------------- Configuration ---------------- #
DATA_DIR = "./data/age_groups_4"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
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
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

            # 🔒 Frys første conv-lag
        for param in self.features[0].parameters():
            param.requires_grad = False


            self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE//16) * (IMG_SIZE//16), 128),
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

# --- Her indsætter vi de sikre UNDER_18 og OVER_18 lister --- #
UNDER_18_CLASSES = []
OVER_18_CLASSES = []

for cls in class_names:
    age_str = cls.split('-')[0]
    if age_str.endswith('+'):
        age = int(age_str[:-1])  # fjern '+' og konverter til int
    else:
        age = int(age_str)

    if age < 18:
        UNDER_18_CLASSES.append(cls)
    else:
        OVER_18_CLASSES.append(cls)

print(f"Under 18 classes: {UNDER_18_CLASSES}")
print(f"Over 18 classes: {OVER_18_CLASSES}")

# --- Definer threshold for beslutning ---
OVER_18_THRESHOLD = 0.5

val_count = int(len(full_dataset)*VAL_SPLIT)
train_count = len(full_dataset) - val_count
train, val = random_split(full_dataset, [train_count, val_count])
val.dataset.transform = val_transform

print(f"Train samples: {train_count}, Validation samples: {val_count}")

train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ---------------- Initialize Model ---------------- #
model = CNN(num_classes=num_class).to(device)

# ---------------- Class Weights ---------------- #
labels = [label for _, label in full_dataset]
counts = torch.bincount(torch.tensor(labels))
weights = len(labels) / (num_class * counts)
weights = weights.to(device)
print("Class weights:", weights)

criterion = nn.CrossEntropyLoss(weight=weights, reduction= "none")
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------------- Training Function ---------------- #
import matplotlib.pyplot as plt

# ---------------- Training Function ---------------- #
def train():
    best_acc = 0.0

    history = {
        'train_loss': [],
        'val_loss': [],
        'illegal_sales_pct': [],
        'annoyance_rate': []
    }

    plt.ion()
    plt.figure(figsize=(15,5))

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

        # ---------------- Validation ---------------- #
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        minors_total = 0
        illegal_count = 0
        adults_total = 0
        adults_flagged = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

                # Business metrics: illegal sales & annoyance
                for i in range(len(labels)):
                    true_class = labels[i].item()
                    pred_class = preds[i].item()
                    # Antag at 'UNDER_18_CLASSES' og 'OVER_18_CLASSES' eksisterer
                    # illegal: minors <18 fejlagtigt klassificeret som 25+
                    if class_names[true_class] in UNDER_18_CLASSES:
                        minors_total += 1
                        if class_names[pred_class] in OVER_18_CLASSES:
                            illegal_count += 1
                    # annoyance: adults >25 fejlagtigt klassificeret som under 25
                    if class_names[true_class] in OVER_18_CLASSES:
                        adults_total += 1
                        if class_names[pred_class] in UNDER_18_CLASSES:
                            adults_flagged += 1

        val_loss /= val_total
        val_acc = val_correct / val_total
        elapsed = time.time() - t0

        illegal_pct = (100 * illegal_count / minors_total) if minors_total > 0 else 0
        annoyance_pct = (100 * adults_flagged / adults_total) if adults_total > 0 else 0

        # Append history
        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_loss)
        history['illegal_sales_pct'].append(illegal_pct)
        history['annoyance_rate'].append(annoyance_pct)

        # Print status
        print(f"EPOCH {epoch}/{EPOCHS} | "
              f"Train Loss: {avg_loss:.3f}, Train Acc: {avg_acc:.3f} | "
              f"Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f} | "
              f"Illegal Sales: {illegal_pct:.1f}% | "
              f"Annoyance: {annoyance_pct:.1f}% | "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "class_names": class_names,
                "img_size": IMG_SIZE,
                "history": history
            }, MODEL_PATH)
            print(f"Saved best model with val_acc={best_acc:.3f} to {MODEL_PATH}")

        # ---------------- Plots ---------------- #
        plt.clf()
        epochs_range = range(1, len(history['train_loss']) + 1)

        # Loss
        plt.subplot(1, 3, 1)
        plt.plot(epochs_range, history['train_loss'], label='Train Loss')
        plt.plot(epochs_range, history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid(True)

        # Illegal Sales
        plt.subplot(1, 3, 2)
        plt.plot(epochs_range, history['illegal_sales_pct'], 'r-o')
        plt.title('Illegal Sales (% of minors classified as adult)')
        plt.xlabel('Epochs')
        plt.ylabel('%')
        plt.ylim(bottom=0)
        plt.grid(True)

        # Annoyance Rate
        plt.subplot(1, 3, 3)
        plt.plot(epochs_range, history['annoyance_rate'], 'g-o')
        plt.title('Annoyance (% of adults flagged as minor)')
        plt.xlabel('Epochs')
        plt.ylabel('%')
        plt.ylim(bottom=0)
        plt.grid(True)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

    plt.ioff()
    plt.show()

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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            face_img = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb).resize((IMG_SIZE, IMG_SIZE))

            tensor = transforms.functional.to_tensor(pil_img)
            tensor = transforms.functional.normalize(
                tensor,
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(tensor)
                probs = torch.softmax(out, dim=1)

                prob_under18 = probs[0, [class_names.index(c) for c in UNDER_18_CLASSES]].sum().item()
                prob_over18 = probs[0, [class_names.index(c) for c in OVER_18_CLASSES]].sum().item()

                decision = "OVER 18" if prob_over18 > OVER_18_THRESHOLD else "ID REQUIRED 🪪"

            color = (0, 255, 0) if decision == "OVER 18" else (0, 0, 255)

            cv2.putText(
                frame,
                f"{decision} ({prob_over18:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        cv2.imshow("Age Detector (q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    train()
    inference()