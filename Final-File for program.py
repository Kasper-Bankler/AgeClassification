# ---------------- IMPORT LIBRARIES ---------------- #
import torch                          # Core PyTorch library
import torch.nn as nn                 # Neural network modules (layers, loss functions)
import torch.optim as optim           # Optimizers (Adam, SGD, etc.)
from torch.utils.data import DataLoader, random_split  # Data loading utilities
import matplotlib.pyplot as plt       # Plotting library
import numpy as np                    # Numerical operations
from dataset import UTKFaceImageDataset, train_transforms, val_transforms
from camera_inference_for_4layer import run_webcam  # Webcam inference after training

# ---------------- CONFIGURATION ---------------- #
DATA_DIR = "./data/UTKFace"            # Path to UTKFace dataset
IMG_SIZE = 224                         # Input image size for the CNN
BATCH_SIZE = 32                        # Number of images per batch
EPOCHS = 7                             # Number of training epochs
LEARNING_RATE = 0.0001                 # Learning rate for optimizer
VAL_SPLIT = 0.2                        # Percentage of data used for validation
MODEL_PATH = "./trained_models/final_age_model.pth"     # File to save the trained model

# Class label mapping
CLASS_NAMES = {0: "0-15", 1: "16-25", 2: "25+"}

# ---------------- DEVICE SETUP ---------------- #
# Use GPU if available, otherwise CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

# ---------------- CNN MODEL DEFINITION ---------------- #
class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # Feature extractor: convolution + activation + pooling
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # Conv layer: 3 input channels (RGB)
            nn.ReLU(),                      # Non-linearity
            nn.MaxPool2d(2),                # Downsample by factor 2

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Freeze the first convolutional layer to stabilize training
        for param in self.features[0].parameters():
            param.requires_grad = False

        # Classifier: fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),                   # Flatten feature maps
            nn.Linear(64 * 56 * 56, 128),   # Fully connected layer
            nn.ReLU(),
            nn.Dropout(0.4),                # Regularization
            nn.Linear(128, num_classes)     # Output logits for each class
        )

    def forward(self, x):
        x = self.features(x)                # Extract features
        return self.classifier(x)           # Classify features

# ---------------- CLASS WEIGHT CALCULATION ---------------- #
def calculate_class_weights(dataset):
    labels = dataset.labels                # All labels in the dataset
    counts = np.bincount(labels)           # Count samples per class
    total = len(labels)                    # Total number of samples
    weights = total / (len(counts) * counts)  # Inverse frequency weighting
    return torch.tensor(weights, dtype=torch.float32).to(device)

# ---------------- TRAINING FUNCTION ---------------- #
def train():
    # Load dataset with training augmentations
    dataset = UTKFaceImageDataset(DATA_DIR, transform=train_transforms)

    # Split dataset into training and validation sets
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Use validation transforms for validation set
    val_ds.dataset.transform = val_transforms

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {train_size}, Validation samples: {val_size}")

    # Initialize model, loss, and optimizer
    model = CNN().to(device)
    weights = calculate_class_weights(dataset)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_illegal = 100.0  # Track best (lowest) illegal prediction rate

    # Store metrics for plotting
    history = {
        "train_loss": [],
        "val_loss": [],
        "illegal": [],
        "annoyance": []
    }

    # ---------------- TRAINING LOOP ---------------- #
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        # Iterate over training batches
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()           # Reset gradients
            outputs = model(imgs)           # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()                 # Backpropagation
            optimizer.step()                # Update weights

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---------------- VALIDATION ---------------- #
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

                # Domain-specific evaluation metrics
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

        # Print epoch summary
        print(
            f"EPOCH {epoch+1}/{EPOCHS} | "
            f"Train Loss {train_loss:.3f} | "
            f"Val Loss {val_loss:.3f} | "
            f"Illegal {illegal_pct:.1f}% | "
            f"Annoyance {annoy_pct:.1f}%"
        )

        # Save metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["illegal"].append(illegal_pct)
        history["annoyance"].append(annoy_pct)

        # Save best-performing model
        if illegal_pct < best_illegal:
            best_illegal = illegal_pct
            torch.save({
                "model_state": model.state_dict(),
                "img_size": IMG_SIZE,
                "class_names": CLASS_NAMES
            }, MODEL_PATH)

            # Plot training history
            plt.figure(figsize=(15, 5))
            x = range(1, len(history["train_loss"]) + 1)

            plt.subplot(1, 3, 1)
            plt.plot(x, history["train_loss"], label="Train")
            plt.plot(x, history["val_loss"], label="Validation")
            plt.title("Loss")
            plt.legend()

            plt.subplot(1, 3, 2)
            plt.plot(x, history["illegal"], "r-o")
            plt.title("Illegal sales (%)")

            plt.subplot(1, 3, 3)
            plt.plot(x, history["annoyance"], "g-o")
            plt.title("Customer annoyance (%)")

            plt.tight_layout()
            plt.savefig("training_history.png", dpi=300)
            plt.close()

    # Final evaluation
    print_final_class_accuracy(model, val_loader, device)

# ---------------- FINAL CLASS ACCURACY ---------------- #
def print_final_class_accuracy(model, loader, device):
    """Compute and print accuracy per class."""
    print("\n--- Final Evaluation by Class ---")
    model.eval()

    class_correct = [0] * len(CLASS_NAMES)
    class_total = [0] * len(CLASS_NAMES)

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
        acc = 100 * correct / total if total else 0
        print(f"Class {cname}: {acc:.2f}% ({correct}/{total})")

# ---------------- MAIN ENTRY POINT ---------------- #
if __name__ == "__main__":
    train()          # Train the model
    run_webcam()     # Start webcam inference after training
