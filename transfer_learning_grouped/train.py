import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import UTKFaceImageDataset, train_transforms, val_transforms
from model import TransferAgeModel

# --- CONFIG ---
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 10
# Update path if needed
DATA_DIR = "/Users/kasperbankler/Documents/GitHub/AgeClassification/data/UTKFace"

# Device Setup
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using Device: {DEVICE}")


def calculate_class_weights(dataset, device):
    """
    Calculates weights to handle imbalance.
    Formula: Total_Samples / (Num_Classes * Class_Count)
    """
    counts = np.bincount(dataset.labels)
    total = len(dataset.labels)
    num_classes = len(counts)

    weights = total / (num_classes * counts)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    print("\n--- CLASS WEIGHTS ---")
    print(f"Class 0 (<16):       {weights[0]:.2f}")
    print(f"Class 1 (16-25):     {weights[1]:.2f} (This should be high)")
    print(f"Class 2 (25+):       {weights[2]:.2f}")
    print("---------------------")

    return weights_tensor


def main():
    # 1. Data Preparation
    train_ds_full = UTKFaceImageDataset(
        root_dir=DATA_DIR, transform=train_transforms)
    val_ds_full = UTKFaceImageDataset(
        root_dir=DATA_DIR, transform=val_transforms)

    # Split
    indices = torch.randperm(len(train_ds_full)).tolist()
    split = int(0.8 * len(train_ds_full))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_dataset = torch.utils.data.Subset(train_ds_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_ds_full, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Setup Model & Class Weights
    model = TransferAgeModel().to(DEVICE)

    # Calculate weights based on the FULL training set
    # (We use train_ds_full to approximate the distribution)
    class_weights = calculate_class_weights(train_ds_full, DEVICE)

    # LOSS FUNCTION: CrossEntropy with weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)  # Shape: [Batch, 4]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate simple training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # Validation Step
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = 100 * correct_val / total_val
        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # 4. Save Model
    torch.save(model.state_dict(), "class_model_weights.pth")
    print("Classification Model Saved.")

    # 5. Final Detailed Evaluation
    print("\nEvaluating Detailed Metrics...")
    evaluate_model(model, val_loader)


def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print Report
    target_names = ['<16 (Block)', '16-25 (Check ID)', '25+ (Approve)']
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['<16', '16-25', '25+'],   # <--- FIXED
                yticklabels=['<16', '16-25', '25+'])   # <--- FIXED
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (3-Class Model)')
    plt.show()


if __name__ == "__main__":
    main()
