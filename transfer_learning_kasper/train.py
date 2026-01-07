import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import UTKFaceImageDataset, train_transforms, val_transforms
from model import TransferAgeModel

# Metrics and Plotting
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# --- Configuration ---
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 10
DATA_DIR = "C:\\Users\\kaspe\\Documents\\GitHub\\AgeClassification\\data\\UTKFace"

# Select Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# --- HELPER FUNCTIONS (Define these OUTSIDE main) ---


def evaluate_legal_accuracy(model, val_loader, device):
    """
    Converts regression output (age) to classification (legal status)
    and calculates accuracy based on Danish laws.
    """
    model.eval()
    true_classes = []
    pred_classes = []
    critical_errors = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)  # <--- FIXED HERE

            # labels are the REAL ages
            outputs = model(inputs)

            # Iterate through the batch
            for i in range(len(labels)):
                real_age = labels[i].item()
                pred_age = outputs[i].item()

                # 1. Determine True Legal Class
                if real_age < 16:
                    true_class = 0
                elif real_age < 18:
                    true_class = 1
                elif real_age < 25:
                    true_class = 2
                else:
                    true_class = 3

                # 2. Determine Predicted Legal Class
                if pred_age < 16:
                    pred_class = 0
                elif pred_age < 18:
                    pred_class = 1
                elif pred_age < 25:
                    pred_class = 2
                else:
                    pred_class = 3

                true_classes.append(true_class)
                pred_classes.append(pred_class)

                # Critical Error: Real < 16, Pred >= 18 (Sold Spirits/Meds)
                if true_class == 0 and pred_class >= 2:
                    critical_errors += 1

    # Metrics
    true_tensor = torch.tensor(true_classes)
    pred_tensor = torch.tensor(pred_classes)
    correct = (true_tensor == pred_tensor).sum().item()
    accuracy = correct / len(true_classes)

    print("\n" + "="*30)
    print("DANISH LEGAL CLASSIFICATION RESULTS")
    print("="*30)
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Critical Safety Failures: {critical_errors}")
    print("-" * 30)

    target_names = ['<16 (Block)', '16-17 (Beer)',
                    '18-24 (ID Check)', '25+ (Approve)']
    print(classification_report(true_classes,
          pred_classes, target_names=target_names))

    return true_classes, pred_classes


def plot_confusion_matrix(true_classes, pred_classes):
    cm = confusion_matrix(true_classes, pred_classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['<16', '16-17', '18-24', '25+'],
                yticklabels=['<16', '16-17', '18-24', '25+'])
    plt.xlabel('Model Prediction')
    plt.ylabel('True Age Group')
    plt.title('Confusion Matrix: Legal Verification')
    plt.show()


# --- MAIN TRAINING LOOP ---

def main():
    # 1. Prepare Data
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

    # 2. Model & Training Setup
    model = TransferAgeModel().to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)  # Reshape labels

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        total_abs_error = 0.0
        total_count = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                total_abs_error += torch.abs(outputs - labels).sum().item()
                total_count += labels.size(0)

        mae = total_abs_error / total_count
        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Val Loss: {val_loss/len(val_loader):.4f} | Val MAE: {mae:.2f} years")

    # 4. Save
    torch.save(model.state_dict(), "age_model_weights.pth")
    print("Model saved.")

    # 5. Evaluate Legal Accuracy (Comparison for Report)
    print("Running Legal Classification Evaluation...")
    true_c, pred_c = evaluate_legal_accuracy(model, val_loader, DEVICE)
    plot_confusion_matrix(true_c, pred_c)


if __name__ == "__main__":
    main()
