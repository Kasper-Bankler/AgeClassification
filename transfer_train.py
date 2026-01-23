# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import custom dataset and model
from dataset import UTKFaceImageDataset, train_transforms, val_transforms
from transfer_model import TransferAgeModel

# Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 10
DATA_DIR = "./data/UTKFace/"

# Define class names
CLASS_NAMES = {0: "Under 16", 1: "16-25", 2: "Over 25"}

# Manual Class Weights
# Format: [Weight for "Under 16", Weight for "16-25", Weight for "25+"]
# Higher numbers force the model to pay more attention to that class.
CLASS_WEIGHTS = [2.0, 2.0, 0.5]

# Device configuration
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using Device: {DEVICE}")

# Function to update plots while training the model


def update_plots(history):
    plt.clf()
    epochs_range = range(1, len(history['train_loss']) + 1)

    # Loss Graph
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)

    # Illegal Sales Graph
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history['illegal_sales_pct'], 'r-o')
    plt.title('Illegal Sales Rate\n(% of under 18 classified as 25+)')
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')
    plt.ylim(bottom=0)
    plt.grid(True)

    # Annoyance Graph
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history['annoyance_rate'], 'g-o')
    plt.title('Customer Annoyance\n(% of 25+ flagged as under 25)')
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')
    plt.ylim(bottom=0)
    plt.grid(True)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

# Function to print final class accuracy


def print_final_class_accuracy(model, loader, device):
    """
    Runs a final pass to calculate and print accuracy per class.
    """
    print("\n--- Final Evaluation by Group ---")
    model.eval()

    # Prepare counters
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))

    with torch.no_grad():
        # Iterate through the data loader
        for inputs, labels, _ in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Compare predictions to ground truth
            c = (predicted == labels).squeeze()

            # Update counters
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Print results
    for i in range(3):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(
                f"Accuracy of {CLASS_NAMES[i]:<10}: {acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
        else:
            print(f"Accuracy of {CLASS_NAMES[i]:<10}: N/A (No samples)")

    # Calculate Overall Accuracy
    total_correct = sum(class_correct)
    total_samples = sum(class_total)
    if total_samples > 0:
        overall_acc = 100 * total_correct / total_samples
        print(
            f"Overall Accuracy      : {overall_acc:.2f}% ({int(total_correct)}/{int(total_samples)})")
    print("---------------------------------")

# Main training function


def main():
    # Initialize plotting
    plt.ion()
    plt.figure(figsize=(15, 5))

    # Load dataset
    train_ds_full = UTKFaceImageDataset(
        root_dir=DATA_DIR, transform=train_transforms)
    val_ds_full = UTKFaceImageDataset(
        root_dir=DATA_DIR, transform=val_transforms)

    # Split dataset into training (80%) and validation (20%) sets
    indices = torch.randperm(len(train_ds_full)).tolist()
    split = int(0.8 * len(train_ds_full))
    train_indices = indices[:split]
    val_indices = indices[split:]
    train_dataset = torch.utils.data.Subset(train_ds_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_ds_full, val_indices)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Setup model
    model = TransferAgeModel().to(DEVICE)

    # Convert CLASS_WEIGHTS to tensor
    weights_tensor = torch.tensor(
        CLASS_WEIGHTS, dtype=torch.float32).to(DEVICE)
    print("Using Manual Class Weights:", weights_tensor)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE)

    # History for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'illegal_sales_pct': [],
        'annoyance_rate': []
    }

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        # Statistics counters
        minors_total = 0
        illegal_count = 0
        adults_total = 0
        adults_flagged = 0

        # Validation loop
        with torch.no_grad():
            for inputs, labels, raw_ages in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                pred_cpu = predicted.cpu().numpy()
                age_cpu = raw_ages.numpy()

                # Statistics calculations
                for i in range(len(pred_cpu)):
                    p_class = pred_cpu[i]
                    true_age = age_cpu[i]

                    # Illegal Sales (Under 18 classified as 25+)
                    if true_age < 18:
                        minors_total += 1
                        if p_class == 2:  # Predicted 25+
                            illegal_count += 1

                    # Annoyance (25+ classified as <25)
                    if true_age > 25:
                        adults_total += 1
                        if p_class < 2:  # Predicted <25
                            adults_flagged += 1

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)

        # Calculate percentages
        illegal_pct = (100 * illegal_count /
                       minors_total) if minors_total > 0 else 0
        annoyance_pct = (100 * adults_flagged /
                         adults_total) if adults_total > 0 else 0

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['illegal_sales_pct'].append(illegal_pct)
        history['annoyance_rate'].append(annoyance_pct)

        # Print epoch statistics
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Illegal Sales: {illegal_pct:.1f}% | "
              f"Annoyance: {annoyance_pct:.1f}%")

        update_plots(history)

    # Print accuracy per class after training
    print_final_class_accuracy(model, val_loader, DEVICE)
    # Save final model and plots
    torch.save(model.state_dict(), "transfer_model.pth")
    plt.savefig('training_metric.png')
    plt.ioff()
    plt.show()


# Run the main function
if __name__ == "__main__":
    main()
