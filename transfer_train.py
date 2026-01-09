import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from dataset import UTKFaceImageDataset, train_transforms, val_transforms
from transfer_model import TransferAgeModel

# --- CONFIG ---
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 10
DATA_DIR = "/Users/kasperbankler/Documents/GitHub/AgeClassification/data/UTKFace"

# Define class names for reporting
CLASS_NAMES = {0: "Under 16", 1: "16-25", 2: "Over 25"}

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using Device: {DEVICE}")

def calculate_class_weights(dataset, device):
    counts = np.bincount(dataset.labels)
    total = len(dataset.labels)
    num_classes = len(counts)
    weights = total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)

def update_plots(history):
    plt.clf()
    epochs_range = range(1, len(history['train_loss']) + 1)

    # 1. Loss Graph
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)

    # 2. Illegal Sales Graph
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history['illegal_sales_pct'], 'r-o')
    plt.title('Illegal Sales Rate\n(% of Minors classified as 25+)')
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')
    plt.ylim(bottom=0)
    plt.grid(True)

    # 3. Annoyance Graph
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history['annoyance_rate'], 'g-o')
    plt.title('Customer Annoyance\n(% of Adults flagged as <25)')
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')
    plt.ylim(bottom=0)
    plt.grid(True)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

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
        for inputs, labels, _ in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Compare predictions to ground truth
            c = (predicted == labels).squeeze()
            
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Print results
    for i in range(3):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"Accuracy of {CLASS_NAMES[i]:<10}: {acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
        else:
            print(f"Accuracy of {CLASS_NAMES[i]:<10}: N/A (No samples)")
    print("---------------------------------")

def main():
    plt.ion()
    plt.figure(figsize=(15, 5))

    # 1. Setup Data
    train_ds_full = UTKFaceImageDataset(root_dir=DATA_DIR, transform=train_transforms)
    val_ds_full = UTKFaceImageDataset(root_dir=DATA_DIR, transform=val_transforms)

    indices = torch.randperm(len(train_ds_full)).tolist()
    split = int(0.8 * len(train_ds_full))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_dataset = torch.utils.data.Subset(train_ds_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_ds_full, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Setup Model
    model = TransferAgeModel().to(DEVICE)
    class_weights = calculate_class_weights(train_ds_full, DEVICE)
    print("Class Weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        'train_loss': [],
        'val_loss': [],
        'illegal_sales_pct': [],
        'annoyance_rate': []     
    }

    # 3. Training Loop
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
        
        # Business metrics
        minors_total = 0
        illegal_count = 0
        adults_total = 0
        adults_flagged = 0
        
        with torch.no_grad():
            for inputs, labels, raw_ages in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() # Accumulate validation loss
                
                _, predicted = torch.max(outputs.data, 1)
                
                pred_cpu = predicted.cpu().numpy()
                age_cpu = raw_ages.numpy()
                
                for i in range(len(pred_cpu)):
                    p_class = pred_cpu[i]
                    true_age = age_cpu[i]
                    
                    # Illegal Sales (Minors < 18)
                    if true_age < 18:
                        minors_total += 1
                        if p_class == 2: # Predicted 25+
                            illegal_count += 1

                    # Annoyance (Adults > 25)
                    if true_age > 25:
                        adults_total += 1
                        if p_class < 2: # Predicted <25
                            adults_flagged += 1

        avg_val_loss = val_loss / len(val_loader)
        
        illegal_pct = (100 * illegal_count / minors_total) if minors_total > 0 else 0
        annoyance_pct = (100 * adults_flagged / adults_total) if adults_total > 0 else 0
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['illegal_sales_pct'].append(illegal_pct)
        history['annoyance_rate'].append(annoyance_pct)

        # PRINT UPDATE: Now includes Validation Loss
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Illegal Sales: {illegal_pct:.1f}% | "
              f"Annoyance: {annoyance_pct:.1f}%")
        
        update_plots(history)

    # 4. Final Wrap up
    print_final_class_accuracy(model, val_loader, DEVICE) # Run final detailed report

    torch.save(model.state_dict(), "final_model_stats.pth")
    plt.savefig('training_metric2.png')
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()