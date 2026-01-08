import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from dataset import UTKFaceImageDataset, train_transforms, val_transforms
from transfer_model import TransferAgeModel

# --- CONFIG ---
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 10
# UPDATE THIS PATH
DATA_DIR = "/Users/kasperbankler/Documents/GitHub/AgeClassification/data/UTKFace"

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

def main():
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
    # Note: using train_ds_full for weights approximation is fine/safer
    class_weights = calculate_class_weights(train_ds_full, DEVICE)
    
    print("Weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- HISTORY TRACKING ---
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'illegal_sales': [],     # Count of <18 sold as 25+
        'annoyance_rate': []     # % of 25+ flagged as young
    }

    # 3. Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Note: We unpack 3 values now (image, label, age)
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        
        # --- VALIDATION LOOP ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        # Custom Stats Counters
        illegal_count = 0
        adults_total = 0
        adults_flagged = 0
        
        with torch.no_grad():
            for inputs, labels, raw_ages in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # --- CALCULATE BUSINESS STATS ---
                pred_cpu = predicted.cpu().numpy()
                age_cpu = raw_ages.numpy()
                
                for i in range(len(pred_cpu)):
                    p_class = pred_cpu[i]
                    true_age = age_cpu[i]
                    
                    # 1. Illegal Sale: Actual < 18 but Predicted Class 2 (25+)
                    if true_age < 18 and p_class == 2:
                        illegal_count += 1
                        
                    # 2. Annoyance: Actual > 25 but Predicted Class 0 (<16) or 1 (16-25)
                    if true_age > 25:
                        adults_total += 1
                        if p_class < 2:
                            adults_flagged += 1

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        annoyance_pct = 100 * adults_flagged / adults_total if adults_total > 0 else 0
        
        # Save Stats
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        history['illegal_sales'].append(illegal_count)
        history['annoyance_rate'].append(annoyance_pct)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_accuracy:.2f}% | "
              f"Illegal Sales: {illegal_count} | "
              f"Annoyance: {annoyance_pct:.1f}%")

    # 4. Save Model & Plots
    torch.save(model.state_dict(), "final_model_stats.pth")
    print("Model saved.")
    
    plot_training_stats(history)

def plot_training_stats(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(18, 5))

    # Plot 1: Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-o', label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot 2: Illegal Sales (The most important chart)
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['illegal_sales'], 'g-o', linewidth=2, label='Illegal Sales (<18 sold as 25+)')
    plt.title('Safety Check: Illegal Sales')
    plt.xlabel('Epochs')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)

    # Plot 3: Annoyance
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['annoyance_rate'], 'm-o', label='Adults Flagged (%)')
    plt.title('Efficiency: Customer Annoyance')
    plt.xlabel('Epochs')
    plt.ylabel('Percentage %')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_stats.png') # Save it so you can put it in report
    plt.show()

if __name__ == "__main__":
    main()