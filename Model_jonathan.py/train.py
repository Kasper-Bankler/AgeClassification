import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import UTKFaceImageDataset, data_transforms
from model import TransferAgeModel

# --- ADDED (for plotting LR vs loss) ---
import matplotlib.pyplot as plt

# --- Configuration ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
DATA_DIR = "/Users/jonathankragh/Documents/GitHub/AgeClassification/data/UTKFace"

# Select Device (Mac M1/M2, Nvidia GPU, or CPU)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# --- ADDED (function to plot LR vs loss) ---
def plot_lr_vs_loss(model, train_loader, criterion, optimizer,
                    lr_start=1e-7, lr_end=1e-1, num_iters=200):
    """
    Plot learning rate vs. loss (LR range test).
    OBS: Ændrer modelvægte – kør før rigtig træning.
    """
    model.train()

    # Start-LR
    optimizer.param_groups[0]["lr"] = lr_start
    lr_mult = (lr_end / lr_start) ** (1 / num_iters)

    lrs = []
    losses = []

    it = 0
    for inputs, labels in train_loader:
        if it >= num_iters:
            break

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        labels = labels.unsqueeze(1)  # Reshape labels to [batch_size, 1]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        lrs.append(optimizer.param_groups[0]["lr"])
        losses.append(loss.item())

        # Øg LR
        optimizer.param_groups[0]["lr"] *= lr_mult
        it += 1

    # Plot
    plt.figure()
    plt.plot(lrs, losses)
    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("Loss (MSE)")
    plt.title("Learning rate vs. loss")
    plt.show()


def main():
    # 1. Prepare Dataset
    # We use the transforms defined in your dataset.py (imported as data_transforms)
    full_dataset = UTKFaceImageDataset(root_dir=DATA_DIR, transform=data_transforms)
    
    # Split 80% Train / 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Initialize Model
    model = TransferAgeModel().to(DEVICE)
    
    # 3. Setup Loss and Optimizer
    # MSELoss is standard for Regression (Age prediction)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- ADDED (one line call to make the plot) ---
    plot_lr_vs_loss(model, train_loader, criterion, optimizer)

    # 4. Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # CRITICAL: Reshape labels to [batch_size, 1] to match model output
            labels = labels.unsqueeze(1)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate average training loss for this epoch
        avg_train_loss = running_loss / len(train_loader)

        # 5. Validation Loop
        model.eval()
        val_loss = 0.0 
        total_absolute_error = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                labels = labels.unsqueeze(1) # Reshape

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate MAE (Mean Absolute Error) for reporting
                # This tells us: "On average, we are off by X years"
                absolute_errors = torch.abs(outputs - labels)
                total_absolute_error += torch.sum(absolute_errors).item()
                total_samples += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        mae = total_absolute_error / total_samples

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"  Train Loss (MSE): {avg_train_loss:.4f}")
        print(f"  Val Loss (MSE):   {avg_val_loss:.4f}")
        print(f"  Val MAE:          {mae:.2f} years (Average Error)")
        print("-" * 30)

    # Save the trained model
    torch.save(model.state_dict(), "age_model_weights.pth")
    print("Model saved to age_model_weights.pth")

if __name__ == "__main__":
    main()
