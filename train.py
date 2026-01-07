import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# Import your custom modules
from dataset import UTKFaceImageDataset, data_transforms
from model import BaselineAgeModel

# --- CONFIGURATION ---
ROOT_DIR = "C:\\Users\\kaspe\\Documents\\GitHub\\AgeClassification\\data\\UTKFace"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10

# Check for Apple Silicon GPU (MPS)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Succesfully using Apple M1 GPU (MPS)!")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")  # For your Desktop
else:
    DEVICE = torch.device("cpu")  # Slow fallback


def main():
    print(f"Training on device: {DEVICE}")

    # 1. Prepare Data
    full_dataset = UTKFaceImageDataset(
        root_dir=ROOT_DIR, transform=data_transforms)

    # Split 80% Train, 20% Test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(
        f"Images: {len(train_dataset)} training, {len(test_dataset)} testing")

    # 2. Initialize Model
    model = BaselineAgeModel().to(DEVICE)

    # 3. Loss and Optimizer
    criterion = nn.L1Loss()  # Mean Absolute Error (Calculates error in years)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)
            # Compare Prediction vs Real Age
            loss = criterion(outputs.view(-1), labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 0:
                print(
                    f"Epoch {epoch+1}, Batch {i}, Loss (MAE): {loss.item():.2f} years")

        # 5. Validation Step (End of every epoch)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs.view(-1), labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)
        print(
            f"--- END EPOCH {epoch+1} --- Average Error on Test Set: {avg_val_loss:.2f} years")

    # 6. Save the model
    torch.save(model.state_dict(), "simple_cnn_model.pth")
    print("Model saved as simple_cnn_model.pth")


if __name__ == "__main__":
    main()

# test
