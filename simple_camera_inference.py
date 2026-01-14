import os
import sys
import time
import cv2
import torch
import torch.nn as nn
from torchvision import transforms

# --- CONFIGURATION ---
# 1. REMOVE the leading slash so os.path.join works correctly
# 2. CHECK if your file is named 'simple_model.pth' or 'simple_model_stats.pth' (default from training)
MODEL_FILENAME = "simple_model.pth" 
SUB_FOLDER = "trained_models"

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# This joins: Base + Folder + Filename safely
WEIGHTS_PATH = os.path.join(BASE_DIR, SUB_FOLDER, MODEL_FILENAME)

print(f"Looking for weights at: {WEIGHTS_PATH}")

# Classes (Must match your training labels)
CLASS_NAMES = ['Under 16', '16-25', 'Over 25']
WINDOW_NAME = "Simple Model Age Recognition"

# --- DEVICE SETUP ---
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

# --- MODEL DEFINITION ---
class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(16 * 56 * 56, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# --- PREPROCESSING ---
def build_preprocess():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# --- CAMERA HANDLING ---
def open_working_camera(max_index=6):
    if sys.platform == "darwin":
        backend = cv2.CAP_AVFOUNDATION
    elif sys.platform.startswith("win"):
        backend = cv2.CAP_DSHOW
    else:
        backend = 0

    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, backend) if backend != 0 else cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue

        for _ in range(10):
            cap.read()
            time.sleep(0.01)

        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"Camera found at index {idx}")
            return cap
        cap.release()
    return None

# --- MAIN LOOP ---
def main():
    # 1. Check weights
    if not os.path.exists(WEIGHTS_PATH):
        print(f"\n[ERROR] Weights file not found!")
        print(f"Checked path: {WEIGHTS_PATH}")
        print(f"Please move your trained model file into the '{SUB_FOLDER}' folder.")
        return

    # 2. Load Model
    print("Loading Simple Model...")
    model = CNN(num_classes=3).to(device)
    
    try:
        state_dict = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.eval()
    preprocess = build_preprocess()

    # 3. Open Camera
    cap = open_working_camera()
    if cap is None:
        print("Could not open any webcam.")
        return

    print("Webcam open. Press 'q' to quit.")

    # 4. Inference Loop
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(rgb_frame).unsqueeze(0).to(device)

            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            
            pred_idx = int(torch.argmax(probs).item())
            confidence = float(probs[pred_idx].item())
            label_text = CLASS_NAMES[pred_idx]

            display_text = f"{label_text} ({confidence:.1%})"
            color = (0, 255, 0) if pred_idx == 2 else (0, 165, 255)

            cv2.putText(frame, display_text, (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()