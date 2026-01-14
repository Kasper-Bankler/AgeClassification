import os
import sys
import time
import cv2
import torch
import torch.nn as nn
from torchvision import transforms

# --- CLASS DEFINITION (Simple Model) ---
# Defined here so the script is standalone (like TransferAgeModel in the other script)
class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(16 * 56 * 56, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: Using "simple_model_stats.pth" because that is what simple_model_train.py saves
WEIGHTS_PATH = os.path.join(BASE_DIR, "trained_models", "simple_model_stats.pth")

print("RUNNING FILE:", os.path.abspath(__file__))
print("WEIGHTS_PATH:", WEIGHTS_PATH)
print("EXISTS?", os.path.exists(WEIGHTS_PATH))

# --- UI / LABELS ---
WINDOW_NAME = "Age Recognition (press q to quit)"
CLASS_NAMES = ['<16 (Block)', '16-25 (ID Check)', '>25 (Approve)']


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def open_working_camera(max_index=6):
    """
    Cross-platform camera open:
    - macOS: AVFoundation
    - Windows: DirectShow (usually most stable)
    - Linux: falls back to default backend when CAP_DSHOW isn't available/meaningful
    """
    if sys.platform == "darwin":
        backend = cv2.CAP_AVFOUNDATION
    elif sys.platform.startswith("win"):
        backend = cv2.CAP_DSHOW
    else:
        backend = 0  # default backend on Linux

    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, backend) if backend != 0 else cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue

        # Warm-up (helpful on macOS / some webcams)
        for _ in range(10):
            cap.read()
            time.sleep(0.01)

        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"Kamera fundet på index {idx}")
            return cap

        cap.release()

    return None


def main():
    device = get_device()
    print(f"Using device: {device}")

    # 1) Load model + weights
    model = CNN(num_classes=3).to(device)
    
    if os.path.exists(WEIGHTS_PATH):
        state = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state)
        print("Weights loaded successfully.")
    else:
        print(f"ADVARSEL: Kunne ikke finde vægte på {WEIGHTS_PATH}")
        # We continue anyway to test camera, but predictions will be random
    
    model.eval()

    preprocess = build_preprocess()

    # 2) Open webcam (scan indices)
    cap = open_working_camera(max_index=6)
    if cap is None:
        raise RuntimeError(
            "Kunne ikke få frames fra noget kamera.\n"
            "macOS: Systemindstillinger → Privacy & Security → Camera → tillad Terminal/Python/VS Code.\n"
            "Windows: Indstillinger → Privacy → Camera → Allow desktop apps, og luk apps som bruger kameraet (Teams/Zoom)."
        )

    print("Webcam åbnet. Tryk 'q' for at lukke.")

    # 3) Loop
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Kunne ikke læse frame fra kameraet.")
                break

            # BGR -> RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Preprocess -> model
            x = preprocess(rgb).unsqueeze(0).to(device)     # [1, 3, 224, 224]
            logits = model(x)                                # [1, 3]
            probs = torch.softmax(logits, dim=1)[0]          # [3]
            pred_idx = int(torch.argmax(probs).item())
            conf = float(probs[pred_idx].item())

            label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"Class {pred_idx}"
            text = f"{label} | conf: {conf:.2f}"

            # Overlay
            cv2.putText(frame, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow(WINDOW_NAME, frame)

            # Quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()