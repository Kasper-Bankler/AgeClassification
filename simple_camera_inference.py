import os
import sys
import time
import cv2
import torch
import torch.nn as nn
from torchvision import transforms

# ---------------- CONFIG ---------------- #
MODEL_PATH = "final_age_model.pth"
WINDOW_NAME = "Age Recognition (press q to quit)"
CLASS_NAMES = {
    0: "0-15",
    1: "16-25",
    2: "25+"
}
IMG_SIZE = 224  # nu 224 pixels

# ---------------- DEVICE ---------------- #
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

# ==========================================================
# 🔹 CNN MODEL (samme som under træning)
# ==========================================================
class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 28, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(28, 56, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(56, 112, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(112, 224, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Freeze første lag (som i træning)
        for param in self.features[0].parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224 * 14 * 14, 224),  # output fra features: 128*14*14
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(224, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ==========================================================
# 🔹 PREPROCESSING
# ==========================================================
def build_preprocess(img_size=224):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ==========================================================
# 🔹 OPEN CAMERA
# ==========================================================
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
        # Warm-up
        for _ in range(10):
            cap.read()
            time.sleep(0.01)
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"Kamera fundet på index {idx}")
            return cap
        cap.release()
    return None

# ==========================================================
# 🔹 WEBCAM LOOP
# ==========================================================
def run_webcam():
    # Load model
    model = CNN().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    preprocess = build_preprocess(img_size=IMG_SIZE)

    cap = open_working_camera()
    if cap is None:
        raise RuntimeError("Kunne ikke åbne kameraet.")

    print("Webcam åbnet. Tryk 'q' for at lukke.")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Kunne ikke læse frame.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = preprocess(rgb).unsqueeze(0).to(device)  # [1,3,224,224]

            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            conf = float(probs[pred_idx].item())

            label = CLASS_NAMES[pred_idx]
            text = f"{label} | conf: {conf:.2f}"

            # Overlay
            cv2.putText(frame, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# ==========================================================
# 🔹 MAIN
# ==========================================================
if __name__ == "__main__":
    run_webcam()
