import os
import time
import cv2
import torch
from torchvision import transforms

from transfer_model import TransferAgeModel

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "trained_models", "class_model_weights.pth")  # <-- ret til din 3-klasse fil

print("RUNNING FILE:", os.path.abspath(__file__))
print("WEIGHTS_PATH:", WEIGHTS_PATH)
print("EXISTS?", os.path.exists(WEIGHTS_PATH))

# --- UI / LABELS ---
WINDOW_NAME = "Age Recognition (press q to quit)"
CLASS_NAMES = ['<16 (Block)', '16-25 (ID Check)', '>25 (Approve)']  # du ønskede >25

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
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def open_working_camera(max_index=6):
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            continue

        for _ in range(30):
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

    # Load model + weights (skal matche Linear(..., 3))
    model = TransferAgeModel().to(device)
    state = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    preprocess = build_preprocess()

    cap = open_working_camera(max_index=6)
    if cap is None:
        raise RuntimeError("Kunne ikke få frames fra noget kamera (tjek macOS permissions).")

    print("Webcam åbnet. Tryk 'q' for at lukke.")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Kunne ikke læse frame fra kameraet.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = preprocess(rgb).unsqueeze(0).to(device)

            logits = model(x)                       # [1, 3]
            probs = torch.softmax(logits, dim=1)[0] # [3]
            pred_idx = int(torch.argmax(probs).item())
            conf = float(probs[pred_idx].item())

            label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"Class {pred_idx}"
            text = f"{label} | conf: {conf:.2f}"

            cv2.putText(frame, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
