import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from PIL import Image

import cv2
import time
import os
import numpy as np


import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


face_cascade = cv2.CascadeClassifier(
  "haarcascade_frontalface_default.xml"
)




if face_cascade.empty():
  raise RuntimeError("Failed to load haarcascade_frontalface_default.xml")




# ---------------- Plots ---------------- ##
def update_plots(history):
   plt.gcf().clear()
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


# ---------------- Configuration ---------------- #
DATA_DIR = "UTKFace"
IMG_SIZE = 200
BATCH_SIZE = 32
EPOCHS = 7
LEARNING_RATE = 0.0001
VAL_SPLIT = 0.2
MODEL_PATH = "test_simple_cnn_model.pth"




if torch.backends.mps.is_available():
  device = torch.device("mps")
elif torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")






# ---------------- CNN Model ---------------- #
class CNN(nn.Module):
  def __init__(self):
      super().__init__()


      self.features = nn.Sequential(
          nn.Conv2d(3, 32, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2),


          nn.Conv2d(32, 64, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2),


          nn.Conv2d(64, 128, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2),


          nn.Conv2d(128, 256, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2),
      )




      self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(256 * 12 * 12, 128),
          nn.ReLU(),
          nn.Dropout(0.4),
          nn.Linear(128, 3)
      )


  def forward(self, x):
      x = self.features(x)
      x = self.classifier(x)
      return x


# ---------------- Data Transforms ---------------- #
train_transform = transforms.Compose([
  transforms.Resize((IMG_SIZE, IMG_SIZE)),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_transform = transforms.Compose([
  transforms.Resize((IMG_SIZE, IMG_SIZE)),
  transforms.ToTensor(),
  transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# ---------------- Load Dataset ---------------- #


class AgeDataset(torch.utils.data.Dataset):
  def __init__(self, root, transform=None):
      self.root = root
      self.transform = transform
      self.files = [f for f in os.listdir(root) if f.endswith(".jpg")]

  def __len__(self):
      return len(self.files)


  def __getitem__(self, idx):
      filename = self.files[idx]
      age = int(filename.split("_")[0])


      if age < 16:
          label = 0
      elif age <= 25:
          label = 1
      else:
          label = 2




      img_path = os.path.join(self.root, filename)
      img = Image.open(img_path).convert("RGB")




      if self.transform:
          img = self.transform(img)


      return img, torch.tensor(label, dtype=torch.long), age




full_dataset = AgeDataset(DATA_DIR, transform=train_transform)


val_count   = int(len(full_dataset) * VAL_SPLIT)
train_count = len(full_dataset) - val_count




train, val = random_split(full_dataset, [train_count, val_count])




# validation skal IKKE have augmentation
val.dataset.transform = val_transform




if len(full_dataset) == 0:
  raise ValueError(f"No images found in {DATA_DIR}!")




train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
#num_workers ændret fra 0 til 4




# ---------------- Initialize Model ---------------- #




model = CNN().to(device)




# ---------------- Training Function ---------------- #
def train():
  best_val = float("inf")




  history = {
       "train_loss": [],
       "val_loss": [],
       "illegal_sales_pct": [],
       "annoyance_rate": []
   }
 
  plt.ion()
  plt.figure(figsize=(15, 4))




  for epoch in range(1, EPOCHS+1):
      model.train()


      train_loss = 0.0
      train_total = 0


      train_correct = 0
      train_samples = 0




      t0 = time.time()






      for images, labels, ages in train_loader:


          images, labels = images.to(device), labels.to(device)


          outputs = model(images)


          # Loss per sample
          losses = base_criterion(outputs, labels)  # shape: [B]
          preds = outputs.argmax(dim=1)


          train_correct += (preds == labels).sum().item()
          train_samples += labels.size(0)




          # Straf-faktor
          penalty = torch.ones_like(losses)


          for i, age in enumerate(ages):
              if age < 18 and preds[i].item() == 2:
                  penalty[i] = 5.0   # 🔥 STRAF (kan justeres)


          # Samlet loss
          loss = (losses * penalty).mean()


          optimizer.zero_grad()
          loss.backward()
          optimizer.step()




          train_loss += loss.item() * images.size(0)
          train_total += images.size(0)




       
      train_loss /= train_total
      train_acc = train_correct / train_samples




      # Validation
      model.eval()


      val_loss = 0.0
      val_total = 0
      val_correct = 0
      val_samples = 0


      illegal_sales = 0     # <16 klassificeret som >25
      minor_count = 0


      annoyed_adults = 0    # >25 klassificeret som <25
      adult_count = 0




      with torch.no_grad():
        for images, labels, ages in val_loader:
            images, labels = images.to(device), labels.to(device)


            # Forward
            outputs = model(images)


            # Loss (SKAL blive)
            loss = criterion(outputs, labels)


            # Predictions
            preds = outputs.argmax(dim=1)


            # Accuracy
            val_correct += (preds == labels).sum().item()
            val_samples += labels.size(0)


            # Loss-opsamling
            val_loss += loss.item() * images.size(0)
            val_total += images.size(0)


            # Business-metrikker
            for y_true, y_pred, age in zip(labels, preds, ages):


                # Illegal sale: <16 → >25
                if age < 18:
                  minor_count += 1
                  if y_pred.item() == 2:
                      illegal_sales += 1




                # Annoyance: >25 → <25
                if y_true.item() == 2:
                    adult_count += 1
                    if y_pred.item() in [0, 1]:
                        annoyed_adults += 1






      val_loss /= val_total
      val_acc = val_correct / val_samples


      illegal_sales_pct = (
          illegal_sales / minor_count * 100
          if minor_count > 0 else 0.0
      )


      annoyance_rate = (
          annoyed_adults / adult_count * 100
          if adult_count > 0 else 0.0
  )






      history["train_loss"].append(train_loss)
      history["val_loss"].append(val_loss)


      history["illegal_sales_pct"].append(illegal_sales_pct)
      history["annoyance_rate"].append(annoyance_rate)




      update_plots(history)




      print(
           f"Epoch {epoch}/{EPOCHS} | "
           f"train_loss={train_loss:.4f} | "
           f"train_acc={train_acc*100:.2f}% | "
           f"val_loss={val_loss:.4f} | "
           f"val_acc={val_acc*100:.2f}% | "
           f"time={time.time() - t0:.1f}s"
       )




      # Save best model
      if val_loss < best_val:
          best_val = val_loss
          torch.save(model.state_dict(), MODEL_PATH)




          print(f"Saved model to {MODEL_PATH}")
 
  print("Training completed.")
  plt.ioff()
  plt.show()
 




def print_validation_accuracy_per_class():
    model.eval()
    num_classes = 3
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes


    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)


            outputs = model(images)
            preds = outputs.argmax(dim=1)


            for i in range(len(labels)):
                label = labels[i].item()
                pred = preds[i].item()


                total_per_class[label] += 1
                if pred == label:
                    correct_per_class[label] += 1


    class_names = ["<16", "16-25", ">25"]


    print("\nValidation accuracy per class:")
    for i in range(num_classes):
        if total_per_class[i] > 0:
            acc = correct_per_class[i] / total_per_class[i]
            print(
                f"  {class_names[i]}: {acc*100:.2f}% "
                f"({correct_per_class[i]}/{total_per_class[i]})"
            )
        else:
            print(f"  {class_names[i]}: no samples")


             




# ---------------- Inference Function ---------------- #


def inference():
  if not os.path.exists(MODEL_PATH):
      print(f"Model not found at {MODEL_PATH}. Train first.")
      return


  state_dict = torch.load(MODEL_PATH, map_location=device)
  model.load_state_dict(state_dict)


  model.to(device)
  model.eval()




  print("Webcam starting... press 'q' to quit.")
  webcam = cv2.VideoCapture(0)
  if not webcam.isOpened():
      print("Cannot open webcam")
      return




  while True:
      ret, frame = webcam.read()
      if not ret:
          break




      # Preprocess frame
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(
          gray,
          scaleFactor=1.2,
          minNeighbors=5,
          minSize=(60, 60)
      )


      if len(faces) == 0:
          cv2.imshow("Age Detector (q to quit)", frame)
          if cv2.waitKey(1) & 0xFF == ord("q"):
              break
          continue


      for (x, y, w, h) in faces:
          face = frame[y:y+h, x:x+w]
          rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
          pil_img = Image.fromarray(rgb_face).resize((IMG_SIZE, IMG_SIZE))

          tensor = transforms.functional.to_tensor(pil_img)
          tensor = transforms.functional.normalize(
              tensor,
              [0.485, 0.456, 0.406],
              [0.229, 0.224, 0.225]
          )
          tensor = tensor.unsqueeze(0).to(device)

          with torch.no_grad():
              logits = model(tensor)
              probs = torch.softmax(logits, dim=1)[0]
              pred_class = probs.argmax().item()
              confidence = probs[pred_class].item()


          if pred_class == 0:
              label = "< 16"
          elif pred_class == 1:
              label = "16-25"
          else:
              label = "> 25"


          text = f"{label} ({confidence*100:.1f}%)"


          cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
          cv2.putText(
              frame,
              text,
              (x, y - 10),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.9,
              (0, 255, 0),
              2
          )
          print(f"Predicted age group: {text}")


      cv2.imshow("Age Detector (q to quit)", frame)

      # Print to terminal
      if cv2.waitKey(1) & 0xFF == ord("q"):
          break




  webcam.release()
  cv2.destroyAllWindows()




if __name__ == "__main__":
  MODE = "train"   # "train" eller "run"




  print(f"Device: {device}")
  print(f"Train samples: {train_count}, Validation samples: {val_count}")




  from collections import Counter




  labels = []
  for _, label, _ in full_dataset:
       labels.append(label.item())




  class_counts = Counter(labels)
  print("Class counts:", class_counts)




  num_samples = sum(class_counts.values())
  class_weights = [
       num_samples / (3 * class_counts[i])
       for i in range(3)
   ]




  weights = torch.tensor(class_weights, dtype=torch.float).to(device)
  print("Class weights:", weights)




  base_criterion = nn.CrossEntropyLoss(weight=weights, reduction="none")
  criterion = nn.CrossEntropyLoss(weight=weights)
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)




  if MODE == "train":
      train()
      print_validation_accuracy_per_class()
     
  elif MODE == "run":
      inference()



