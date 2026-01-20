# AgeClassification

A PyTorch-based age classification system that categorizes faces into three age groups: Under 16, 16-25, and Over 25. The project focuses on two approaches: a custom CNN model and a transfer learning model using MobileNetV2, both trained on the UTKFace dataset.

## Features

- **Three-Class Age Classification**: Categorizes faces into age groups (Under 16, 16-25, Over 25)
- **Two Model Architectures**:
  - Simple CNN: Lightweight custom convolutional neural network
  - Transfer Learning: Fine-tuned MobileNetV2 for improved accuracy
- **Real-time Inference**: Live webcam support for age classification
- **Custom Metrics**: Tracks "illegal sales" (minors classified as 25+) and "customer annoyance" (adults flagged as under 25)
- **Cross-platform Support**: Compatible with Windows, macOS, and Linux
- **Hardware Acceleration**: Supports CUDA, MPS (Apple Silicon), and CPU

## Dataset

This project uses the **UTKFace** dataset, which contains over 20,000 face images with annotations for age, gender, and ethnicity. The images are labeled with ages ranging from 0 to 116 years old.

**Dataset Structure**: Images are named with the format `[age]_[gender]_[ethnicity]_[timestamp].jpg`

**Age Group Mapping**:
- Class 0: Under 16 years old
- Class 1: 16-25 years old
- Class 2: Over 25 years old

Place the UTKFace dataset in the `data/UTKFace/` directory.

## Installation

### Prerequisites

- Python 3.7+
- PyTorch
- OpenCV
- torchvision
- matplotlib
- PIL (Pillow)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Kasper-Bankler/AgeClassification.git
cd AgeClassification
```

2. Install dependencies:
```bash
pip install torch torchvision opencv-python matplotlib pillow
```

3. Download the UTKFace dataset and place it in the `data/UTKFace/` directory.

## Usage

### Training Models

#### Simple CNN Model

Train the custom CNN model:

```bash
python simple_train.py
```

**Configuration**:
- Batch size: 32
- Learning rate: 0.0001
- Epochs: 10
- Class weights: [2.0, 2.0, 0.5] (weighted to focus on minors)

The trained model will be saved as `simple_model_stats.pth` with training metrics visualized in `simple_training_metric.png`.

#### Transfer Learning Model

Train the MobileNetV2-based transfer learning model:

```bash
python transfer_train.py
```

**Configuration**:
- Architecture: MobileNetV2 (pretrained on ImageNet)
- Fine-tuning: Last 4 blocks unfrozen
- Batch size: 32
- Learning rate: 0.0001
- Epochs: 10

The trained model will be saved as `final_model_stats.pth` with training metrics in `training_metric.png`.

### Real-time Inference

#### Simple Model Inference

Run age classification using the simple CNN model with your webcam:

```bash
python simple_camera_inference.py
```

Make sure your trained model file (`simple_model.pth` or `simple_model_stats.pth`) is in the `trained_models/` folder.

#### Transfer Model Inference

Run age classification using the transfer learning model:

```bash
python transfer_camera_inference.py
```

**Controls**:
- Press `q` to quit the inference window

**Permissions**: On macOS, ensure camera permissions are granted to your terminal/IDE in System Settings → Privacy & Security → Camera.

## Model Architectures

### Simple CNN

A lightweight custom convolutional neural network:
- **Input**: 224x224 RGB images
- **Feature Extractor**: 2 convolutional blocks (3→16→16 channels)
- **Pooling**: MaxPool2d after each conv block
- **Classifier**: Fully connected layer with dropout (0.3)
- **Output**: 3 classes

### Transfer Learning (MobileNetV2)

Fine-tuned MobileNetV2 architecture:
- **Base Model**: MobileNetV2 pretrained on ImageNet
- **Fine-tuning**: Last 4 blocks trainable
- **Custom Classifier**: 
  - Dropout (0.3)
  - Linear (1280 → 512)
  - ReLU
  - Dropout (0.3)
  - Linear (512 → 3)
- **Output**: 3 classes

## Project Structure

```
AgeClassification/
├── data/
│   └── UTKFace/              # Dataset directory
├── trained_models/           # Saved model weights
├── plots/                    # Training visualizations
├── dataset.py               # Dataset loader and transforms
├── simple_train.py          # Simple CNN training script
├── transfer_model.py        # Transfer learning model definition
├── transfer_train.py        # Transfer learning training script
├── simple_camera_inference.py   # Real-time inference (simple model)
├── transfer_camera_inference.py # Real-time inference (transfer model)
└── README.md
```

## Training Metrics

Both models track the following metrics during training:
- **Training Loss**: Cross-entropy loss on training set
- **Validation Loss**: Cross-entropy loss on validation set
- **Illegal Sales Rate**: Percentage of under-16 individuals classified as 25+
- **Customer Annoyance**: Percentage of 25+ individuals flagged as under 25

Class weights are applied to prioritize correct classification of minors to minimize illegal sales risk.

## Requirements

- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision utilities and pretrained models
- `opencv-python` - Real-time computer vision and webcam access
- `matplotlib` - Training metric visualization
- `Pillow` - Image processing

## Device Support

The code automatically detects and uses the best available hardware:
- **CUDA**: NVIDIA GPUs
- **MPS**: Apple Silicon (M1/M2/M3 chips)
- **CPU**: Fallback for systems without GPU acceleration
