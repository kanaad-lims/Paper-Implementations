# AlexNet Implementation in PyTorch — Learner-Oriented Documentation

This document explains the **AlexNet architecture implementation from scratch in PyTorch**. The explanation is structured for **learners** and focuses on understanding each component and design decision.

---

## Table of Contents

1. [What is AlexNet?](#1-what-is-alexnet)
2. [Dataset and Preprocessing](#2-dataset-and-preprocessing)
3. [Device Configuration](#3-device-configuration)
4. [AlexNet Model Architecture](#4-alexnet-model-architecture)
5. [Forward Pass Logic](#5-forward-pass-logic)
6. [Loss Function and Optimizer](#6-loss-function-and-optimizer)
7. [Training Loop](#7-training-loop)
8. [Evaluation Phase](#8-evaluation-phase)
9. [Important Notes for Learners](#9-important-notes-for-learners)
10. [Key Takeaways](#10-key-takeaways)

---

## 1. What is AlexNet?

**AlexNet** is a classic deep convolutional neural network introduced in 2012 that demonstrated the power of deep CNNs for image classification.

### Key Characteristics

- **8 learnable layers** (5 convolutional + 3 fully connected)
- **Large convolution kernels** in early layers for broad feature detection
- **ReLU activations** instead of sigmoid/tanh
- **Max pooling** for spatial downsampling
- **Dropout** for regularization
- **Fully connected layers** for final classification

### Adaptation in This Code

- Works with **224×224 RGB images**
- Classifies images into **101 classes** (Caltech-101 dataset)

---

## 2. Dataset and Preprocessing

### Dataset Used

```python
torchvision.datasets.Caltech101
```

**Properties:**
- Contains 101 object categories
- RGB images of varying resolutions
- Well-suited for testing large CNNs like AlexNet

> **Note:** The header comment mentions CIFAR-10, but the actual dataset used is Caltech-101.

### Image Transformations

```python
transforms.Resize((224, 224))
transforms.ToTensor()
transforms.Normalize(0, 1)
```

| Transformation | Purpose |
|----------------|---------|
| `Resize(224, 224)` | AlexNet was originally designed for 224×224 images |
| `ToTensor()` | Converts image to PyTorch tensor format [C, H, W] |
| `Normalize(0, 1)` | Keeps pixel values in a consistent numeric range |

---

## 3. Device Configuration

```python
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
```

- Uses **GPU** if available, otherwise **CPU**
- Essential for faster training on deep networks
- All tensors and models are moved to this device

---

## 4. AlexNet Model Architecture

### Model Definition

```python
class ConvNet(nn.Module):
```

The model inherits from `nn.Module`, which is required for all PyTorch models.

### Layer-by-Layer Breakdown

#### **Layer 1: First Convolutional Layer**

```python
self.conv1 = nn.Conv2d(3, 96, 11, stride=4, padding=2)
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Input channels | 3 | RGB color channels |
| Output channels | 96 | Number of learned filters |
| Kernel size | 11×11 | Large receptive field for capturing spatial patterns |
| Stride | 4 | Reduces spatial size early for computational efficiency |

**Why large kernels?** Captures low-level spatial patterns across wider areas.

#### **Pooling 1**

```python
self.pool1 = nn.MaxPool2d(3, 2)
```

- Reduces spatial dimensions
- Adds translation invariance
- Retains most important features

#### **Layer 2: Second Convolutional Layer**

```python
self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)
```

- Smaller kernel (5×5)
- Learns more complex patterns from previous features
- Increases depth to 256 channels

#### **Pooling 2**

```python
self.pool2 = nn.MaxPool2d(3, 2)
```

Further spatial dimension reduction.

#### **Layers 3–5: Deeper Convolutional Layers**

```python
self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)
self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
self.conv5 = nn.Conv2d(384, 256, 3, 1, 1)
```

**Why 3×3 kernels?**
- Capture local patterns efficiently
- Preserve spatial size with padding
- Computationally efficient compared to larger kernels

#### **Final Pooling**

```python
self.pool5 = nn.MaxPool2d(3, 2)
```

Produces final feature map size: **256 × 6 × 6**

#### **Fully Connected Layers**

```python
self.fc1 = nn.Linear(256*6*6, 4096)
self.fc2 = nn.Linear(4096, 4096)
self.fc3 = nn.Linear(4096, 101)
```

| Layer | Input Size | Output Size | Purpose |
|-------|------------|-------------|---------|
| `fc1` | 256×6×6 = 9216 | 4096 | Converts spatial features to dense representation |
| `fc2` | 4096 | 4096 | Deep feature abstraction |
| `fc3` | 4096 | 101 | Outputs logits for 101 classes |

### Activation and Regularization

```python
self.relu = nn.ReLU(inplace=True)
self.dropout = nn.Dropout(p=0.5)
```

#### **Why ReLU?**
- Prevents vanishing gradients
- Faster convergence
- Introduces non-linearity
- Produces sparse activations

#### **Why Dropout?**
- Prevents overfitting
- Forces network to learn robust features
- Randomly drops 50% of neurons during training

---

## 5. Forward Pass Logic

```python
def forward(self, x):
```

### Flow Summary

```
Input Image (3×224×224)
    ↓
Conv1 → ReLU → Pool1
    ↓
Conv2 → ReLU → Pool2
    ↓
Conv3 → ReLU
    ↓
Conv4 → ReLU
    ↓
Conv5 → ReLU → Pool5
    ↓
Flatten (256×6×6 → 9216)
    ↓
FC1 → ReLU → Dropout
    ↓
FC2 → ReLU → Dropout
    ↓
FC3 (101 class scores)
```

### Architecture Visualization

```
Input Image
    ↓
Feature Extraction (Convolutional Layers)
    ↓
Flatten
    ↓
Dense Classification (Fully Connected Layers)
    ↓
Class Scores (Logits)
```

---

## 6. Loss Function and Optimizer

### Loss Function

```python
nn.CrossEntropyLoss()
```

- Combines `LogSoftmax` + Negative Log Likelihood
- Standard choice for multi-class classification
- Automatically applies softmax internally

### Optimizer

```python
torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
```

**Why SGD with Momentum?**
- Momentum accelerates convergence
- Reduces oscillations in the loss landscape
- Historically used in the original AlexNet paper
- Momentum term: 0.9 (helps escape local minima)

---

## 7. Training Loop

### Steps per Batch

1. **Forward pass** - compute predictions
2. **Loss computation** - compare with ground truth
3. **Zero gradients** - clear previous gradients
4. **Backpropagation** - compute gradients
5. **Weight update** - apply optimizer step

```python
loss.backward()
optimizer.step()
```

### Epoch-Level Loss Tracking

```python
avg_loss = total_epoch_loss / len(train_loader)
```

- Monitors training progress
- Helps identify convergence issues
- Useful for debugging

---

## 8. Evaluation Phase

### Switching to Evaluation Mode

```python
model.eval()
```

**Effects:**
- Disables dropout layers
- Uses batch normalization running statistics
- Essential for correct inference behavior

### Accuracy Calculation

```python
_, predicted = torch.max(output, 1)
```

- Extracts class with highest logit
- Compares with ground truth labels
- Computes percentage of correct predictions

### Metrics Computed

- **Test Accuracy** - performance on unseen data
- **Train Accuracy** - performance on training data

**Why track both?**
- Identifies **underfitting** (both low)
- Identifies **overfitting** (high train, low test)

---

## 9. Important Notes for Learners

### Architectural Mismatch Awareness

- AlexNet was designed for **ImageNet** (1000 classes, millions of images)
- **Caltech-101** is much smaller
- Risk of **overfitting** due to dataset size

###  Training Duration

```python
epochs = 5  # Very small for AlexNet
```

- 5 epochs is insufficient for convergence
- Expect **low accuracy** unless trained longer
- Original AlexNet trained for 90 epochs

### Normalization Caveat

```python
transforms.Normalize(0, 1)
```

- Standard practice: normalize using **per-channel mean & std**
- Current usage is valid but **non-standard**
- Better approach:
  ```python
  transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
  ```

### Model Size

- AlexNet has **~60 million parameters**
- Requires significant GPU memory
- Consider batch size adjustments if memory issues occur

---

## 10. Key Takeaways

### What You Learn from This Implementation

| Concept | Learning |
|---------|----------|
| **CNN Structure** | How classical CNNs are organized hierarchically |
| **Feature Evolution** | How feature map sizes evolve through layers |
| **Pooling & ReLU** | Why these components are critical for deep learning |
| **Dense Classification** | How fully connected layers perform final classification |
| **Training Pipeline** | End-to-end training & evaluation workflow |
| **Architecture Adaptation** | How to adapt research architectures to real datasets |

### Practical Skills Gained

- Building deep CNNs from scratch
- Understanding layer-wise transformations
- Implementing training loops with PyTorch
- Evaluating model performance
- Debugging deep learning models


