# V1 Model Summary
## Baseline 3-Block CNN for Facial Expression Recognition

---

## Architecture

```
Input: 64×64 RGB Image (normalized to [-1, 1])
          │
          ▼
┌─────────────────────────┐
│  Block 1 (3 → 64 ch)    │  2 Conv layers + Skip Connection
│  Conv 3×3 → BN → ReLU   │
│  Conv 3×3 → BN          │
│  + Skip (1×1 Conv + BN) │
│  → ReLU → MaxPool 2×2   │
└─────────────────────────┘
          │ 32×32×64
          ▼
┌─────────────────────────┐
│  Block 2 (64 → 128 ch)  │  2 Conv layers + Skip Connection
│  Conv 3×3 → BN → ReLU   │
│  Conv 3×3 → BN          │
│  + Skip (1×1 Conv + BN) │
│  → ReLU → MaxPool 2×2   │
└─────────────────────────┘
          │ 16×16×128
          ▼
┌─────────────────────────┐
│  Block 3 (128 → 256 ch) │  3 Conv layers + Skip Connection
│  Conv 3×3 → BN → ReLU   │
│  Conv 3×3 → BN → ReLU   │
│  Conv 3×3 → BN          │
│  + Skip (1×1 Conv + BN) │
│  → ReLU → MaxPool 2×2   │
└─────────────────────────┘
          │ 8×8×256
          ▼
┌─────────────────────────┐
│  Global Average Pooling │
└─────────────────────────┘
          │ 256
          ▼
┌─────────────────────────┐
│  FC: 256 → 128 → ReLU   │
│  Dropout (p=0.5)        │
│  FC: 128 → 6            │
└─────────────────────────┘
          │
          ▼
    Output: 6 Classes
```

---

## Key Specifications

| Parameter | Value |
|-----------|-------|
| Input Size | 64 × 64 × 3 (RGB) |
| Total Parameters | ~1.8M |
| Convolutional Blocks | 3 |
| Filter Progression | 64 → 128 → 256 |
| Kernel Size | 3×3 (all convolutions) |
| Pooling | MaxPool 2×2 |
| Classifier | 256 → 128 → 6 |
| Dropout Rate | 0.5 |
| Output Classes | 6 emotions |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | FER2013 (~29K images) |
| Train/Val/Test Split | 80% / 20% / separate |
| Learning Rate | 0.0003 |
| Optimizer | Adam |
| Batch Size | 32 |
| Max Epochs | 60 |
| Early Stopping | 10 epochs patience |
| Loss Function | CrossEntropyLoss (weighted) |

### Data Augmentation
- Horizontal Flip (p=0.5)
- Random Rotation (±15°)
- Random Translation (±10%)

### NOT Used (saved for V2+)
- SE Attention
- Spatial Attention
- Label Smoothing
- LR Scheduler

---

## Emotion Classes

| Index | Emotion |
|-------|---------|
| 0 | Happiness |
| 1 | Surprise |
| 2 | Sadness |
| 3 | Anger |
| 4 | Disgust |
| 5 | Fear |

---

## Design Rationale

1. **3×3 Convolutions**: Increased effective receptive field while keeping parameters low (VGGNet principle)

2. **Skip Connections**: Enable gradient flow during training (ResNet principle)

3. **BatchNorm**: Stabilizes training, allows higher learning rates

4. **Global Average Pooling**: Reduces overfitting vs. fully connected layers, provides spatial invariance

5. **Dropout (0.5)**: Strong regularization for the classifier head

6. **Simple Baseline**: No attention mechanisms - establishes baseline before adding complexity

---

## Explainability

**Method**: Grad-CAM (Gradient-weighted Class Activation Mapping)

**Target Layer**: Last Conv2d in Block 3 (`block3.body[-2]`)

**Purpose**: Visualize which facial regions the model focuses on for each prediction

---

## File Locations

```
src/models/v1_baseline.py   - Model architecture
src/configs/v1_config.py    - Training configuration
results/v1/models/          - Saved model weights
results/v1/visualizations/  - Confusion matrix, GradCAM
results/v1/history.json     - Training history
```
