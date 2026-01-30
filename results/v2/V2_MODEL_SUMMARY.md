# V2 Model Summary
## 4-Block CNN with SE Attention

---

## Changes from V1

| Change | V1 | V2 | Rationale |
|--------|----|----|-----------|
| **Blocks** | 3 | 4 | More model capacity to learn complex patterns |
| **Filters** | 64→128→256 | 64→128→256→512 | Deeper feature hierarchy |
| **SE Attention** | No | Yes | Learn which channels (features) matter most |
| **Learning Rate** | 0.0003 | 0.0001 | More stable training, prevent collapse |
| **Class Weights** | Yes | Yes | Keep addressing class imbalance |
| **Weight Decay** | 0 | 1e-4 | Light L2 regularization |
| **Parameters** | ~1.8M | ~5.5M | Increased capacity |

### Why These Changes?

**V1 Problem**: Model collapsed completely - predicted "happiness" for ALL inputs (29.84% accuracy = happiness class proportion). Zero recall on all other classes.

**V2 Strategy**: Incremental improvement - add capacity and attention before trying more aggressive fixes:

1. **4th Block (512 channels)**: V1 may have lacked capacity to learn discriminative features for all 6 classes
2. **SE Attention**: Helps model learn which feature channels are important for each emotion
3. **Lower LR (0.0001)**: High LR can cause training instability and mode collapse
4. **Weight Decay**: Prevents weights from growing too large

**NOT added yet** (saved for V3 if needed):
- Focal Loss
- Spatial Attention
- Label Smoothing
- LR Scheduler

---

## Architecture

```
Input: 64×64 RGB Image (normalized to [-1, 1])
          │
          ▼
┌─────────────────────────────────┐
│  Block 1 (3 → 64 ch)            │  2 Conv layers + Skip + SE
│  Conv 3×3 → BN → ReLU           │
│  Conv 3×3 → BN                  │
│  + Skip (1×1 Conv + BN)         │
│  → ReLU → SE Attention          │
│  → MaxPool 2×2                  │
└─────────────────────────────────┘
          │ 32×32×64
          ▼
┌─────────────────────────────────┐
│  Block 2 (64 → 128 ch)          │  2 Conv layers + Skip + SE
│  Conv 3×3 → BN → ReLU           │
│  Conv 3×3 → BN                  │
│  + Skip (1×1 Conv + BN)         │
│  → ReLU → SE Attention          │
│  → MaxPool 2×2                  │
└─────────────────────────────────┘
          │ 16×16×128
          ▼
┌─────────────────────────────────┐
│  Block 3 (128 → 256 ch)         │  3 Conv layers + Skip + SE
│  Conv 3×3 → BN → ReLU           │
│  Conv 3×3 → BN → ReLU           │
│  Conv 3×3 → BN                  │
│  + Skip (1×1 Conv + BN)         │
│  → ReLU → SE Attention          │
│  → MaxPool 2×2                  │
└─────────────────────────────────┘
          │ 8×8×256
          ▼
┌─────────────────────────────────┐
│  Block 4 (256 → 512 ch)         │  2 Conv layers + Skip + SE  ← NEW
│  Conv 3×3 → BN → ReLU           │
│  Conv 3×3 → BN                  │
│  + Skip (1×1 Conv + BN)         │
│  → ReLU → SE Attention          │
│  → MaxPool 2×2                  │
└─────────────────────────────────┘
          │ 4×4×512
          ▼
┌─────────────────────────────────┐
│  Global Average Pooling         │
└─────────────────────────────────┘
          │ 512
          ▼
┌─────────────────────────────────┐
│  FC: 512 → 128 → ReLU           │
│  Dropout (p=0.5)                │
│  FC: 128 → 6                    │
└─────────────────────────────────┘
          │
          ▼
    Output: 6 Classes
```

---

## SE Attention Block

Squeeze-and-Excitation (SE) learns to weight channels by importance:

```
Input: H×W×C
    │
    ▼
┌─────────────────────┐
│ Global Avg Pool     │  "Squeeze" - compress spatial info
└─────────────────────┘
    │ 1×1×C
    ▼
┌─────────────────────┐
│ FC: C → C/16        │  "Excitation" - learn channel weights
│ ReLU                │
│ FC: C/16 → C        │
│ Sigmoid             │
└─────────────────────┘
    │ 1×1×C (weights)
    ▼
┌─────────────────────┐
│ Scale: Input × W    │  Apply learned weights
└─────────────────────┘
    │ H×W×C
    ▼
Output (reweighted)
```

**Why SE Attention?**
- Learns which feature channels are informative for each class
- Eyes, mouth, eyebrows activate different channels - SE learns their importance
- Minimal parameter overhead (~0.1% of block parameters)
- Proven effective in image classification (SENet won ImageNet 2017)

---

## Key Specifications

| Parameter | Value |
|-----------|-------|
| Input Size | 64 × 64 × 3 (RGB) |
| Total Parameters | 5,563,910 (~5.5M) |
| Convolutional Blocks | 4 |
| Filter Progression | 64 → 128 → 256 → 512 |
| SE Reduction Ratio | 16 |
| Kernel Size | 3×3 (all convolutions) |
| Pooling | MaxPool 2×2 |
| Classifier | 512 → 128 → 6 |
| Dropout Rate | 0.5 |
| Output Classes | 6 emotions |

### Parameter Breakdown

| Component | Parameters |
|-----------|------------|
| Block 1 (3→64) | 39,808 |
| Block 2 (64→128) | 232,448 |
| Block 3 (128→256) | 1,518,336 |
| Block 4 (256→512) | 3,706,880 |
| Classifier | 66,438 |
| **Total** | **5,563,910** |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | FER2013 (~29K images) |
| Train/Val/Test Split | 80% / 20% / separate |
| Learning Rate | 0.0001 |
| Optimizer | Adam |
| Weight Decay | 1e-4 |
| Batch Size | 32 |
| Max Epochs | 60 |
| Early Stopping | 10 epochs patience |
| Loss Function | CrossEntropyLoss (weighted) |
| Label Smoothing | 0.0 |
| LR Scheduler | None |

### Class Weights (Inverse Frequency)

| Class | Weight | Samples |
|-------|--------|---------|
| Happiness | 0.548 | 7,215 |
| Surprise | 1.248 | 3,171 |
| Sadness | 0.819 | 4,830 |
| Anger | 0.991 | 3,995 |
| Disgust | 9.076 | 436 |
| Fear | 0.966 | 4,097 |

### Data Augmentation
- Horizontal Flip (p=0.5)
- Random Rotation (±15°)
- Random Translation (±10%)

### NOT Used (saved for V3+)
- Focal Loss
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

1. **4th Block (512 channels)**: Increased capacity for learning more complex, discriminative features. V1's 256-channel maximum may have been insufficient.

2. **SE Attention**: Channel attention mechanism that learns which features matter. For facial expressions, different channels encode different facial components (eyes, mouth, eyebrows) - SE helps the model weight them appropriately.

3. **Lower Learning Rate (0.0001)**: V1's collapse suggests training instability. Lower LR provides more gradual updates, reducing risk of the model getting stuck in a degenerate solution.

4. **Weight Decay (1e-4)**: Light L2 regularization prevents weights from growing unbounded, which can contribute to confident but wrong predictions.

5. **Skip Connections with BatchNorm**: ResNet-style shortcuts enable gradient flow through the deeper network.

6. **Global Average Pooling**: Spatial invariance and reduced overfitting vs fully connected layers.

---

## Expected Results

Based on changes made, V2 should show:

- [ ] Predictions spread across ALL 6 classes (not just happiness)
- [ ] Per-class recall > 0% for every emotion
- [ ] Overall accuracy: 50-60%
- [ ] Confusion matrix shows diagonal pattern (not single column)
- [ ] Training curves show gradual improvement (not instant plateau)

---

## Explainability

**Method**: Grad-CAM (Gradient-weighted Class Activation Mapping)

**Target Layer**: Last Conv2d in Block 4 (`block4.body[-2]`)

**Purpose**: Visualize which facial regions the model focuses on for each prediction

---

## File Locations

```
src/models/v2_model.py      - Model architecture
src/configs/v2_config.py    - Training configuration
results/v2/models/          - Saved model weights
results/v2/visualizations/  - Confusion matrix, GradCAM
results/v2/history.json     - Training history
```

---

## Training Command

```bash
./train_docker.sh --version v2
```

## Evaluation Command

```bash
./evaluate_docker.sh --version v2
```
