# V4 Model Summary
## Final Model with SE + Spatial Attention

---

## Changes from V2

| Change | V2 | V4 | Rationale |
|--------|----|----|-----------|
| **Spatial Attention** | No | Yes | Focus on key facial regions (eyes, mouth, eyebrows) |
| **Learning Rate** | 0.0001 | 0.0003 | Higher capacity model benefits from faster initial learning |
| **Label Smoothing** | 0.0 | 0.1 | Prevent overconfident predictions, improve generalization |
| **LR Scheduler** | No | Yes | Adaptive learning rate for better convergence |
| **Augmentation** | Basic | Full | Color jitter, blur, random erasing for robustness |

### Why These Changes?

**V2 Progress**: Fixed V1's collapse with SE attention and 4th block, achieving proper class distribution.

**V4 Strategy**: Add full enhancements for production-quality model:

1. **Spatial Attention**: Complements SE (channel attention) by focusing on WHERE to look, not just WHAT features matter
2. **Label Smoothing (0.1)**: Prevents model from becoming overconfident, improves calibration
3. **LR Scheduler (ReduceLROnPlateau)**: Automatically reduces learning rate when validation accuracy plateaus
4. **Full Augmentation**: More varied training data for better generalization

---

## Architecture

```
Input: 64x64 RGB Image (normalized to [-1, 1])
          |
          v
+----------------------------------+
|  Layer 1 (3 -> 64 ch)            |  ImprovedBlock with dual attention
|  Conv 3x3 -> BN -> ReLU          |
|  Conv 3x3 -> BN                  |
|  + Skip (1x1 Conv + BN)          |
|  -> ReLU -> SE Attention         |
|  -> Spatial Attention            |
|  -> MaxPool 2x2                  |
+----------------------------------+
          | 32x32x64
          v
+----------------------------------+
|  Layer 2 (64 -> 128 ch)          |  ImprovedBlock with dual attention
|  Conv 3x3 -> BN -> ReLU          |
|  Conv 3x3 -> BN                  |
|  + Skip (1x1 Conv + BN)          |
|  -> ReLU -> SE Attention         |
|  -> Spatial Attention            |
|  -> MaxPool 2x2                  |
+----------------------------------+
          | 16x16x128
          v
+----------------------------------+
|  Layer 3 (128 -> 256 ch)         |  ImprovedBlock with dual attention
|  Conv 3x3 -> BN -> ReLU          |
|  Conv 3x3 -> BN -> ReLU          |
|  Conv 3x3 -> BN                  |
|  + Skip (1x1 Conv + BN)          |
|  -> ReLU -> SE Attention         |
|  -> Spatial Attention            |
|  -> MaxPool 2x2                  |
+----------------------------------+
          | 8x8x256
          v
+----------------------------------+
|  Layer 4 (256 -> 512 ch)         |  ImprovedBlock with dual attention
|  Conv 3x3 -> BN -> ReLU          |
|  Conv 3x3 -> BN                  |
|  + Skip (1x1 Conv + BN)          |
|  -> ReLU -> SE Attention         |
|  -> Spatial Attention            |
|  -> MaxPool 2x2                  |
+----------------------------------+
          | 4x4x512
          v
+----------------------------------+
|  Global Average Pooling          |
+----------------------------------+
          | 512
          v
+----------------------------------+
|  Flatten                         |
|  Dropout (p=0.5)                 |
|  FC: 512 -> 128 -> ReLU          |
|  BatchNorm1d(128)                |
|  Dropout (p=0.3)                 |
|  FC: 128 -> 6                    |
+----------------------------------+
          |
          v
    Output: 6 Classes
```

---

## Attention Mechanisms

### SE Attention (Channel Attention)
```
Input: H x W x C
    |
    v
+-----------------------+
| Global Avg Pool       |  "Squeeze" - compress spatial info
+-----------------------+
    | 1x1xC
    v
+-----------------------+
| FC: C -> C/16         |  "Excitation" - learn channel weights
| ReLU                  |
| FC: C/16 -> C         |
| Sigmoid               |
+-----------------------+
    | 1x1xC (weights)
    v
+-----------------------+
| Scale: Input x W      |  Apply learned weights
+-----------------------+
    | H x W x C
    v
Output (channel-reweighted)
```

### Spatial Attention (NEW in V4)
```
Input: H x W x C
    |
    v
+-----------------------+
| Channel-wise Mean     |  -> H x W x 1
| Channel-wise Max      |  -> H x W x 1
| Concatenate           |  -> H x W x 2
+-----------------------+
    | H x W x 2
    v
+-----------------------+
| Conv2d(2 -> 1, k=7)   |  Learn spatial importance
| Sigmoid               |
+-----------------------+
    | H x W x 1 (spatial map)
    v
+-----------------------+
| Scale: Input x Map    |  Apply spatial weights
+-----------------------+
    | H x W x C
    v
Output (spatially-weighted)
```

**Why Dual Attention?**
- **SE Attention**: Learns WHICH features (channels) are important for each emotion
- **Spatial Attention**: Learns WHERE to focus (eyes vs mouth vs eyebrows)
- Combined: Comprehensive attention for facial expression recognition

---

## Key Specifications

| Parameter | Value |
|-----------|-------|
| Input Size | 64 x 64 x 3 (RGB) |
| Total Parameters | ~5.5M |
| Convolutional Blocks | 4 |
| Filter Progression | 64 -> 128 -> 256 -> 512 |
| SE Reduction Ratio | 16 |
| Spatial Attention Kernel | 7x7 |
| Kernel Size | 3x3 (all convolutions) |
| Pooling | MaxPool 2x2 |
| Classifier | 512 -> 128 -> 6 |
| Dropout Rates | 0.5 (first), 0.3 (second) |
| Output Classes | 6 emotions |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | FER2013 (~29K images) |
| Train/Val/Test Split | 80% / 20% / separate |
| Learning Rate | 0.0003 |
| Optimizer | Adam |
| Weight Decay | 1e-4 |
| Batch Size | 32 |
| Max Epochs | 500 (early stopping) |
| Early Stopping | 10 epochs patience |
| Loss Function | CrossEntropyLoss (weighted) |
| Label Smoothing | 0.1 |
| LR Scheduler | ReduceLROnPlateau |
| LR Scheduler Factor | 0.5 |
| LR Scheduler Patience | 5 |
| Minimum LR | 1e-6 |

### Class Weights (Inverse Frequency)

| Class | Weight | Samples |
|-------|--------|---------|
| Happiness | 0.548 | 7,215 |
| Surprise | 1.248 | 3,171 |
| Sadness | 0.819 | 4,830 |
| Anger | 0.991 | 3,995 |
| Disgust | 9.076 | 436 |
| Fear | 0.966 | 4,097 |

### Data Augmentation (Full Pipeline)
- Horizontal Flip (p=0.5)
- Random Rotation (+/-15 degrees)
- Random Translation (+/-10%)
- **Color Jitter** (brightness=0.2, contrast=0.2, saturation=0.1)
- **Gaussian Blur** (p=0.1, kernel_size=3)
- **Random Erasing** (p=0.1, scale=0.02-0.1)

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

1. **Dual Attention (SE + Spatial)**: Complete attention mechanism - learns both what features matter (channels) and where to focus (spatial regions).

2. **Graduated Dropout (0.5 -> 0.3)**: First dropout layer is aggressive (0.5) for regularization, second is lighter (0.3) to preserve learned representations.

3. **BatchNorm in Classifier**: Added batch normalization between FC layers for stable training.

4. **Label Smoothing (0.1)**: Softens hard labels to prevent overconfident predictions. Instead of [0,0,1,0,0,0], uses [0.017, 0.017, 0.9, 0.017, 0.017, 0.017].

5. **LR Scheduler**: Automatically reduces learning rate by 0.5x when validation accuracy plateaus for 5 epochs. Prevents getting stuck in local minima.

6. **Full Augmentation**: More aggressive data augmentation improves generalization to real-world images.

---

## Actual Results

Training completed successfully:

- [x] Predictions spread across ALL 6 classes
- [x] Per-class recall > 20% for every emotion
- [x] **Test accuracy: 71.74%** (trained for ~60 epochs with early stopping)
- [x] Better calibrated confidence scores (due to label smoothing)
- [x] Improved focus on facial regions (visible in spatial attention maps)
- [x] Stable training (LR scheduler reduced LR as needed)

---

## Explainability

**Method**: Grad-CAM (Gradient-weighted Class Activation Mapping)

**Target Layer**: Last Conv2d in Layer 4 (`layer4.body[-2]`)

**Purpose**: Visualize which facial regions the model focuses on. V4's spatial attention should show more focused activations on relevant facial features.

---

## File Locations

```
src/models/v4_final.py       - Model architecture
src/configs/v4_config.py     - Training configuration
results/v4/models/           - Saved model weights
results/v4/visualizations/   - Confusion matrix, GradCAM
results/v4/history.json      - Training history
```

---

## Training Command

```bash
# Local
cd src && python train.py --version v4

# Cluster (with v1 and v2)
sbatch train_gpu.slurm

# Cluster (v4 only)
MODEL_VERSIONS="v4" sbatch train_gpu.slurm
```

## Evaluation Command

```bash
python evaluate.py --version v4
```
