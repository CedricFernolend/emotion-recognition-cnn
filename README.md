# Facial Emotion Recognition CNN

Classifies 6 emotions **happiness, surprise, sadness, anger, disgust, fear** using a CNN with S&E and Spatial Attention. 

Three model versions show the improvement of adding onto our baseline v1 Model.

**Authors:** Zeynep Belde, Alena Daneker, Cedric Fernolend

## Model Versions

| | V1 Baseline | V2 SE Attention | V3 SE + Spatial Attention |
|---|---|---|---|
| **Blocks** | 3 | 4 | 4 |
| **Filters** | 64→128→256 | 64→128→256→512 | 64→128→256→512 |
| **SE Attention** | No | Yes | Yes |
| **Spatial Attention** | No | No | Yes |
| **Label Smoothing** | 0.0 | 0.0 | 0.15 |
| **LR Scheduler** | No | No | ReduceLROnPlateau |
| **Dropout** | 0.5 | 0.5 | 0.6 + 0.36 |
| **Weight Decay** | 0 | 1e-4 | 5e-4 |
| **Test Accuracy Fer2013** | 66.85% | 67.72% | **69.89%** |

### V3 Architecture in Detail

```
Input: 3×64×64
├── Block 1: 3→64   + SE + Spatial(k=7)  → MaxPool → 64×32×32
├── Block 2: 64→128 + SE + Spatial(k=7)  → MaxPool → 128×16×16
├── Block 3: 128→256 + SE + Spatial(k=7) → MaxPool → 256×8×8   (3 convs)
├── Block 4: 256→512 + SE + Spatial(k=3) → MaxPool → 512×4×4
├── Global Average Pool → 512
└── Classifier: Dropout(0.6) → FC(512,128) → ReLU → BN → Dropout(0.36) → FC(128,6)
```

### V1 & V2

**V1** is a 3-block baseline CNN. Serves as performance baseline from the preliminary report.

**V2** adds a 4th convolutional block and SE attention in every block, trained at a lower LR=1e-4 with light weight decay. Achieved a +0.87% improvement over V1.

## Quick Start

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

cd src
python train.py --version v3        # Train V3 (default)
python train.py --version v1        # Train V1
python evaluate.py --version v3     # Evaluate (also for v1 / v2)
python visualize,py --version v3    # Generate Visualizations (-//-)
```

## Project Structure

```
├── src/
│   ├── configs/
│   │   ├── base_config.py          # Shared settings
│   │   ├── v1_config.py            # V1 hyperparameters
│   │   ├── v2_config.py            # V2 ..
│   │   └── v3_config.py            # V3 ..
│   ├── models/
│   │   ├── v1_model.py             # Model V1 Specification
│   │   ├── v2_model.py             # Model V2 ..
│   │   └── v3_model.py             # Model V3 ..
│   ├── data.py                     # Dataset loading, augmentation, dataloaders
│   ├── train.py                    # Training pipeline
│   ├── evaluate.py                 # Test accuracy and classification report
│   ├── visualize.py                # Generate visualizations
│   └── webcam_demo.py              # Webcam emotion recognition demo
├── data/raw/                       # train/test splits, 6 emotion folders each
├── results/
│   ├── comparison/                 # Visualization of model comparisons (gradcam, conf. matrix and accuracy on test)
│   ├── v1/                         # Model weights, history.json, visualizations
│   ├── v2/
│   └── v3/
├── train_gpu.slurm                 # SLURM script for training on GPU Server
├── train_colab.ipynb               # Google Colab notebook for training
└── requirements.txt
```

## Data Setup

Organize images into the following folder structure:

```
data/raw/
├── train/
│   ├── happiness/
│   ├── surprise/
│   ├── sadness/
│   ├── anger/
│   ├── disgust/
│   └── fear/
└── test/
    └── (same 6 folders)
```

Images are resized to 64×64, normalized to [-1, 1] range (mean=0.5, std=0.5). 20 Percent of training data gets used as validation dataset.

### Data Augmentation

**V3 data augmentation:** random horizontal flip, rotation (±15°), translation (±10%), color jitter (brightness/contrast=0.2), gaussian blur (p=0.1), random erasing (p=0.1).

**V1/V2 augmentation:** horizontal flip, rotation, translation only.

## Training

```bash
cd src
python train.py --version v3 # v1 and v2 are also available
```

### V3 Training parameters

| Setting | Value |
|---|---|
| Image Size | 64×64 |
| Batch Size | 32 |
| Learning Rate | 0.0003 (scheduled) |
| Epochs (max) | 60 |
| Dropout | 0.6 / 0.36 (graduated) |
| Weight Decay | 5e-4 |
| Label Smoothing | 0.15 |
| Early Stopping | patience=10 |
| Optimizer | Adam |
| LR Scheduler | ReduceLROnPlateau (factor=0.7, patience=3, min=1e-6) |

All version configs are in `src/configs/`.

## Evaluation

```bash
cd src
python evaluate.py --version v3     
```

Outputs test accuracy and per-class precision/recall/F1

### Webcam Demo

```bash
cd src
python webcam_demo.py [--camera ID]
```

Controls: `q` quit, `s` save frame, `1` detection only, `2` GradCAM overlay, `3` Spatial Attention overlay (V3 only), `g` toggle grayscale input, `m` switch model (V3 <-> V1), `SPACE` pause/unpause

### Visualizations

```bash
cd src
python visualize.py                      # Generate all for v1, v2, v3
python visualize.py --version v3         # Single version only
```

Per-version outputs in `results/{version}/visualizations/`:
- `gradcam_{emotion}.png`
- `spatial_attention_{emotion}.png`
- `training_curves.png`
- `confusion_matrix.png`
- `f1_scores.png`

Version comparing outputs in `results/comparison/` (when visualizing multiple versions):
- `accuracy_comparison.png`
- `confusion_matrix_comparison.png`
- `gradcam_comparison.png`

## Dataset

Currently using Fer2013. Future Plans to use RAF-DB 
