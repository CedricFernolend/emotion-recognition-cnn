# Information for Evaluators
The required python files for generating a csv and process a mp4 are insinde the /eval folder.

```bash
python process_mp4.py video.mp4
```
generates a video_processed.mp4 with gradcam overlay + predicted emotion, similar to the webcam demo

```bash
python predict_to_csv.py 
```
takes all images in the eval_data folder and generates a predictions.csv file inside the eval_results folder. it has the following structure:

    image_name,true_label,predicted_label,match(0,1)

the eval_data folder must be structured like this:
```
eval_data/
├── happiness/
│   └──picture.jpg ...
├── surprise/
├── sadness/
├── anger/
├── disgust/
└── fear/
```

# Facial Emotion Recognition CNN

Classifies 6 emotions **happiness, surprise, sadness, anger, disgust, fear** using a CNN with S&E and Spatial Attention. 

### Final Architecture (V3)

```
Input: 3×64×64
├── Block 1: 3→64   + SE + Spatial(k=7)  → MaxPool → 64×32×32
├── Block 2: 64→128 + SE + Spatial(k=7)  → MaxPool → 128×16×16
├── Block 3: 128→256 + SE + Spatial(k=7) → MaxPool → 256×8×8   (3 convs)
├── Block 4: 256→512 + SE + Spatial(k=3) → MaxPool → 512×4×4
├── Global Average Pool → 512
└── Classifier: Dropout(0.5) → FC(512,128) → ReLU → BN → Dropout(0.3) → FC(128,6)
```

### V1 & V2

**V1** is a 3-block baseline CNN. Serves as performance baseline from the preliminary report.

**V2** adds a 4th convolutional block and SE attention in every block.

## Quick Start

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

cd src
python train.py --version v3        # Train V3 (default)
python train.py --version v1        # Train V1
python evaluate.py --version v3     # Evaluate (also for v1 / v2)
python visualize,py --version v3    # Generate Visualizations (-//-), without argument does all three + comparison visuals
```

## Project Structure

```
├── other/ # contains results for some of our experiments
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
│   ├── v1/                         # our best models all trained on RAF-DB
│   ├── v2/
│   └── v3/
├── train_gpu.slurm                 # SLURM script for training on GPU Server (not used later on, not tested with current model and training conf.)
├── train_colab.ipynb               # Google Colab notebook for training (final models were trained on google collab via this notbook)
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

diffrent data Augmentation has been used for RAF-DB and FER-2013 Datasets. Details in src/configs/v3_config.py (Raf-DB) and src/configs/v3_config.old (FER-2013)

## Training

```bash
cd src
python train.py --version v3 # v1 and v2 are also available
```

### V3 Training parameters

| Setting | FER-2013 | RAF-DB |
|---|---|
| Image Size | 64×64 | -//- |
| Batch Size | 32 | -//- |
| Learning Rate | 0.0003 (scheduled) | 0.00015(scheduled)|
| Epochs (max) | 60 | -//-|
| Dropout | 0.6 / 0.36 | 0.5 / 0.3 |
| Weight Decay | 5e-4 | -//- |
| Label Smoothing | 0.15 | 0.10|
| Early Stopping | patience=10 | -//- |
| Optimizer | Adam | -//- |
| LR Scheduler | ReduceLROnPlateau (factor=0.7, patience=3, min=1e-6) | ReduceLROnPlateau (factor=0.5, patience=5, min=1e-6) |

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

Source FER-2013: https://www.kaggle.com/datasets/msambare/fer2013
Source RAF-DB: https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset

```
for RAF-DB:
1 -> surbprise
2 -> fear
3 -> disgust
4 -> happiness
5 -> sadness
6 -> anger
7 -> neutral(unused)
```
