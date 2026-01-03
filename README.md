# Emotion Recognition Project

Team: Zeynep Belde, Alena Daneker, Cedric Fernolend

CNN that recognizes 6 emotions from facial images: happiness, surprise, sadness, anger, disgust, fear.

## Setup

1. Install Python 3.8 or newer

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

On Windows use: `venv\Scripts\activate`

3. Install packages:
```bash
pip install -r requirements.txt
```

## Data Setup

Put your images in folders like this:
```
data/raw/train/happiness/img1.jpg
data/raw/train/anger/img2.jpg
data/raw/val/sadness/img3.jpg
data/raw/test/fear/img4.jpg
```

You need three folders: `train`, `val`, and `test`. Inside each, create 6 folders with these exact names:
- happiness
- surprise
- sadness
- anger
- disgust
- fear

## How to Use

### Train the model:
```bash
cd src
python train.py
```

### Test the model:
```bash
cd src
python evaluate.py
```

### See what the model looks at (Grad-CAM):
```bash
cd src
python gradcam.py
```

Edit the image path in gradcam.py first.

## Files

- `src/config.py` - Change settings here (batch size, learning rate, etc)
- `src/data.py` - Loads images from folders
- `src/model.py` - The CNN architecture
- `src/train.py` - Training script
- `src/evaluate.py` - Testing and results
- `src/gradcam.py` - Visualizations

## Where to Add Your Code

- To change model architecture: edit `src/model.py`
- To change training settings: edit `src/config.py`
- To add data preprocessing: edit `src/data.py`
- To change training loop: edit `src/train.py`

All trained models save to `results/models/`
All visualizations save to `results/visualizations/`
