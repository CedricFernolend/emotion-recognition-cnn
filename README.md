# Emotion Recognition CNN

Facial emotion recognition system classifying 6 emotions: happiness, surprise, sadness, anger, disgust, fear.

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train model
cd src && python train.py

# Evaluate model
python evaluate.py

# Real-time webcam recognition
python webcam.py
```

## Project Structure

```
├── src/
│   ├── config.py       # Configuration settings
│   ├── model.py        # CNN architecture with spatial attention
│   ├── data.py         # Dataset and data loading
│   ├── train.py        # Training pipeline with early stopping
│   ├── evaluate.py     # Model evaluation and metrics
│   ├── gradcam.py      # Grad-CAM visualization
│   ├── webcam.py       # Real-time webcam recognition
│   └── camera_test.py  # Camera diagnostic utility
├── tests/              # Test suite
├── data/raw/           # Training data (train/val/test splits)
├── results/
│   ├── models/         # Saved model weights
│   └── visualizations/ # Generated visualizations
└── requirements.txt
```

## Data Setup

Organize images in folder structure:
```
data/raw/
├── train/
│   ├── happiness/
│   ├── surprise/
│   ├── sadness/
│   ├── anger/
│   ├── disgust/
│   └── fear/
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

## Usage

### Training
```bash
cd src
python train.py
```
Features: early stopping, learning rate scheduling, model checkpointing.

### Evaluation
```bash
python evaluate.py
```
Generates accuracy metrics, classification report, and confusion matrix.

### Webcam Recognition
```bash
python webcam.py [--model PATH] [--camera ID] [--threshold 0-100] [--no-fps]
```
Controls: `q` to quit, `s` to save frame.

### Grad-CAM Visualization
```bash
python gradcam.py
```
Edit `image_path` in script to visualize model attention on specific images.

### Camera Test
```bash
python camera_test.py [--camera ID]
```
Run before webcam recognition to verify camera access.

## Configuration

Edit `src/config.py` to modify:

| Setting | Default | Description |
|---------|---------|-------------|
| IMAGE_SIZE | 64 | Input image dimensions |
| BATCH_SIZE | 32 | Training batch size |
| LEARNING_RATE | 0.0003 | Initial learning rate |
| NUM_EPOCHS | 30 | Maximum training epochs |
| DROPOUT_RATE | 0.5 | Dropout probability |

## Model Architecture

4-block CNN with:
- Progressive filter expansion: 64 → 128 → 256 → 512
- Spatial attention modules
- Skip connections
- Batch normalization
- Dropout regularization

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Authors

Zeynep Belde, Alena Daneker, Cedric Fernolend
