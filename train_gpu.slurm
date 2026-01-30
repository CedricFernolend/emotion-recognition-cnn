#!/bin/bash
#SBATCH --job-name=emotion_cnn          # Job name
#SBATCH --output=logs/slurm_%j.out      # Standard output log (%j = job ID)
#SBATCH --error=logs/slurm_%j.err       # Standard error log
#SBATCH --time=24:00:00                 # Time limit (24 hours)
#SBATCH --partition=NvidiaAll           # GPU partition (adjust based on your cluster)
#SBATCH --ntasks=1                      # Run on a single node

# Model versions to train (space-separated, or "all" for v1-v4)
# Usage:
#   MODEL_VERSIONS="v1 v2" sbatch train_gpu.slurm     # Train v1 and v2
#   MODEL_VERSIONS="v2" sbatch train_gpu.slurm        # Train only v2
#   MODEL_VERSIONS="all" sbatch train_gpu.slurm       # Train all versions
#   sbatch train_gpu.slurm                            # Default: train v1 and v2
MODEL_VERSIONS=${MODEL_VERSIONS:-"v1 v2 v4"}

# Expand "all" to all versions
if [ "$MODEL_VERSIONS" = "all" ]; then
    MODEL_VERSIONS="v1 v2 v4"
fi

# Print job info
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"
echo "Model versions to train: $MODEL_VERSIONS"
echo "=========================================="

# Load necessary modules (adjust these based on your cluster)
# Common examples - check with: module avail
module load python/3.10        # or python/3.11, python3, etc.
module load cuda/11.8          # or cuda/12.1, etc.
# module load cudnn/8.6        # if needed

# Navigate to project directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up Python virtual environment
if [ ! -d "venv_cluster" ]; then
    echo "Creating virtual environment..."
    python -m venv venv_cluster
fi

# Activate virtual environment
source venv_cluster/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Verify GPU is available
echo ""
echo "=========================================="
echo "GPU Check"
echo "=========================================="
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Set data path if not already set (adjust for your cluster's data location)
# export DATA_PATH="/path/to/your/cluster/data/raw"

# Change to src directory
cd src

# Train each version
for VERSION in $MODEL_VERSIONS; do
    echo ""
    echo "=========================================="
    echo "Starting training for model version: $VERSION"
    echo "Time: $(date)"
    echo "=========================================="

    python train.py --version $VERSION
    TRAIN_EXIT_CODE=$?

    if [ $TRAIN_EXIT_CODE -ne 0 ]; then
        echo "ERROR: Training failed for $VERSION with exit code $TRAIN_EXIT_CODE"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Running evaluation for model version: $VERSION"
    echo "=========================================="

    python evaluate.py --version $VERSION

    echo ""
    echo "$VERSION completed at: $(date)"
    echo "=========================================="
done

# Training complete
echo ""
echo "=========================================="
echo "All jobs finished at: $(date)"
echo "Trained versions: $MODEL_VERSIONS"
echo "=========================================="

# Print summary of results
echo ""
echo "=========================================="
echo "Results Summary"
echo "=========================================="
cd ..
for VERSION in $MODEL_VERSIONS; do
    HISTORY_FILE="results/$VERSION/history.json"
    if [ -f "$HISTORY_FILE" ]; then
        echo ""
        echo "=== $VERSION ==="
        python -c "
import json
with open('$HISTORY_FILE') as f:
    h = json.load(f)
print(f\"Best Val Acc: {h.get('best_val_acc', 'N/A'):.2f}%\")
print(f\"Test Acc: {h.get('test_acc', 'N/A'):.2f}%\")
print(f\"Epochs trained: {len(h.get('train_acc', []))}\")
"
    else
        echo "$VERSION: No history file found"
    fi
done
