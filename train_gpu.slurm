#!/bin/bash
#SBATCH --job-name=emotion_cnn          # Job name
#SBATCH --output=logs/slurm_%j.out      # Standard output log (%j = job ID)
#SBATCH --error=logs/slurm_%j.err       # Standard error log
#SBATCH --time=24:00:00                 # Time limit (24 hours)
#SBATCH --partition=NvidiaAll           # GPU partition (adjust based on your cluster)
#SBATCH --ntasks=1                      # Run on a single node

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"

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
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Run training
echo "Starting training..."
cd src
python train.py

# Training complete
echo "Job finished at: $(date)"
