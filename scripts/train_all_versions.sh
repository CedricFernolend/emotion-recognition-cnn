#!/bin/bash
# Submit training jobs for all model versions to the SLURM cluster.
# Usage: ./scripts/train_all_versions.sh

cd "$(dirname "$0")/.."

echo "Submitting training jobs for all model versions..."
echo ""

for version in v1 v2 v3 v4; do
    echo "Submitting $version..."
    MODEL_VERSION=$version sbatch train_gpu.slurm
    sleep 2  # Brief pause between submissions
done

echo ""
echo "All jobs submitted! Check status with: squeue -u \$USER"
echo ""
echo "To train only specific versions:"
echo "  MODEL_VERSION=v1 sbatch train_gpu.slurm"
echo "  MODEL_VERSION=v2 sbatch train_gpu.slurm"
echo "  etc."
