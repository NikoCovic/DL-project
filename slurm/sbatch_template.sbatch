#!/bin/bash
#SBATCH --job-name=airbench_cifar10_train
#SBATCH --account=deep_learning
#SBATCH --time=01:00:00
#SBATCH --output=slurm/out/%x/%j.log
#SBATCH --error=slurm/out/%x/%j.log
#SBATCH --mail-user=blipshitz@ethz.ch
#SBATCH --mail-type=END,FAIL

set -euo pipefail
nvidia-smi

# Run the script
uv run .py
