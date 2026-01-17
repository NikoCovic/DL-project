#!/bin/bash
#SBATCH --job-name=optima_experiment
#SBATCH --account=deep_learning
#SBATCH --time=01:30:00
#SBATCH --output=slurm/out/%x/%j.log
#SBATCH --error=slurm/out/%x/%j.log
#SBATCH --mail-user=blipshitz@ethz.ch
#SBATCH --mail-type=END,FAIL

set -euo pipefail
nvidia-smi

# Run the script
uv run sharpness_experiment.py --config sharpness_configs/student-optim_all-cifar.json
