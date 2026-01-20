#!/bin/bash
#SBATCH --job-name=adaptive_sharpness_sweep
#SBATCH --account=deep_learning
#SBATCH --time=02:00:00
#SBATCH --gpus 5060ti:1
#SBATCH --output=slurm/out/%x/%j.log
#SBATCH --error=slurm/out/%x/%j.log
#SBATCH --mail-user=blipshitz@ethz.ch
#SBATCH --mail-type=END,FAIL

set -euo pipefail
nvidia-smi

# Run the script
uv run --python 3.13.8 adaptive_sharpness_sweep.py \
	--config sharpness_configs/asam_sweep.json \
	"$@"