#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --account=infra01
#SBATCH --time=00:50:00
#SBATCH --nodes=1                
#SBATCH --ntasks-per-node=1      
#SBATCH --gpus-per-task=4
#SBATCH --partition=normal
#SBATCH --environment=/users/alexpadula/scratch/cifar10-airbench/torch.toml
#SBATCH --output=slurm/out/%x/%j.log
#SBATCH --error=slurm/out/%x/%j.log

set -euo pipefail
nvidia-smi

pip install -U wandb pyhessian

cd /users/alexpadula/scratch/DL-project
python adaptive_sharpness_sweep.py --config sharpness_configs/adaptive_sharpness_sweep.json