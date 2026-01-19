#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --account=infra01
#SBATCH --time=02:00:00
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
python sharpness_experiment.py --config sharpness_configs/clariden-optim_all-cifar.json