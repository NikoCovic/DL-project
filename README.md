Start optima experiment sbatch: `sbatch slurm/optima_experiment.sbatch`

Start student cluster interactive session: `srun --account=deep_learning --pty bash`
Start optima experiment: `uv run optima/main.py --config optima/configs/student-optim_all-cifar.json`