import json
from dataclasses import dataclass
from pathlib import Path
import uuid
from typing import Any
from airbench94_muon import CifarLoader

@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    wandb_project: str
    number_gpus: int
    runs_per_gpu: int
    metric_batch_size: int
    hessian_num_batches: int
    metric_dataloader_object: Any



def load_experiment_config(path: str | Path) -> ExperimentConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    experiment_id_raw = data.get("experiment_id", "auto")
    experiment_id = str(uuid.uuid4())[:8] if experiment_id_raw in (None, "auto", "") else str(experiment_id_raw)
    
    wandb_project_raw = data.get("wandb_project", "auto")
    wandb_project = str(uuid.uuid4())[:8] if wandb_project_raw in (None, "auto", "") else str(wandb_project_raw)

    number_gpus = int(data.get("number_gpus"))
    runs_per_gpu = int(data.get("runs_per_gpu"))
    metric_batch_size = int(data.get("metric_batch_size"))
    hessian_num_batches = int(data.get("hessian_num_batches"))

    if str(data.get("metric_dataloader")) == "AirBenchCifarLoader":
        metric_dataloader_object = CifarLoader(
            'dataloaders/cifar10',
            train=True,
            batch_size=metric_batch_size,
        )

    if number_gpus <= 0:
        raise ValueError("config.number_gpus must be > 0")
    if runs_per_gpu <= 0:
        raise ValueError("config.runs_per_gpu must be > 0")
    if metric_batch_size <= 0:
        raise ValueError("config.metric_batch_size must be > 0")
    if hessian_num_batches <= 0:
        raise ValueError("config.hessian_num_batches must be > 0")

    return ExperimentConfig(
        experiment_id=experiment_id,
        number_gpus=number_gpus,
        runs_per_gpu=runs_per_gpu,
        metric_batch_size=metric_batch_size,
        hessian_num_batches=hessian_num_batches,
        metric_dataloader_object=metric_dataloader_object,
        wandb_project=wandb_project,
    )