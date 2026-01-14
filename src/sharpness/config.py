import json
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
from typing import Any
from .airbench94_muon import CifarLoader


@dataclass(frozen=True)
class AdaptiveSharpnessConfig:
    rho: float
    eta: float
    ascent_steps: int
    ascent_lr: float
    use_eval_mode: bool

@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    wandb_project: str
    wandb_mode: str
    number_gpus: int
    runs_per_gpu: int
    metric_batch: Any
    metric_dataloader: str
    metric_batch_size: int
    hessian_num_batches: int
    metrics: list[str]
    epochs_per_run: int
    dataset_path: Path
    adaptive_sharpness: AdaptiveSharpnessConfig | None = None

    def represent(self) -> dict[str, Any]:
        """Converts config to a flat dict suitable for W&B."""
        
        def _flatten(obj, prefix=""):
            result = {}
            items = asdict(obj).items() if hasattr(obj, "__dataclass_fields__") else obj.items()
            
            for key, value in items:
                new_key = f"{prefix}{key}"
                
                if hasattr(value, "__dataclass_fields__"):
                    # Recurse into nested dataclasses
                    result.update(_flatten(value, prefix=f"{new_key}."))
                elif isinstance(value, Path):
                    result[new_key] = str(value)
                else:
                    result[new_key] = value
            return result

        return _flatten(self)

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
    metrics = list(data.get("metrics", []))
    epochs_per_run = int(data.get("epochs_per_run"))
    wandb_mode = str(data["wandb_mode"])

    project_root = Path(__file__).resolve().parents[2]
    dataset_path = project_root / Path(data.get("dataset_path"))

    metric_dataloader = str(data.get("metric_dataloader"))
    if metric_dataloader == "AirBenchCifarLoader":
        metric_batch = next(iter(CifarLoader(
            str(dataset_path),
            train=True,
            batch_size=metric_batch_size,
        )))
    else:
        raise ValueError(f"Unknown metric_dataloader: {data.get('metric_dataloader')!r}")

    if number_gpus <= 0:
        raise ValueError("config.number_gpus must be > 0")
    if runs_per_gpu <= 0:
        raise ValueError("config.runs_per_gpu must be > 0")
    if metric_batch_size <= 0:
        raise ValueError("config.metric_batch_size must be > 0")
    if hessian_num_batches <= 0:
        raise ValueError("config.hessian_num_batches must be > 0")
    if epochs_per_run <= 0:
        raise ValueError("config.epochs_per_run must be > 0")

    adaptive_sharpness = None
    if "adaptive_sharpness" in data:
        adaptive_raw = data["adaptive_sharpness"]
        adaptive_sharpness = AdaptiveSharpnessConfig(
            rho=float(adaptive_raw["rho"]),
            eta=float(adaptive_raw["eta"]),
            ascent_steps=int(adaptive_raw["ascent_steps"]),
            ascent_lr=float(adaptive_raw["ascent_lr"]),
            use_eval_mode=bool(adaptive_raw["use_eval_mode"]),
        )

    return ExperimentConfig(
        experiment_id=experiment_id,
        number_gpus=number_gpus,
        runs_per_gpu=runs_per_gpu,
        metric_batch=metric_batch,
        metric_dataloader=metric_dataloader,
        metric_batch_size=metric_batch_size,
        hessian_num_batches=hessian_num_batches,
        metrics=metrics,
        wandb_project=wandb_project,
        epochs_per_run=epochs_per_run,
        dataset_path=dataset_path,
        adaptive_sharpness=adaptive_sharpness,
        wandb_mode=wandb_mode,
    )