import json
from dataclasses import dataclass
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
    checkpoint_enabled: bool = False
    checkpoint_dir: Path | None = None

    def represent(self) -> dict[str, Any]:
        """Return a JSON-serializable dict suitable for W&B config."""

        out: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "wandb_project": self.wandb_project,
            "wandb_mode": self.wandb_mode,
            "number_gpus": self.number_gpus,
            "runs_per_gpu": self.runs_per_gpu,
            "metric_dataloader": self.metric_dataloader,
            "metric_batch_size": self.metric_batch_size,
            "hessian_num_batches": self.hessian_num_batches,
            "metrics": list(self.metrics),
            "epochs_per_run": self.epochs_per_run,
            "dataset_path": str(self.dataset_path),
            "checkpoint_enabled": bool(self.checkpoint_enabled),
            "checkpoint_dir": str(self.checkpoint_dir) if self.checkpoint_dir is not None else None,
        }
        if self.adaptive_sharpness is not None:
            out["adaptive_sharpness"] = {
                "rho": self.adaptive_sharpness.rho,
                "eta": self.adaptive_sharpness.eta,
                "ascent_steps": self.adaptive_sharpness.ascent_steps,
                "ascent_lr": self.adaptive_sharpness.ascent_lr,
                "use_eval_mode": self.adaptive_sharpness.use_eval_mode,
            }
        return out

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

    checkpointing = data.get("checkpointing", {})
    checkpoint_enabled = bool(checkpointing.get("enabled", False))
    checkpoint_dir_raw = checkpointing.get("dir")
    if checkpoint_enabled:
        if checkpoint_dir_raw in (None, ""):
            raise ValueError("checkpointing.dir must be set when checkpointing.enabled is true")
        checkpoint_dir = project_root / Path(checkpoint_dir_raw)
    else:
        checkpoint_dir = None

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
        checkpoint_enabled=checkpoint_enabled,
        checkpoint_dir=checkpoint_dir,
    )