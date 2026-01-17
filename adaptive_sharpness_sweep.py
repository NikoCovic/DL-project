import argparse
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Allow running as a script without installing the package.
sys.path.append(str(Path(__file__).resolve().parent))

from src.sharpness.airbench94_muon import CifarNet, CIFAR_MEAN, CIFAR_STD
from src.sharpness import adaptive_sharpness


@dataclass(frozen=True)
class SweepConfig:
    checkpoint_dir: Path
    dataset_path: Path
    split: str
    batch_size: int
    device: str
    params: dict
    output_path: Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_grid(params: dict) -> list[dict]:
    """Turn {k: value-or-list} into a list of all combinations."""
    keys = list(params.keys())
    values = [v if isinstance(v, list) else [v] for v in params.values()]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def find_model_files(checkpoint_dir: Path) -> list[Path]:
    return sorted(checkpoint_dir.rglob("model.pt"))


def load_all_checkpoints(model_files: list[Path]) -> list[dict]:
    ckpts = []
    for p in tqdm(model_files, desc="loading checkpoints"):
        ckpts.append({"path": str(p), "data": torch.load(p, map_location="cpu")})
    return ckpts


def load_all_models(checkpoints: list[dict], device: str) -> list[dict]:
    models = []
    for ckpt in tqdm(checkpoints, desc="loading models"):
        model = CifarNet().to(device).to(memory_format=torch.channels_last)
        model.load_state_dict(ckpt["data"]["model_state_dict"], strict=True)
        models.append({"path": ckpt["path"], "model": model})
    return models


def load_cifar_tensors(dataset_path: Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    def load_one(name: str):
        d = torch.load(dataset_path / f"{name}.pt", map_location="cpu")
        return d["images"], d["labels"]

    if split == "train":
        images, labels = load_one("train")
    elif split == "test":
        images, labels = load_one("test")
    elif split == "both":
        train_images, train_labels = load_one("train")
        test_images, test_labels = load_one("test")
        images = torch.cat([train_images, test_images], dim=0)
        labels = torch.cat([train_labels, test_labels], dim=0)
    else:
        raise ValueError(f"split must be one of: train/test/both (got {split!r})")

    # uint8 (N,H,W,C) -> float16 (N,C,H,W), normalized
    x = images.to(torch.float16).div_(255).permute(0, 3, 1, 2).contiguous()
    mean = CIFAR_MEAN.view(1, 3, 1, 1).to(dtype=x.dtype)
    std = CIFAR_STD.view(1, 3, 1, 1).to(dtype=x.dtype)
    x = (x - mean) / std
    y = labels.to(torch.long)
    return x, y


def adaptive_sharpness_over_dataset(

    model: torch.nn.Module,
    dataloader: DataLoader,
    adaptive_cfg,
    device: str,
) -> float:
    total = 0.0
    count = 0
    for x, y in tqdm(dataloader, desc="dataset", leave=False):
        x = x.to(device)
        y = y.to(device)
        val = adaptive_sharpness(model, F.cross_entropy, (x, y), adaptive_cfg)
        total += float(val) * int(x.shape[0])
        count += int(x.shape[0])
    return total / count


def parse_config(raw: dict) -> SweepConfig:
    checkpoint_dir = Path(raw["checkpoint_dir"])
    dataset_path = Path(raw.get("dataset_path", "data/cifar10"))
    split = str(raw.get("split", "train"))
    batch_size = int(raw.get("batch_size", 512))
    device = str(raw.get("device", "cuda"))
    output_path = Path(raw.get("output_path", "adaptive_sharpness_sweep_results.jsonl"))

    params = raw.get("parameters")
    if params is None:
        params = raw.get("adaptive_sharpness")
    if params is None:
        raise ValueError("Config must contain either 'parameters' or 'adaptive_sharpness'")

    return SweepConfig(
        checkpoint_dir=checkpoint_dir,
        dataset_path=dataset_path,
        split=split,
        batch_size=batch_size,
        device=device,
        params=dict(params),
        output_path=output_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Adaptive sharpness grid sweep over saved checkpoints")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    args = parser.parse_args()

    cfg = parse_config(load_json(Path(args.config)))

    model_files = find_model_files(cfg.checkpoint_dir)
    if not model_files:
        raise FileNotFoundError(f"No model.pt found under {cfg.checkpoint_dir}")

    checkpoints = load_all_checkpoints(model_files)
    models = load_all_models(checkpoints, cfg.device)

    x, y = load_cifar_tensors(cfg.dataset_path, cfg.split)
    dataloader = DataLoader(
        TensorDataset(x, y),
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    combos = as_grid(cfg.params)
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    with cfg.output_path.open("w", encoding="utf-8") as f:
        for combo in tqdm(combos, desc="hyperparams"):
            # Just a tiny object with the fields adaptive_sharpness() expects.
            adaptive_cfg = type("AdaptiveCfg", (), combo)

            per_model = []
            for m in tqdm(models, desc="models", leave=False):
                val = adaptive_sharpness_over_dataset(m["model"], dataloader, adaptive_cfg, cfg.device)
                per_model.append(val)

                row = {"checkpoint": m["path"], "adaptive_sharpness": val, **combo}
                f.write(json.dumps(row) + "\n")
                f.flush()

            mean_val = sum(per_model) / len(per_model)
            print({"mean_adaptive_sharpness": mean_val, **combo})


if __name__ == "__main__":
    main()
