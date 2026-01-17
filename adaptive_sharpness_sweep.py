import argparse
import json
import math
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, pearsonr
from tqdm.auto import tqdm

# Allow running as a script without installing the package.
sys.path.append(str(Path(__file__).resolve().parent))

from src.sharpness.airbench94_muon import CifarNet, CIFAR_MEAN, CIFAR_STD
from src.sharpness import adaptive_sharpness
from src.sharpness.config import AdaptiveSharpnessConfig


@dataclass(frozen=True)
class SweepConfig:
    checkpoint_dir: Path
    dataset_path: Path
    split: str
    batch_size: int
    device: str
    subset_batches: int | None
    params: dict
    output_path: Path


@dataclass(frozen=True)
class RunRow:
    optimizer: str
    rho: float
    loss_gap: float
    acc_gap: float
    train_loss: float
    test_loss: float
    train_acc: float
    test_acc: float
    sharpness: float
    path: str


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
    # model_files = model_files[:3]
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


def iter_batches(x: torch.Tensor, y: torch.Tensor, batch_size: int):
    for i in range(0, x.shape[0], batch_size):
        yield x[i : i + batch_size], y[i : i + batch_size]


def adaptive_sharpness_on_batches(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    adaptive_cfg: AdaptiveSharpnessConfig,
    device: str,
    batch_size: int,
) -> float:
    total = 0.0
    count = 0
    for xb, yb in iter_batches(x, y, batch_size):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        val = adaptive_sharpness(model, F.cross_entropy, (xb, yb), adaptive_cfg)
        bs = xb.shape[0]
        total += float(val) * bs
        count += bs
    return total / max(count, 1)


def stats_on_batches(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: str,
    batch_size: int,
) -> dict:
    """Return mean loss and accuracy over batches."""
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total = 0

    with torch.no_grad():
        for xb, yb in iter_batches(x, y, batch_size):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            pred = logits.argmax(dim=1)
            acc = (pred == yb).float().mean().item()
            bs = xb.shape[0]
            total_loss += float(loss.item()) * bs
            total_acc += float(acc) * bs
            total += bs

    if was_training:
        model.train()

    denom = max(total, 1)
    return {
        "loss": total_loss / denom,
        "acc": total_acc / denom,
    }


def train_test_gaps(
    model: torch.nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    device: str,
    batch_size: int,
) -> dict:
    train_stats = stats_on_batches(model, x_train, y_train, device, batch_size)
    test_stats = stats_on_batches(model, x_test, y_test, device, batch_size)
    return {
        "train_loss": float(train_stats["loss"]),
        "test_loss": float(test_stats["loss"]),
        "train_test_loss_gap": float(train_stats["loss"] - test_stats["loss"]),
        "train_acc": float(train_stats["acc"]),
        "test_acc": float(test_stats["acc"]),
        "train_test_acc_gap": float(train_stats["acc"] - test_stats["acc"]),
    }


def parse_config(raw: dict) -> SweepConfig:
    checkpoint_dir = Path(raw["checkpoint_dir"])
    dataset_path = Path(raw.get("dataset_path", "data/cifar10"))
    split = str(raw.get("split", "train"))
    batch_size = int(raw.get("batch_size", 512))
    device = str(raw.get("device", "cuda"))
    subset_batches = raw.get("subset_batches")
    if subset_batches is not None:
        subset_batches = int(subset_batches)
        if subset_batches <= 0:
            raise ValueError("subset_batches must be a positive integer")
    output_path = Path(raw.get("output_path", "adaptive_sharpness_sweep_results.csv"))

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
        subset_batches=subset_batches,
        params=dict(params),
        output_path=output_path,
    )


def maybe_subset_batches(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    subset_batches: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if subset_batches is None:
        return x, y
    take = min(x.shape[0], subset_batches * batch_size)
    return x[:take], y[:take]


def kendall_tau_b(x: list[float], y: list[float]) -> float:
    if len(x) != len(y):
        raise ValueError("kendall_tau_b requires lists of the same length")
    if len(x) < 2:
        return float("nan")
    tau, _ = kendalltau(x, y, variant="b")
    return float(tau) if tau is not None else float("nan")


def pearson_r(x: list[float], y: list[float]) -> float:
    if len(x) != len(y):
        raise ValueError("pearson_r requires lists of the same length")
    if len(x) < 2:
        return float("nan")
    r, _ = pearsonr(x, y)
    return float(r) if r is not None else float("nan")


def optimizer_from_path(path_str: str) -> str:
    name = Path(path_str).parent.name
    parts = name.split("-")
    if len(parts) >= 2 and parts[-1].isdigit():
        base = "-".join(parts[:-1])
        return base if base else name
    return name


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

    # Load full CIFAR splits onto the GPU once (single-batch evaluation).
    x_train, y_train = load_cifar_tensors(cfg.dataset_path, "train")
    x_test, y_test = load_cifar_tensors(cfg.dataset_path, "test")
    x_sweep, y_sweep = load_cifar_tensors(cfg.dataset_path, cfg.split)

    x_train, y_train = maybe_subset_batches(
        x_train, y_train, cfg.batch_size, cfg.subset_batches
    )
    x_test, y_test = maybe_subset_batches(
        x_test, y_test, cfg.batch_size, cfg.subset_batches
    )
    x_sweep, y_sweep = maybe_subset_batches(
        x_sweep, y_sweep, cfg.batch_size, cfg.subset_batches
    )

    combos = as_grid(cfg.params)
    if "rho" not in cfg.params:
        raise ValueError("Config parameters must include 'rho' for correlation reporting")

    rho_values = cfg.params["rho"]
    if not isinstance(rho_values, list):
        rho_values = [rho_values]
    rho_values = sorted(set(rho_values))

    rows: list[RunRow] = []

    for m in tqdm(models, desc="models"):
        print(f"model: {m['path']}")
        optimizer_name = optimizer_from_path(m["path"])
        gap = train_test_gaps(
            m["model"],
            x_train,
            y_train,
            x_test,
            y_test,
            cfg.device,
            cfg.batch_size,
        )
        loss_gap_value = float(gap["train_test_loss_gap"])
        acc_gap_value = float(gap["train_test_acc_gap"])
        train_loss_value = float(gap["train_loss"])
        test_loss_value = float(gap["test_loss"])
        train_acc_value = float(gap["train_acc"])
        test_acc_value = float(gap["test_acc"])

        rho_sums = {float(r): 0.0 for r in rho_values}
        rho_counts = {float(r): 0 for r in rho_values}

        for combo in tqdm(combos, desc="hyperparams", leave=False):
            if "ascent_steps" not in combo:
                raise ValueError("Config parameters must include 'ascent_steps' to derive ascent_lr")
            rho = float(combo["rho"])
            ascent_steps = int(combo["ascent_steps"])
            if ascent_steps <= 0:
                raise ValueError("ascent_steps must be a positive integer")
            ascent_lr = rho / ascent_steps
            combo_with_lr = {**combo, "ascent_lr": ascent_lr}

            adaptive_cfg = AdaptiveSharpnessConfig(**combo_with_lr)
            val = adaptive_sharpness_on_batches(
                m["model"],
                x_sweep,
                y_sweep,
                adaptive_cfg,
                cfg.device,
                cfg.batch_size,
            )

            rho_sums[rho] += float(val)
            rho_counts[rho] += 1

        for rho in rho_values:
            rho = float(rho)
            if rho_counts[rho] == 0:
                continue
            avg_sharp = rho_sums[rho] / rho_counts[rho]
            rows.append(
                RunRow(
                    optimizer=optimizer_name,
                    rho=float(rho),
                    loss_gap=loss_gap_value,
                    acc_gap=acc_gap_value,
                    train_loss=train_loss_value,
                    test_loss=test_loss_value,
                    train_acc=train_acc_value,
                    test_acc=test_acc_value,
                    sharpness=float(avg_sharp),
                    path=str(m["path"]),
                )
            )

    best_rho = None
    best_tau = float("nan")
    best_abs_tau = float("-inf")
    print(
        "Note: LossGap = train_loss - test_loss and AccGap = train_acc - test_acc. "
        "Because higher accuracy usually means lower loss, AccGap often moves opposite to LossGap. "
        "If LossGap is negative (train loss < test loss), sharper models implying worse generalization "
        "will show NEGATIVE correlation with LossGap but POSITIVE correlation with AccGap."
    )
    for rho in rho_values:
        rho = float(rho)
        rho_rows = [row for row in rows if row.rho == rho]
        loss_gaps = [row.loss_gap for row in rho_rows]
        acc_gaps = [row.acc_gap for row in rho_rows]
        sharps = [row.sharpness for row in rho_rows]
        tau_loss = kendall_tau_b(loss_gaps, sharps)
        tau_acc = kendall_tau_b(acc_gaps, sharps)
        pr_loss = pearson_r(loss_gaps, sharps)
        pr_acc = pearson_r(acc_gaps, sharps)
        print(
            f"rho={rho}: kendall_tau(loss)={tau_loss}, kendall_tau(acc)={tau_acc}, "
            f"pearson_r(loss)={pr_loss}, pearson_r(acc)={pr_acc}"
        )

        header = (
            f"{'Optimizer':<16} {'Row':<12} {'Kendall_tau':>12} "
            f"{'Pearson_r':>12} {'TrainLoss':>12} {'TestLoss':>12} "
            f"{'TrainAcc':>10} {'TestAcc':>10} {'LossGap':>12} "
            f"{'AccGap':>10} {'Sharpness':>12} Model"
        )
        print(header)
        for opt_name in sorted({row.optimizer for row in rho_rows}):
            opt_rows = [row for row in rho_rows if row.optimizer == opt_name]
            for row in opt_rows:
                print(
                    f"{opt_name:<16} {'run':<12} {'':>12} {'':>12} "
                    f"{row.train_loss:>12.6f} {row.test_loss:>12.6f} "
                    f"{row.train_acc:>10.6f} {row.test_acc:>10.6f} "
                    f"{row.loss_gap:>12.6f} {row.acc_gap:>10.6f} "
                    f"{row.sharpness:>12.6f} {row.path}"
                )

            opt_loss_gaps = [row.loss_gap for row in opt_rows]
            opt_acc_gaps = [row.acc_gap for row in opt_rows]
            opt_sharps = [row.sharpness for row in opt_rows]
            opt_tau_loss = kendall_tau_b(opt_loss_gaps, opt_sharps)
            opt_pr_loss = pearson_r(opt_loss_gaps, opt_sharps)
            opt_tau_acc = kendall_tau_b(opt_acc_gaps, opt_sharps)
            opt_pr_acc = pearson_r(opt_acc_gaps, opt_sharps)
            print(
                f"{opt_name:<16} {'kendall_tau_loss':<16} {opt_tau_loss:>12.6f} "
                f"{'':>12} {'':>12} {'':>12}"
            )
            print(
                f"{opt_name:<16} {'pearson_r_loss':<16} {'':>12} "
                f"{opt_pr_loss:>12.6f} {'':>12} {'':>12}"
            )
            print(
                f"{opt_name:<16} {'kendall_tau_acc':<16} {opt_tau_acc:>12.6f} "
                f"{'':>12} {'':>12} {'':>12}"
            )
            print(
                f"{opt_name:<16} {'pearson_r_acc':<16} {'':>12} "
                f"{opt_pr_acc:>12.6f} {'':>12} {'':>12}"
            )

        if not math.isnan(tau_loss) and abs(tau_loss) > best_abs_tau:
            best_abs_tau = abs(tau_loss)
            best_tau = tau_loss
            best_rho = rho

    if best_rho is None:
        print("best_rho: n/a (insufficient data)")
    else:
        print(f"best_rho={best_rho} (kendall_tau={best_tau})")
        header = (
            f"{'Optimizer':<16} {'Kendall_tau':>12} {'Pearson_r':>12} "
            f"{'Kendall_tau_acc':>16} {'Pearson_r_acc':>14}"
        )
        print(header)
        best_rows = [row for row in rows if row.rho == best_rho]
        for opt_name in sorted({row.optimizer for row in best_rows}):
            opt_rows = [row for row in best_rows if row.optimizer == opt_name]
            loss_gaps = [row.loss_gap for row in opt_rows]
            acc_gaps = [row.acc_gap for row in opt_rows]
            sharps = [row.sharpness for row in opt_rows]
            tau_loss = kendall_tau_b(loss_gaps, sharps)
            pr_loss = pearson_r(loss_gaps, sharps)
            tau_acc = kendall_tau_b(acc_gaps, sharps)
            pr_acc = pearson_r(acc_gaps, sharps)
            print(
                f"{opt_name:<16} {tau_loss:>12.6f} {pr_loss:>12.6f} "
                f"{tau_acc:>12.6f} {pr_acc:>12.6f}"
            )


if __name__ == "__main__":
    main()
