import sys
from pathlib import Path
src_path = Path(__file__).resolve().parents[1] 
sys.path.append(str(src_path))

from src.sharpness.airbench94_muon import CifarNet, train, NormalizedMuonConfig, VanillaMuonConfig, SGDConfig, AdamConfig
import torch
import torch.nn.functional as F
from tqdm import tqdm
# import loss_landscapes
# import loss_landscapes.metrics as metrics
import torch.multiprocessing as mp
import numpy as np
import wandb
import time
import argparse
from shutil import copy2

from src.sharpness import pyhessian_sharpness, sam_sharpness, samlike_sharpness, adaptive_sharpness
from src.sharpness.config import load_experiment_config, ExperimentConfig
from src.sharpness.wandb_sync import sync_all_runs_parallel

# from src.edge_of_stability.hessian import Hessian

# def niko_sharpness(model, data_batch):
#     criterion = torch.nn.CrossEntropyLoss()
#     device = next(model.parameters()).device
#     hessian = Hessian(model, data_batch, criterion, device=device)
#     hessian.update_params()
#     _, es = hessian.eigenvalues()
#     e = es[0]
#     print(f"Niko Sharpness (Top Hessian Eigenvalue): {e}")
#     return e


def train_and_log(
    experiment_name,
    model,
    optimizer_config,
    experiment_config: ExperimentConfig,
    run_number: int,
    config_file_path: Path,
):
    logs = []
    run_dirs = []

    inputs, targets = experiment_config.metric_batch

    start = time.time()

    wandb_config = optimizer_config.represent()
    wandb_config.update(experiment_config.represent())
    wandb_config["run_number"] = int(run_number)

    run = wandb.init(
        project=experiment_config.wandb_project,
        group=experiment_name,
        config=wandb_config,
        reinit=True,
        mode=experiment_config.wandb_mode,
        # settings=wandb.Settings(
        #     _disable_stats=True,      # Disable system hardware metrics
        #     _disable_meta=True,       # Disable system metadata collection
        #     console="off"             # Disable console logging/wrapping
        # )
    )
    run_dirs.append(str(Path(run.dir).parent))  # Store parent dir for later wandb sync if in offline mode
    middle = time.time()
    
    def log_epoch_metrics(epoch, model, training_accuracy, validation_accuracy):
        model.eval()

        device = next(model.parameters()).device
        metric_batch = (inputs.to(device), targets.to(device))

        # Free any stale cached blocks before expensive higher-order/autograd workloads.
        # torch.cuda.empty_cache()

        log = {
            "epoch": epoch,
            "train_acc": training_accuracy,
            "val_acc": validation_accuracy,
            "gap": training_accuracy - validation_accuracy,
        }

        if epoch > 0:
            if "pyhessian_sharpness" in experiment_config.metrics:
                log["hessian"], log["relative_hessian"] = pyhessian_sharpness(model, metric_batch)
            
            if "adaptive_sharpness" in experiment_config.metrics:
                log["adaptive_sharpness"] = adaptive_sharpness(
                    model,
                    F.cross_entropy,
                    metric_batch,
                    experiment_config.adaptive_sharpness,
                )
            
            if "sam_sharpness" in experiment_config.metrics:
                log["sam_sharpness"] = sam_sharpness(model, metric_batch)

            # if "niko_sharpness" in experiment_config.metrics:
            #     log["niko_sharpness"] = niko_sharpness(model, metric_batch)
            
        
        run.log(log)
        logs.append(log)

        # print(f"[{name}] Epoch {epoch}  SAM Sharpness: {sam_sharp}  Hessian Top Eigenvalue: {hess_sharp}")
        model.train()

    final_acc = train(
        "run",
        model,
        experiment_config=experiment_config,
        optimizer_config=optimizer_config,
        log_epoch_metrics_callback=log_epoch_metrics,
        epochs=experiment_config.epochs_per_run,
    )

    post = time.time()
    run.log({"tta_val_accuracy": final_acc, "tta_gap": logs[-1]["val_acc"] - final_acc})

    if experiment_config.checkpoint_enabled:
        # Save final model + configs into: checkpoint_dir/experiment_id/optimizer_name-run_number
        if experiment_config.checkpoint_dir is None:
            raise ValueError("checkpoint_dir is None but checkpointing is enabled")
        ckpt_base = Path(experiment_config.checkpoint_dir) / str(experiment_config.experiment_id)
        gpu_id = torch.cuda.current_device()
        ckpt_run_dir = ckpt_base / f"{optimizer_config.name}-{int(run_number)}-{gpu_id}"
        ckpt_run_dir.mkdir(parents=True, exist_ok=True)

        # Model weights (CPU state_dict for portability)
        state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(
            {
                "model_state_dict": state_dict_cpu,
                "final_acc": float(final_acc),
                "optimizer": optimizer_config.represent(),
                "experiment_id": str(experiment_config.experiment_id),
            },
            ckpt_run_dir / "model.pt",
        )

        # Save the exact config file that was passed to the program.
        copy2(config_file_path, ckpt_run_dir / "config.json")

    run.finish()
    end = time.time()

    wasted_time = (middle - start) + (end - post)
    
    print(f"Wasted Time: {wasted_time:.2f}s of {end - start:.2f}s total ({100.0 * wasted_time / (end - start):.2f}%)")

    return final_acc, logs, run_dirs


def worker(
    experiment_name,
    gpu_id,
    runs_per_gpu,
    optimizer_config,
    experiment_config: ExperimentConfig,
    config_file_path: Path,
):
    all_acc, all_logs, all_run_dirs = [], [], []
    torch.cuda.set_device(gpu_id)
    model = CifarNet().to(f'cuda:{gpu_id}').to(memory_format=torch.channels_last)
    # model = torch.compile(model, mode="max-autotune-no-cudagraphs")

    for run in tqdm(range(runs_per_gpu)) if gpu_id == 0 else range(runs_per_gpu):
        # 1-indexed run number (matches "(1-5 currently)" naming)
        run_number = int(run) + 1
        acc, logs, run_dirs = train_and_log(
            experiment_name,
            model,
            optimizer_config,
            experiment_config,
            run_number,
            config_file_path,
        )
        all_acc.append(acc)
        all_logs.append(logs)
        all_run_dirs.extend(run_dirs)
        
    return all_acc, all_logs, all_run_dirs


def train_distributed(
    experiment_name,
    gpus,
    runs_per_gpu,
    optimizer_config,
    experiment_config: ExperimentConfig,
    config_file_path: Path,
):

    if gpus == 1:
        return worker(experiment_name, 0, runs_per_gpu, optimizer_config, experiment_config, config_file_path)

    ctx = mp.get_context('spawn')
    with ctx.Pool(gpus) as pool:
        out = [
            pool.apply_async(
                worker,
                args=(experiment_name, gpu_id, runs_per_gpu, optimizer_config, experiment_config, config_file_path),
            )
            for gpu_id in range(gpus)
        ]
        results = [p.get() for p in out]

        pool.close()
        pool.join()
    
    all_accs, all_logs, all_run_dirs = [], [], []
    for accs, logs_list, run_dirs in results:
        all_accs.extend(accs)
        all_logs.extend(logs_list)
        all_run_dirs.extend(run_dirs)

    return all_accs, all_logs, all_run_dirs


def print_aggregated_metrics(name, all_accs, all_logs, metrics=None, epochs=None):
    print(f"=== {name} Results ===")
    print(f"Final Accuracy: {sum(all_accs) / len(all_accs):.4f} ± {np.std(all_accs):.4f}\n")

    if not metrics:
        metrics = list(all_logs[0][0].keys())
    
    if not epochs:
        epochs = range(len(all_logs[0]))

    for epoch in epochs:
        for metric in metrics:
            vals = [log[epoch][metric] for log in all_logs if metric in log[epoch]]
            if vals:
                mean_val = sum(vals) / len(vals)
                std_val = np.std(vals)
                print(f"{metric} @ epoch {epoch}: {mean_val:.4f} ± {std_val:.4f}")
            else:
                print(f"{metric} @ epoch {epoch}: [No Data]")
        print()
    print("\n")


def main():
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Run CIFAR10 experiment with sharpness/Hessian metrics")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment JSON config (e.g. experiment-configs/stclstr-quick.json)",
    )
    args = parser.parse_args()

    config_file_path = Path(args.config).resolve()
    experiment_config = load_experiment_config(config_file_path)

    gpus = experiment_config.number_gpus
    runs_per_gpu = experiment_config.runs_per_gpu
    experiment_name = experiment_config.experiment_id
    # experiment_name = "hessian-every-epoch"

    # print("Performing Warmup...")
    # train_distributed("warmup", gpus, 1, NormalizedMuonConfig(), experiment_config)

    run_dirs = []

    print("Running Normalized Muon Experiments...")
    norm_muon_accs, norm_muon_logs, dirs = train_distributed(experiment_name, gpus, runs_per_gpu, NormalizedMuonConfig(), experiment_config, config_file_path)
    run_dirs.extend(dirs)

    print("Running Vanilla Muon Experiments...")
    vanilla_muon_accs, vanilla_muon_logs, dirs = train_distributed(experiment_name, gpus, runs_per_gpu, VanillaMuonConfig(), experiment_config, config_file_path)
    run_dirs.extend(dirs)

    print("Running SGD Experiments...")
    sgd_accs, sgd_logs, dirs = train_distributed(experiment_name, gpus, runs_per_gpu, SGDConfig(), experiment_config, config_file_path)
    run_dirs.extend(dirs)

    # print("Running Adam Experiments...")
    adam_accs, adam_logs, dirs = train_distributed(experiment_name, gpus, runs_per_gpu, AdamConfig(), experiment_config, config_file_path)
    run_dirs.extend(dirs)

    print_aggregated_metrics("Normalized Muon", norm_muon_accs, norm_muon_logs)
    print_aggregated_metrics("Vanilla Muon", vanilla_muon_accs, vanilla_muon_logs)
    print_aggregated_metrics("SGD", sgd_accs, sgd_logs)
    print_aggregated_metrics("Adam", adam_accs, adam_logs)

    print(f"See https://wandb.ai/padlex/cifar10-airbench -> {experiment_name} for detailed results.")
    # print("Run `wandb sync --sync-all` to upload offline logs.")

    print("Syncing all runs to wandb in parallel...")
    if experiment_config.wandb_mode == "offline":
        sync_all_runs_parallel(run_dirs, 64)

if __name__ == "__main__":
    main()
