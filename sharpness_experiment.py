import sys
from pathlib import Path
src_path = Path(__file__).resolve().parents[1] 
sys.path.append(str(src_path))

from src.sharpness.airbench94_muon import CifarNet, train, NormalizedMuonConfig, VanillaMuonConfig, SGDConfig, AdamConfig
import torch
from tqdm import tqdm
# import loss_landscapes
# import loss_landscapes.metrics as metrics
import torch.multiprocessing as mp
import numpy as np
import wandb
import time
import argparse

from src.sharpness import pyhessian_sharpness, sam_sharpness, samlike_sharpness
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


def train_and_log(experiment_name, model, optimizer_config, experiment_config: ExperimentConfig):
    logs = []
    run_dirs = []

    inputs, targets = experiment_config.metric_batch

    start = time.time()
    run = wandb.init(
        project=experiment_config.wandb_project,
        group=experiment_name,
        config=optimizer_config.represent(),
        reinit=True,
        mode="offline",
        # settings=wandb.Settings(
        #     _disable_stats=True,      # Disable system hardware metrics
        #     _disable_meta=True,       # Disable system metadata collection
        #     console="off"             # Disable console logging/wrapping
        # )
    )
    run_dirs.append(str(Path(run.dir).parent))
    # print(f"CUSTOM: Run `wandb sync {str(Path(run.dir).parent)}` to upload offline logs.")
    middle = time.time()
    
    def callback_fn(epoch, model, training_accuracy, validation_accuracy):
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
            "sam_sharpness": sam_sharpness(model, metric_batch),
            # "niko_sharpness": niko_sharpness(model, metric_batch)
        }

        if epoch > 0:
            log["hessian"], log["relative_hessian"] = pyhessian_sharpness(model, metric_batch)
            # print(f"Hessian Top Eigenvalue: {log['hessian']}, Relative Hessian: {log['relative_hessian']}")
        
        run.log(log)
        logs.append(log)

        # print(f"[{name}] Epoch {epoch}  SAM Sharpness: {sam_sharp}  Hessian Top Eigenvalue: {hess_sharp}")
        model.train()

    final_acc = train(
        "run",
        model,
        experiment_config=experiment_config,
        optimizer_config=optimizer_config,
        callback=callback_fn,
        epochs=experiment_config.epochs_per_run,
    )

    post = time.time()
    run.log({"tta_val_accuracy": final_acc, "tta_gap": logs[-1]["val_acc"] - final_acc})
    run.finish()
    end = time.time()

    wasted_time = (middle - start) + (end - post)
    
    print(f"Wasted Time: {wasted_time:.2f}s of {end - start:.2f}s total ({100.0 * wasted_time / (end - start):.2f}%)")

    return final_acc, logs, run_dirs


def worker(experiment_name, gpu_id, runs_per_gpu, optimizer_config, experiment_config: ExperimentConfig):
    all_acc, all_logs, all_run_dirs = [], [], []
    torch.cuda.set_device(gpu_id)
    model = CifarNet().to(f'cuda:{gpu_id}').to(memory_format=torch.channels_last)
    # model = torch.compile(model, mode="max-autotune-no-cudagraphs")

    for run in tqdm(range(runs_per_gpu)) if gpu_id == 0 else range(runs_per_gpu):
        acc, logs, run_dirs = train_and_log(experiment_name, model, optimizer_config, experiment_config)
        all_acc.append(acc)
        all_logs.append(logs)
        all_run_dirs.extend(run_dirs)
        
    return all_acc, all_logs, all_run_dirs


def train_distributed(experiment_name, gpus, runs_per_gpu, optimizer_config, experiment_config: ExperimentConfig):

    if gpus == 1:
        return worker(experiment_name, 0, runs_per_gpu, optimizer_config, experiment_config)

    ctx = mp.get_context('spawn')
    with ctx.Pool(gpus) as pool:
        out = [
            pool.apply_async(
                worker,
                args=(experiment_name, gpu_id, runs_per_gpu, optimizer_config, experiment_config),
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

    experiment_config = load_experiment_config(args.config)

    gpus = experiment_config.number_gpus
    runs_per_gpu = experiment_config.runs_per_gpu
    experiment_name = experiment_config.experiment_id
    # experiment_name = "hessian-every-epoch"

    print("Performing Warmup...")
    train_distributed("warmup", gpus, 1, NormalizedMuonConfig(), experiment_config)

    run_dirs = []

    print("Running Normalized Muon Experiments...")
    norm_muon_accs, norm_muon_logs, dirs = train_distributed(experiment_name, gpus, runs_per_gpu, NormalizedMuonConfig(), experiment_config)
    run_dirs.extend(dirs)

    print("Running Vanilla Muon Experiments...")
    vanilla_muon_accs, vanilla_muon_logs, dirs = train_distributed(experiment_name, gpus, runs_per_gpu, VanillaMuonConfig(), experiment_config)
    run_dirs.extend(dirs)

    print("Running SGD Experiments...")
    sgd_accs, sgd_logs, dirs = train_distributed(experiment_name, gpus, runs_per_gpu, SGDConfig(), experiment_config)
    run_dirs.extend(dirs)

    # print("Running Adam Experiments...")
    adam_accs, adam_logs, dirs = train_distributed(experiment_name, gpus, runs_per_gpu, AdamConfig(), experiment_config)
    run_dirs.extend(dirs)

    print_aggregated_metrics("Normalized Muon", norm_muon_accs, norm_muon_logs)
    print_aggregated_metrics("Vanilla Muon", vanilla_muon_accs, vanilla_muon_logs)
    print_aggregated_metrics("SGD", sgd_accs, sgd_logs)
    print_aggregated_metrics("Adam", adam_accs, adam_logs)

    print(f"See https://wandb.ai/padlex/cifar10-airbench -> {experiment_name} for detailed results.")
    # print("Run `wandb sync --sync-all` to upload offline logs.")

    print("Syncing all runs to wandb in parallel...")
    sync_all_runs_parallel(run_dirs, 64)

if __name__ == "__main__":
    main()
