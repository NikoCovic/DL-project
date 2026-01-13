import sys
from pathlib import Path
src_path = Path(__file__).resolve().parents[1] 
sys.path.append(str(src_path))

from src.sharpness.airbench94_muon import CifarNet, train, MuonConfig, SGDConfig, AdamConfig
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

from src.edge_of_stability.hessian import Hessian

def niko_sharpness(model, data_batch):
    criterion = torch.nn.CrossEntropyLoss()
    device = next(model.parameters()).device
    hessian = Hessian(model, data_batch, criterion, device=device)
    es, _ = hessian.eigenvalues()
    e = es[0]
    return e


def train_and_log(experiment_name, model, optimizer_config, experiment_config: ExperimentConfig):
    logs = []

    inputs, targets = experiment_config.metric_batch
    
    def callback_fn(epoch, model, training_accuracy, validation_accuracy):
        model.eval()

        device = next(model.parameters()).device
        metric_batch = (inputs.to(device), targets.to(device))

        # Free any stale cached blocks before expensive higher-order/autograd workloads.
        torch.cuda.empty_cache()

        log = {
            "epoch": epoch,
            "train_acc": training_accuracy,
            "val_acc": validation_accuracy,
            "gap": training_accuracy - validation_accuracy,
            "sam_sharpness": sam_sharpness(model, metric_batch),
            # "niko_sharpness": niko_sharpness(model, metric_batch)
        }

        if epoch == 16:
            # print(f"Calculating Hessian for epoch {epoch}...")
            log["hessian"], log["relative_hessian"] = pyhessian_sharpness(model, metric_batch)
        
        wandb.log(log)
        logs.append(log)

        # print(f"[{name}] Epoch {epoch}  SAM Sharpness: {sam_sharp}  Hessian Top Eigenvalue: {hess_sharp}")
        model.train()
    
    start = time.time()
    wandb.init(project=experiment_config.wandb_project, group=experiment_name, config=optimizer_config.represent())
    middle = time.time()

    final_acc = train(
        "run",
        model,
        experiment_config=experiment_config,
        optimizer_config=optimizer_config,
        callback=callback_fn,
        epochs=experiment_config.epochs_per_run,
    )

    post = time.time()
    wandb.log({"tta_val_accuracy": final_acc, "tta_gap": logs[-1]["val_acc"] - final_acc})
    wandb.finish()
    end = time.time()

    wasted_time = (middle - start) + (end - post)
    
    print(f"Wasted Time: {wasted_time:.2f}s of {end - start:.2f}s total ({100.0 * wasted_time / (end - start):.2f}%)")

    return final_acc, logs


def worker(experiment_name, gpu_id, runs_per_gpu, optimizer_config, experiment_config: ExperimentConfig):
    all_acc, all_logs = [], []
    torch.cuda.set_device(gpu_id)
    model = CifarNet().to(f'cuda:{gpu_id}').to(memory_format=torch.channels_last)
    # model = torch.compile(model, mode="max-autotune-no-cudagraphs")

    for run in tqdm(range(runs_per_gpu)) if gpu_id == 0 else range(runs_per_gpu):
        acc, logs = train_and_log(experiment_name, model, optimizer_config, experiment_config)
        all_acc.append(acc)
        all_logs.append(logs)
        
    return all_acc, all_logs


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
    
    all_accs, all_logs = [], []
    for accs, logs_list in results:
        all_accs.extend(accs)
        all_logs.extend(logs_list)

    return all_accs, all_logs


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
    train_distributed("warmup", gpus, 1, MuonConfig(), experiment_config)

    # print("Running Muon Experiments...")
    muon_accs, muon_logs = train_distributed(experiment_name, gpus, runs_per_gpu, MuonConfig(), experiment_config)

    print("Running SGD Experiments...")
    sgd_accs, sgd_logs = train_distributed(experiment_name, gpus, runs_per_gpu, SGDConfig(), experiment_config)

    # print("Running Adam Experiments...")
    adam_accs, adam_logs = train_distributed(experiment_name, gpus, runs_per_gpu, AdamConfig(), experiment_config)

    # print_aggregated_metrics("Muon", muon_accs, muon_logs)
    # print_aggregated_metrics("SGD", sgd_accs, sgd_logs)
    # print_aggregated_metrics("Adam", adam_accs, adam_logs)

    print(f"See https://wandb.ai/padlex/cifar10-airbench -> {experiment_name} for detailed results.")

if __name__ == "__main__":
    main()
