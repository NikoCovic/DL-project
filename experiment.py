from typing import final
import uuid
from airbench94_muon import CifarNet, train, MuonConfig, SGDConfig, CifarLoader, BatchNorm, AdamConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pyhessian import hessian
from itertools import islice
# import loss_landscapes
# import loss_landscapes.metrics as metrics
import torch.multiprocessing as mp
import numpy as np
import wandb
import time

import copy

EXPERIMENT_ID = str(uuid.uuid4())[:8]


def pyhessian_sharpness(model, loader, num_batches=10):
    """
    Computes relative sharpness by scaling the top Hessian eigenvalue 
    by the squared norm of the model parameters.
    """
    model = model._orig_mod if hasattr(model, "_orig_mod") else model
    criterion = torch.nn.CrossEntropyLoss()
    limited_loader = list(islice(loader, num_batches))
    
    hessian_comp = hessian(model, criterion, dataloader=limited_loader, cuda=True)
    lambda_max = hessian_comp.eigenvalues(top_n=1)[0][0]

    # 3. Compute Squared Frobenius Norm of all parameters
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total_norm_sq += torch.norm(p.data)**2
    
    # Calculate relative sharpness:
    relative_sharpness = lambda_max * total_norm_sq.item()

    return lambda_max, relative_sharpness

@torch.no_grad()
def get_sam_sharpness(model, loader, rho=0.05, num_batches=10):
    model.eval()
    device = next(model.parameters()).device
    total_sharpness = 0.0

    for batch_idx, (inputs, targets) in enumerate(iter(loader)):
        if batch_idx >= num_batches:
            break
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 1. Compute base loss and gradients
        with torch.enable_grad():
            outputs = model(inputs)
            base_loss = F.cross_entropy(outputs, targets)
            model.zero_grad()
            base_loss.backward()
        
        # 2. Compute gradient norm
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        grad_norm = torch.linalg.norm(
            torch.stack([torch.linalg.norm(g) for g in grads])
        ) + 1e-12
        
        # 3. Perturb parameters
        scale = rho / grad_norm
        epsilons = []
        for p in model.parameters():
            if p.grad is not None:
                eps = p.grad * scale  # Moved scale calculation out
                p.add_(eps)
                epsilons.append(eps)
        
        # 4. Compute perturbed loss
        adv_loss = F.cross_entropy(model(inputs), targets)
        
        # 5. Restore parameters
        for p, eps in zip(
            (p for p in model.parameters() if p.grad is not None), 
            epsilons
        ):  
            p.sub_(eps)
                
        total_sharpness += (adv_loss - base_loss).item()
        model.zero_grad()  # Clean up gradients

    return total_sharpness / min(num_batches, len(loader))

@torch.no_grad()
def get_samlike_sharpness(model, loader, rho=0.05, num_batches=10):
    model.eval()
    device = next(model.parameters()).device
    total_sharpness = 0.0

    loader_iter = iter(loader)

    for batch_idx, (inputs, targets) in enumerate(loader_iter):
        if batch_idx >= num_batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)

        # 1. Compute base loss and gradients
        with torch.enable_grad():
            outputs = model(inputs)
            base_loss = F.cross_entropy(outputs, targets)
            model.zero_grad()
            base_loss.backward()

        # 2. Compute TRUE gradient norm (flatten and concatenate)
        grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
        grad_norm = torch.linalg.norm(torch.cat(grads)) + 1e-12

        # 3. Perturb parameters
        scale = rho / grad_norm
        epsilons = []
        for p in model.parameters():
            if p.grad is not None:
                eps = p.grad * scale
                p.add_(eps)
                epsilons.append(eps)

        # 4. Compute perturbed loss
        adv_loss = F.cross_entropy(model(inputs), targets)

        # 5. Restore parameters
        for p, eps in zip(
            (p for p in model.parameters() if p.grad is not None),
            epsilons
        ):
            p.sub_(eps)

        # Normalize by gradient norm squared and rho squared
        # This approximates the Hessian eigenvalue in gradient direction
        sharpness = (adv_loss - base_loss) / (grad_norm**2 * rho**2 + 1e-12)
        total_sharpness += sharpness.item()
        
        model.zero_grad()

    return total_sharpness / min(num_batches, len(loader))


def train_and_log(experiment_name, model, optimizer_config):
    logs = []
    metric_loader = CifarLoader('dataloaders/cifar10', train=True, batch_size=1000)

    def callback_fn(epoch, model, training_accuracy, validation_accuracy):
        model.eval()

        log = {
            "epoch": epoch,
            "train_acc": training_accuracy,
            "val_acc": validation_accuracy,
            "gap": training_accuracy - validation_accuracy,
            "sam_sharpness": get_sam_sharpness(model, metric_loader),
            # "fisher_rao_norm": compute_fisher_rao_norm(model),
        }

        if epoch > 0:
            # print(f"Calculating Hessian for epoch {epoch}...")
            log["hessian"], log["relative_hessian"] = pyhessian_sharpness(model, metric_loader)
        
        wandb.log(log)
        logs.append(log)

        # print(f"[{name}] Epoch {epoch}  SAM Sharpness: {sam_sharp}  Hessian Top Eigenvalue: {hess_sharp}")
        model.train()
    
    start = time.time()
    wandb.init(project="cifar10-airbench", group=experiment_name, config=optimizer_config.represent())
    middle = time.time()

    final_acc = train("run", model, optimizer_config, callback=callback_fn, epochs=16)

    post = time.time()
    wandb.log({"tta_val_accuracy": final_acc, "tta_gap": logs[-1]["val_acc"] - final_acc})
    wandb.finish()
    end = time.time()

    wasted_time = (middle - start) + (end - post)
    
    print(f"Wasted Time: {wasted_time:.2f}s of {end - start:.2f}s total ({100.0 * wasted_time / (end - start):.2f}%)")

    return final_acc, logs


def worker(experiment_name, gpu_id, runs_per_gpu, optimizer_config):
    all_acc, all_logs = [], []
    torch.cuda.set_device(gpu_id)
    model = CifarNet().to(f'cuda:{gpu_id}').to(memory_format=torch.channels_last)
    model = torch.compile(model, mode="max-autotune")

    for run in tqdm(range(runs_per_gpu)) if gpu_id == 0 else range(runs_per_gpu):
        acc, logs = train_and_log(experiment_name, model, optimizer_config)
        all_acc.append(acc)
        all_logs.append(logs)
        
    return all_acc, all_logs


def train_distributed(experiment_name, gpus, runs_per_gpu, optimizer_config):

    if gpus == 1:
        return worker(experiment_name, 0, runs_per_gpu, optimizer_config)

    with mp.Pool(gpus) as pool:
        out = [pool.apply_async(worker, args=(experiment_name, gpu_id, runs_per_gpu, optimizer_config)) for gpu_id in range(gpus)]
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
    gpus = 1
    runs_per_gpu = 5
    experiment_name = f"experiment-{EXPERIMENT_ID}"
    # experiment_name = "hessian-every-epoch"

    print("Performing Warmup...")
    train_distributed("warmup", gpus, 1, MuonConfig())

    # print("Running Muon Experiments...")
    muon_accs, muon_logs = train_distributed(experiment_name, gpus, runs_per_gpu, MuonConfig())

    print("Running SGD Experiments...")
    sgd_accs, sgd_logs = train_distributed(experiment_name, gpus, runs_per_gpu, SGDConfig())

    # print("Running Adam Experiments...")
    adam_accs, adam_logs = train_distributed(experiment_name, gpus, runs_per_gpu, AdamConfig())

    # print_aggregated_metrics("Muon", muon_accs, muon_logs)
    # print_aggregated_metrics("SGD", sgd_accs, sgd_logs)
    # print_aggregated_metrics("Adam", adam_accs, adam_logs)

    print(f"See https://wandb.ai/padlex/cifar10-airbench -> {experiment_name} for detailed results.")

if __name__ == "__main__":
    main()
