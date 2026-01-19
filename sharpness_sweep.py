import torch.multiprocessing as mp
import wandb
import torch
from src.sharpness.airbench94_muon import CifarNet, train, SGDConfig, AdamConfig, VanillaMuonConfig, NormalizedMuonConfig

NUM_GPUS = 4
NUM_RUNS = 128
NUM_EPOCHS = 16


# 1. Define your sweep config (keep as you had it)
normalized_muon_sweep = {
    'method': 'bayes',
    'run_cap': NUM_RUNS,
    'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
    'parameters': {
        'muon_lr': {'min': 0.001, 'max': 0.05},
        'bias_lr': {'min': 0.001, 'max': 0.05},
        'head_lr': {'min': 0.05, 'max': 0.4},
        'lr_scheduler': {'value': False},
    }
}

vanilla_muon_sweep = {
    'method': 'bayes',
    'run_cap': NUM_RUNS,
    'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
    'parameters': {
        'muon_lr': {'min': 0.001, 'max': 0.05},
        'bias_lr': {'min': 0.001, 'max': 0.05},
        'head_lr': {'min': 0.05, 'max': 0.4},
        'lr_scheduler': {'value': False},
    }
}


sgd_sweep = {
    'method': 'bayes',
    'run_cap': NUM_RUNS,
    'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
    'parameters': {
        'bias_lr': {'min': 0.001, 'max': 0.5},
        'head_lr': {'min': 0.001, 'max': 0.5},
        'filter_lr': {'min': 0.001, 'max': 0.5},
    }
}


adam_sweep = {
    'method': 'bayes',
    'run_cap': NUM_RUNS,
    'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
    'parameters': {
        'bias_lr': {'min': 0.001, 'max': 0.5},
        'head_lr': {'min': 0.001, 'max': 0.5},
        'filter_lr': {'min': 0.0001, 'max': 0.5},
        'beta1': {'min': 0.8, 'max': 0.99},
        'beta2': {'min': 0.99, 'max': 0.9999},
    }
}


def sweep_worker(sweep_id, device_id, optimizer):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    model = CifarNet().to(f'cuda').to(memory_format=torch.channels_last)
    # model = torch.compile(model, mode="max-autotune")

    class ExperimentConfig:
        def __init__(self):
            self.dataset_path = "data/cifar10"
    experiment_config = ExperimentConfig()
    

    def train_wrapper():
        with wandb.init() as run:
            # Extract parameters from wandb.config
            acc = train(run, model, experiment_config, optimizer(**wandb.config), epochs=NUM_EPOCHS)
            wandb.log({"val_accuracy": acc})

    # Start the agent on this specific process
    wandb.agent(sweep_id, function=train_wrapper)


if __name__ == "__main__":

    mp.set_start_method('spawn', force=True)

    # sweep_id = wandb.sweep(sweep=normalized_muon_sweep, project="normalized-muon-tuning")
    # optimizer = NormalizedMuonConfig

    # sweep_id = wandb.sweep(sweep=vanilla_muon_sweep, project="vanilla-muon-tuning")
    # optimizer = VanillaMuonConfig

    # sweep_id = wandb.sweep(sweep=sgd_sweep, project="sgd-tuning")
    # optimizer = SGDConfig

    sweep_id = wandb.sweep(sweep=adam_sweep, project="adam-tuning")
    optimizer = AdamConfig
    ctx = mp.get_context('spawn')

    processes = []
    for i in range(NUM_GPUS):
        p = ctx.Process(target=sweep_worker, args=(sweep_id, i, optimizer))
        p.start()
        processes.append(p)

    # Wait for all workers to finish
    for p in processes:
        p.join()