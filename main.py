from torch.optim import Muon, RMSprop
from torch.nn import MSELoss, CrossEntropyLoss
from torch.utils.data import DataLoader
import torch

from tqdm import tqdm
from time import time_ns
import json

import matplotlib.pyplot as plt
from dataclasses import asdict

from src.hessian import Hessian
from src.trackers import SharpnessTracker, SpectralNormTracker, EffSharpnessTracker, EffSpectralNormTracker, UpdateSharpnessTracker, UpdateSpectralNormTracker
from src.networks import MLP
from src.datasets import CIFAR10Dataset
from src.configs import *
from typing import Union, Literal, Annotated, Set
import tyro
from tyro.conf import arg


ValidOptim = Union[
    Literal["muon", "rmsprop"]
]
ValidModel = Union[
    Literal["mlp"]
]
ValidDataset = Union[
    Literal["cifar10"]
]


def main(optim:ValidOptim,
         model:ValidModel,
         dataset:ValidDataset,
         cifar10_config:Annotated[CIFAR10Config, arg(name="cifar10")],
         mlp_config:Annotated[MLPConfig, arg(name="mlp")],
         muon_config:Annotated[MuonConfig, arg(name="muon")],
         rmsprop_config:Annotated[RMSpropConfig, arg(name="rmsprop")],
         trackers:Set[Literal["sharpness", "spectral_norm", "eff_sharpness", "eff_spectral_norm", "update_sharpness", "update_spectral_norm"]],
         sharpness_config:Annotated[TrackerConfig, arg(name="sharpness")],
         spectral_norm_config:Annotated[TrackerConfig, arg(name="spectral_norm")],
         eff_sharpness_config:Annotated[TrackerConfig, arg(name="eff_sharpness")],
         eff_spectral_norm_config:Annotated[TrackerConfig, arg(name="eff_spectral_norm")],
         update_sharpness_config:Annotated[TrackerConfig, arg(name="update_sharpness")],
         update_spectral_norm_config:Annotated[TrackerConfig, arg(name="update_spectral_norm")],
         n_epochs:int=500,
         seed:int=42):
    
    torch.manual_seed(seed)
    
    # Construct the dataset
    dataset_name = dataset
    if dataset == "cifar10":
        dataset = CIFAR10Dataset(n_classes=cifar10_config.n_classes,
                                 n_samples_per_class=cifar10_config.n_samples_per_class,
                                 loss=cifar10_config.loss)
        loss = cifar10_config.loss
    
    # Create the dataloder
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        
    # Construct the loss
    if loss == "mse":
        loss_fn = MSELoss()
    elif loss == "ce":
        loss_fn = CrossEntropyLoss()
    
    # Construct the model
    model_name = model
    if model == "mlp":
        model = MLP(input_shape=dataset.input_shape, 
                    output_dim=dataset.output_dim, 
                    activation=mlp_config.activation, 
                    n_hidden=mlp_config.n_hidden, 
                    width=mlp_config.width, 
                    bias=mlp_config.bias)
        
    # Construct the optimizer
    optim_name = optim
    if optim == "muon":
        optim = Muon(model.parameters(),
                     lr=muon_config.lr,
                     momentum=muon_config.momentum,
                     weight_decay=muon_config.weight_decay,
                     nesterov=muon_config.nesterov)
    elif optim == "rmsprop":
        optim = RMSprop(model.parameters(),
                        lr=rmsprop_config.lr,
                        alpha=rmsprop_config.alpha,
                        eps=rmsprop_config.eps)
        
    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create the Hessian
    hessian = Hessian(model, next(iter(dataloader)), loss_fn, device=device)

    # Create the trackers
    tracker_names = trackers
    trackers = {}
    results = {}
    for tracker_name in tracker_names:
        if tracker_name == "sharpness":
            trackers[tracker_name] = SharpnessTracker(hessian, **asdict(sharpness_config))
            results[tracker_name] = asdict(sharpness_config)
        elif tracker_name == "spectral_norm":
            trackers[tracker_name] = SpectralNormTracker(hessian, **asdict(spectral_norm_config))
            results[tracker_name] = asdict(spectral_norm_config)
        elif tracker_name == "eff_sharpness":
            trackers[tracker_name] = EffSharpnessTracker(hessian, optim, model, **asdict(eff_sharpness_config))
            results[tracker_name] = asdict(eff_sharpness_config)
        elif tracker_name == "eff_spectral_norm":
            trackers[tracker_name] = EffSpectralNormTracker(hessian, optim, model, **asdict(eff_spectral_norm_config))
            results[tracker_name] = asdict(eff_spectral_norm_config)
        elif tracker_name == "update_sharpness":
            trackers[tracker_name] = UpdateSharpnessTracker(hessian, optim, model, **asdict(update_sharpness_config))
            results[tracker_name] = asdict(update_sharpness_config)
        elif tracker_name == "update_spectral_norm":
            trackers[tracker_name] = UpdateSpectralNormTracker(hessian, optim, model, **asdict(update_spectral_norm_config))
            results[tracker_name] = asdict(update_spectral_norm_config)
        
    # Begin the training loop
    pbar = tqdm(range(n_epochs))

    train_loss_history = []

    for epoch in pbar:

        # Update the Hessian parameters to the ones pre-epoch
        hessian.update_params()

        n = 0
        avg_train_loss = 0

        for i, (inputs, targets) in enumerate(dataloader):

            optim.zero_grad()

            y_pred = model(inputs.to(device))
            loss = loss_fn(targets.to(device), y_pred)

            loss.backward()
            n += 1

            optim.step()
        
            # Update average loss
            avg_train_loss += loss.item()

        # Store loss
        avg_train_loss /= n
        train_loss_history.append(avg_train_loss)

        # Update trackers and pbar
        text = f"train_loss: {avg_train_loss:.2e}"
        for tracker in trackers:
            trackers[tracker].update()
            #val = "-" if trackers[tracker].time <= trackers[tracker].n_warmup else f"{trackers[tracker].measurements[-1]:.2e}"
            #text += f"{tracker}: {val} | "
        pbar.set_description(text)

    # Plot the results
    fig, ax = plt.subplots(len(trackers)+1, squeeze=False)
    ax[0, 0].plot(train_loss_history, label="train_loss")
    ax[0, 0].legend()
    for i, tracker in enumerate(trackers):
        ax[i+1,0].plot(trackers[tracker].measurements, label=tracker)
        ax[i+1,0].legend()
    plt.show()

    # Save the results
    experiment_name = f"experiment-{time_ns()}"
    experiment_dir = f"experiments/{experiment_name}.json"

    with open(experiment_dir, "w") as f:
        results["optim"] = optim_name
        results["dataset"] = dataset_name
        results["model"] = model_name
        results["trackers"] = list(tracker_names)
        results["n_epochs"] = n_epochs
        results["seed"] = seed
        # Add the configs
        results["mlp"] = asdict(mlp_config)
        results["muon"] = asdict(muon_config)
        results["rmsprop"] = asdict(rmsprop_config)
        results["cifar10"] = asdict(cifar10_config)
        # Add the measurments results
        for tracker in trackers:
            results[tracker]["measurements"] = trackers[tracker].measurements

        json.dump(results, f)


if __name__ == "__main__":
    tyro.cli(main)
