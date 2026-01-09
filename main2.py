from torch.optim import Muon, RMSprop
from torch.nn import MSELoss, CrossEntropyLoss
from torch.utils.data import DataLoader
import torch

from tqdm import tqdm

import matplotlib.pyplot as plt

from src.hessian import Hessian
from src.trackers import SharpnessTracker, EffSharpnessTracker, UpdateSharpnessTracker
from src.networks import MLP
from src.datasets import CIFAR10Dataset
from src.configs import *
from typing import Union, Literal, Annotated, Set
import tyro
from tyro.conf import subcommand, arg


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
         trackers:Set[Literal["sharpness", "spectral_norm", "eff_sharpness", "eff_spectral_norm", "update_sharpness", "update_spectral_norm", "train_loss"]],
         n_epochs:int=500,
         seed:int=42):
    
    torch.manual_seed(seed)
    
    # Construct the dataset
    if dataset == "cifar10":
        dataset = CIFAR10Dataset(n_classes=cifar10_config.n_classes,
                                 n_samples_per_class=cifar10_config.n_samples_per_class,
                                 loss=cifar10_config.loss)
        loss = cifar10_config.loss
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        
    # Construct the loss
    if loss == "mse":
        loss_fn = MSELoss()
    elif loss == "ce":
        loss_fn = CrossEntropyLoss()
    
    # Construct the model
    if model == "mlp":
        model = MLP(input_shape=dataset.input_shape, 
                    output_dim=dataset.output_dim, 
                    activation=mlp_config.activation, 
                    n_hidden=mlp_config.n_hidden, 
                    width=mlp_config.width, 
                    bias=mlp_config.bias)
        
    # Construct the optimizer
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
    for tracker_name in tracker_names:
        if tracker_name == "sharpness":
            trackers[tracker_name] = SharpnessTracker(hessian)
        elif tracker_name == "eff_sharpness":
            trackers[tracker_name] = EffSharpnessTracker(hessian, optim, model)
        elif tracker_name == "update_sharpness":
            trackers[tracker_name] = UpdateSharpnessTracker(hessian, optim, model)
        
    # Begin the training loop
    pbar = tqdm(range(n_epochs))
    for epoch in pbar:

        # Update the Hessian parameters to the ones pre-epoch
        hessian.update_params()

        for i, (inputs, targets) in enumerate(dataloader):

            optim.zero_grad()

            y_pred = model(inputs.to(device))
            loss = loss_fn(targets.to(device), y_pred)

            loss.backward()

            optim.step()

        # Update trackers and pbar
        text = ""
        for tracker in trackers:
            trackers[tracker].update()
            val = "-" if trackers[tracker].time <= trackers[tracker].n_warmup else f"{trackers[tracker].measurements[-1]:.2e}"
            text += f"{tracker}: {val} | "
        pbar.set_description(text)

    # Plot the results
    fig, ax = plt.subplots(len(trackers), squeeze=False)
    for i, tracker in enumerate(trackers):
        ax[0,i].plot(trackers[tracker].measurements)
    plt.show()

if __name__ == "__main__":
    tyro.cli(main)
