from src.edge_of_stability.preconditioner import MuonPreconditioner
from src.edge_of_stability.networks import MLP
from src.edge_of_stability.datasets import CIFAR10Dataset
from src.edge_of_stability.hessian import Hessian
from src.edge_of_stability.preconditioner_factory import fetch_preconditioner
from src.edge_of_stability.utils import *

from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Muon, Adam, RMSprop
import torch
from src.sharpness.airbench94_muon import VanillaMuon

from tqdm import tqdm
import matplotlib.pyplot as plt
from math import sqrt

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(device)

    torch.manual_seed(42)

    dataset = CIFAR10Dataset(n_classes=10, 
                             n_samples_per_class=500, 
                             loss="mse", device=device)

    model = MLP(dataset.input_shape, 
                dataset.output_dim, 
                n_hidden=4,
                activation="tanh")
    
    lr_muon = 2e-3
    lr_adam = 2e-5
    lr_rmsprop = 2e-5
    lr = lr_adam
    
    optim = VanillaMuon(model.parameters(), lr=lr_muon, weight_decay=0, nesterov=False)
    #optim = Adam(model.parameters(), lr=lr_adam)
    #optim = RMSprop(model.parameters(), lr=lr_rmsprop)
    
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    n_epochs = 500

    loss_fn = nn.MSELoss()

    hessian = Hessian(model, next(iter(dataloader)), loss_fn, device=device)

    pbar = tqdm(range(n_epochs))

    n = sum([p.numel() for p in model.parameters()])

    print(f"n = {n}, 1/sqrt(n) = {1 / sqrt(n):.2e}")

    measurements = []

    for epoch in pbar:
        hessian.update_params()

        for inputs, targets in dataloader:
            predictions = model(inputs.to(device))

            loss = loss_fn(predictions, targets.to(device))

            loss.backward()

            optim.step()

        preconditioner = fetch_preconditioner(optim, model, hessian.params)
        preconditioner_sqrt = preconditioner.pow(0.5)

        # Compute ||P - P^{1/2}P^{1/2}||_2
        def mv(v):
            Pv = preconditioner.dot(v, inplace=False)
            Psqv = preconditioner_sqrt.dot(v, inplace=True)
            PsqPsqv = preconditioner_sqrt.dot(Psqv, inplace=True)
            return params_sum(Pv, PsqPsqv, alpha=-1)
        operator = TorchLinearOperator(mv, list(model.parameters()))

        #s = hessian.commutativity_measure(preconditioner=preconditioner)

        #s = hessian.spectral_norm(preconditioner=preconditioner)
        #_, es = hessian.eigenvalues(preconditioner=preconditioner)
        #e = es[0]

        #m = [optim.state[p]["exp_avg"] for p in optim.state]
        m = [p.grad for p in model.parameters() if p.requires_grad]
        Pm = preconditioner.dot(m)#params_normalize(preconditioner.dot(m))
        params_current = [p for p in model.parameters() if p.requires_grad]
        params_old = hessian.params
        params_diff = params_sum(params_current, params_old, alpha=-1)
        params_scale(params_diff, -1/lr, inplace=True)

        dist = params_norm(params_sum(Pm, params_diff, alpha=-1))
        dist /= params_norm(Pm) + params_norm(params_diff)
        dist = dist.cpu().item()

        cosine_dist = abs(params_dot_product(Pm, params_diff) / (params_norm(Pm) * params_norm(params_diff))).cpu().item()

        #q = hessian.rayleigh_quotient(Pm, preconditioner=preconditioner)

        #alignment = hessian.alignment(m, preconditioner=preconditioner)
        pbar.set_description(f"dist(Pm, diff): {dist:.2e}")

        measurements.append(dist)

    #thresh = fetch_threshold(optim, metric="eff_sharpness")

    plt.plot(measurements)
    #plt.hlines([thresh], xmin=0, xmax=n_epochs, colors="black", linestyles="--")
    plt.show()


if __name__ == "__main__":
    test()
        
        
