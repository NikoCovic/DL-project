from src.edge_of_stability.preconditioner import MuonPreconditioner
from src.edge_of_stability.networks import MLP
from src.edge_of_stability.datasets import CIFAR10Dataset
from src.edge_of_stability.hessian import Hessian
from src.edge_of_stability.preconditioner import fetch_preconditioner
from src.edge_of_stability.utils import TorchLinearOperator, spectral_norm, params_sum

from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Muon, Adam, RMSprop
import torch

from tqdm import tqdm
import matplotlib.pyplot as plt

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(42)

    dataset = CIFAR10Dataset(n_classes=10, 
                             n_samples_per_class=500, 
                             loss="mse", device=device)

    model = MLP(dataset.input_shape, 
                dataset.output_dim, 
                activation="tanh")
    
    optim = Muon(model.parameters(), lr=2e-3, weight_decay=0, nesterov=False)
    #optim = Adam(model.parameters(), lr=2e-3)
    #optim = RMSprop(model.parameters(), lr=2e-5)
    
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    n_epochs = 500

    loss_fn = nn.MSELoss()

    hessian = Hessian(model, next(iter(dataloader)), loss_fn, device=device)

    pbar = tqdm(range(n_epochs))

    measurements = []

    for epoch in pbar:
        hessian.update_params()

        for inputs, targets in dataloader:
            predictions = model(inputs.to(device))

            loss = loss_fn(predictions, targets.to(device))

            loss.backward()

            optim.step()

        preconditioner = fetch_preconditioner(optim, model)
        preconditioner_sqrt = preconditioner.pow(0.5)

        # Compute ||P - P^{1/2}P^{1/2}||_2
        def mv(v):
            Pv = preconditioner.dot(v, inplace=False)
            Psqv = preconditioner_sqrt.dot(v, inplace=True)
            PsqPsqv = preconditioner_sqrt.dot(Psqv, inplace=True)
            return params_sum(Pv, PsqPsqv, alpha=-1)
        operator = TorchLinearOperator(mv, list(model.parameters()))

        #s = hessian.commutativity_measure(preconditioner=preconditioner)

        s = hessian.spectral_norm(preconditioner=preconditioner)
        _, es = hessian.eigenvalues(preconditioner=preconditioner)
        e = es[0]

        pbar.set_description(f"sigma(PH) - rho(PH): {s - e:.2e}")

        measurements.append(s-e)

    plt.plot(measurements)
    plt.show()


if __name__ == "__main__":
    test()
        
        
